"""Solve a single frame's camera pose from detected landmarks.

Repurposes solver_clean.solve_frame, which models:
  - Boat template in world coords:
        bow at (-HULL_L/3, 0, 0)
        stern at (2*HULL_L/3, 0, 0)
        mast_base at (0, 0, 0.7)        (boom height)
        mast_top  at (0, 0, 0.7 + MAST_H)
    → bow/stern lie on the z=0 plane (horizontal hull)
    → mast_base→mast_top is purely vertical
  - Boat 4-DOF: x, y, yaw, heel
  - Camera 5-DOF: pitch, height, yaw, cam_x, cam_y
                  (roll fixed at 0, focal length fixed = FIXED_F = 1500)

Inputs:
  --predictions     anchor_4fps_2min_predictions.json (cache from
                    render_dots.py — has per-frame solved boats)
  --frame           frame stem to pick (defaults to one with the
                    most boats so the solve is well-conditioned)
  --frames-dir      where the original frames live (used to read the
                    image and write the overlay)
  --img-w / --img-h frame dimensions (defaults to actual file)

Outputs:
  camera_solve_<stem>.json   solved camera + per-boat parameters
  camera_solve_<stem>.jpg    overlay: original frame with the
                             reprojected 3D boat models drawn

Usage:
  python3 solve_camera_from_frame.py
  python3 solve_camera_from_frame.py --frame frame_0500_f007620
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

# Use solver_clean's geometry constants and helpers but write our
# own residual/solve so we can let focal length be a free parameter
# (solver_clean fixes focal at FIXED_F).
import solver_clean
solver_clean.USE_MAST_ONLY = False
from solver_clean import (
    solve_frame as solve_frame_fixed_focal,
    _boat_world, _R, BOAT_TEMPLATE,
    FIXED_F, HULL_L, MAST_H,
)


def _project_f(world_pts: np.ndarray, R: np.ndarray,
                 cam_pos: np.ndarray, focal: float,
                 cx: float, cy: float) -> np.ndarray:
    """Pinhole projection with explicit focal length."""
    tvec = -R @ cam_pos
    pts_cam = (R @ world_pts.T).T + tvec
    z = pts_cam[:, 2:3]
    # Clamp z to a 1.0 m floor: projection still degrades when
    # boats approach the camera plane, but residuals stay
    # bounded (~focal × X_max metres rather than billions of
    # px), so the FD Jacobian doesn't blow up.
    z = np.where(z < 1.0, 1.0, z)
    xy = pts_cam[:, :2] / z
    return np.column_stack([
        focal * xy[:, 0] + cx,
        focal * xy[:, 1] + cy,
    ])


def _residuals_free_f(params: np.ndarray, boats_meas: list,
                        cx: float, cy: float,
                        active_mask: np.ndarray | None = None,
                        heel_prior_weight: float = 0.0,
                        heel_deadzone_rad: float = 0.0,
                        focal_prior_target: float | None = None,
                        focal_prior_weight: float = 0.0,
                        kpt_weights: tuple = (1.0, 1.0, 1.0, 1.0),
                        ) -> np.ndarray:
    """Residuals with focal as a free parameter.

    Layout:
      [pitch, h, yaw, cam_x, cam_y, focal,
       b0_x, b0_y, b0_yaw, b0_heel,
       b1_x, b1_y, b1_yaw, b1_heel, ...]

    When active_mask (n_boats, 4) is provided, only those keypoints
    contribute to the reprojection residual — used for iterative-drop
    refinement.

    heel_prior_weight: px-equivalent penalty per radian of heel; pulls
    each boat's heel toward 0 when the image data is weak/ambiguous.
    Set to 0 to disable.
    """
    pitch, h, yaw = params[0], params[1], params[2]
    cam_x, cam_y = params[3], params[4]
    focal = params[5]
    R = _R(pitch, yaw)
    cam_pos = np.array([cam_x, cam_y, h])
    blocks = []
    idx = 6
    n = len(boats_meas)
    heel_residuals = []
    for bi, meas in enumerate(boats_meas):
        bx, by, byaw, bheel = (
            params[idx], params[idx + 1],
            params[idx + 2], params[idx + 3],
        )
        idx += 4
        world_pts = _boat_world(bx, by, byaw, bheel)
        proj = _project_f(world_pts, R, cam_pos, focal, cx, cy)
        for lmi in range(4):
            if active_mask is None or active_mask[bi, lmi]:
                w = float(kpt_weights[lmi])
                blocks.append((proj[lmi] - meas[lmi]) * w)
        if heel_prior_weight > 0:
            # Deadzone: no penalty within ±heel_deadzone_rad, then
            # linear pull back toward the deadzone edge above it.
            if heel_deadzone_rad > 0:
                clipped = max(-heel_deadzone_rad,
                                 min(heel_deadzone_rad, bheel))
                heel_residuals.append(
                    heel_prior_weight * (bheel - clipped)
                )
            else:
                heel_residuals.append(heel_prior_weight * bheel)
    if heel_residuals:
        blocks.append(np.array(heel_residuals))
    if (focal_prior_target is not None
            and focal_prior_weight > 0):
        blocks.append(np.array([
            (params[5] - focal_prior_target)
            * focal_prior_weight]))
    return np.concatenate(blocks) if blocks else np.zeros(0)


def _per_point_errors(params: np.ndarray, boats_meas: list,
                          cx: float, cy: float) -> np.ndarray:
    """Return (n_boats, 4) per-keypoint pixel error magnitude."""
    pitch, h, yaw = params[0], params[1], params[2]
    cam_x, cam_y = params[3], params[4]
    focal = params[5]
    R = _R(pitch, yaw)
    cam_pos = np.array([cam_x, cam_y, h])
    n = len(boats_meas)
    errs = np.zeros((n, 4), dtype=np.float64)
    idx = 6
    for bi, meas in enumerate(boats_meas):
        bx, by, byaw, bheel = (
            params[idx], params[idx + 1],
            params[idx + 2], params[idx + 3],
        )
        idx += 4
        world_pts = _boat_world(bx, by, byaw, bheel)
        proj = _project_f(world_pts, R, cam_pos, focal, cx, cy)
        errs[bi] = np.linalg.norm(proj - meas, axis=1)
    return errs


def solve_frame_free_focal(
    boats_meas: list, img_shape: tuple,
    focal_init: float = FIXED_F,
    focal_min: float = 1.0, focal_max: float = 1e6,
    pitch_init: float = 0.3, h_init: float = 20.0,
    h_min: float = 5.0, h_max: float = 200.0,
    cam_xy_min: float = -200.0, cam_xy_max: float = 200.0,
    cam_y_max: float = 200.0,
    active_mask: np.ndarray | None = None,
    p_init: np.ndarray | None = None,
    heel_prior_weight: float = 10.0,
    heel_deadzone_rad: float = 0.0,
    focal_prior_target: float | None = None,
    focal_prior_weight: float = 0.0,
    kpt_weights: tuple = (1.0, 1.0, 1.0, 1.0),
):
    """Solve a single frame with focal as a free 6th camera param.

    Needs at least 2 boats with full 4-keypoint detections to be
    over-determined (5 cam + 1 focal + 4 per-boat = 6 + 4N unknowns;
    8N residuals). Recommended: 3+ boats for a stable fit.

    active_mask (n_boats, 4) bool excludes specific keypoints from
    the residual (used by iterative-drop refinement). p_init
    optionally warm-starts the parameter vector from a prior solve.
    """
    H_img, W_img = img_shape
    cx, cy = W_img / 2.0, H_img / 2.0
    n = len(boats_meas)

    if p_init is not None:
        p0 = np.array(p_init, dtype=np.float64).copy()
    else:
        R0 = _R(pitch_init, 0.0)
        K0 = np.array(
            [[focal_init, 0, cx], [0, focal_init, cy], [0, 0, 1]],
            dtype=np.float64,
        )
        K0_inv = np.linalg.inv(K0)
        cam0 = np.array([0.0, 0.0, h_init])
        p0 = [pitch_init, h_init, 0.0, 0.0, 0.0, focal_init]
        for meas in boats_meas:
            base_uv = meas[2]
            ray = R0.T @ (K0_inv
                            @ np.array([base_uv[0], base_uv[1], 1.0]))
            if abs(ray[2]) > 1e-6:
                s = -cam0[2] / ray[2]
                gnd = cam0 + s * ray
                p0.extend([float(gnd[0]), float(gnd[1]), 0.0, 0.0])
            else:
                p0.extend([0.0, 50.0, 0.0, 0.0])
        p0 = np.array(p0, dtype=np.float64)

    # Bounds. Focal range, pitch and height physical bounds, yaw
    # unconstrained for an isolated single-frame solve, boat heel
    # capped at ±60° to reject the 39° -> 80° drift case.
    lb = np.full_like(p0, -np.inf)
    ub = np.full_like(p0, np.inf)
    lb[0] = 0.01;       ub[0] = np.pi / 2 - 0.01
    lb[1] = h_min;      ub[1] = h_max
    lb[2] = -np.pi;     ub[2] = np.pi
    lb[3] = cam_xy_min; ub[3] = cam_xy_max
    lb[4] = cam_xy_min; ub[4] = min(cam_xy_max, cam_y_max)
    lb[5] = max(1.0, focal_min); ub[5] = focal_max
    for k in range(n):
        offset = 6 + 4 * k
        lb[offset + 3] = np.radians(-60.0)
        ub[offset + 3] = np.radians(60.0)
    p0 = np.clip(p0, lb, ub)

    # Sparsity pattern for the Jacobian: reprojection of boat i
    # only depends on cam params (0..5) and boat-i's own block
    # (6+4i .. 6+4i+3). Heel residual on boat i depends only on
    # params[6+4i+3]. Focal prior depends only on params[5].
    # scipy's FD Jacobian uses this to evaluate columns in
    # groups that don't share non-zero rows, cutting the number
    # of forward calls per Jac from O(p) to ~p/3 in practice.
    n_kp_active = (int(active_mask.sum()) if active_mask is not None
                            else 4 * n)
    n_heel = (n if heel_prior_weight > 0 else 0)
    n_focal = (1 if focal_prior_target is not None
                       and focal_prior_weight > 0 else 0)
    n_resid = 2 * n_kp_active + n_heel + n_focal
    n_params = len(p0)
    jac_sparsity = np.zeros((n_resid, n_params), dtype=bool)
    row = 0
    for bi in range(n):
        for lmi in range(4):
            if active_mask is None or active_mask[bi, lmi]:
                for k in range(6):
                    jac_sparsity[row:row + 2, k] = True
                bo = 6 + 4 * bi
                for k in range(bo, bo + 4):
                    jac_sparsity[row:row + 2, k] = True
                row += 2
    if heel_prior_weight > 0:
        for bi in range(n):
            jac_sparsity[row, 6 + 4 * bi + 3] = True
            row += 1
    if n_focal:
        jac_sparsity[row, 5] = True
        row += 1
    result = least_squares(
        _residuals_free_f, p0,
        args=(boats_meas, cx, cy, active_mask,
                heel_prior_weight, heel_deadzone_rad,
                focal_prior_target, focal_prior_weight,
                kpt_weights),
        bounds=(lb, ub),
        jac_sparsity=jac_sparsity,
        loss="linear", max_nfev=200, verbose=0,
    )

    p = result.x
    pitch_out = float(p[0]); h_out = float(p[1]); yaw_out = float(p[2])
    cam_x_out = float(p[3]); cam_y_out = float(p[4])
    focal_out = float(p[5])
    R_out = _R(pitch_out, yaw_out)
    cam_out = np.array([cam_x_out, cam_y_out, h_out])

    boats_out = []
    for k in range(n):
        offset = 6 + 4 * k
        bx = float(p[offset]); by = float(p[offset + 1])
        byaw = float(p[offset + 2]); bheel = float(p[offset + 3])
        byaw = (byaw + np.pi) % (2 * np.pi) - np.pi
        bheel = (bheel + np.pi) % (2 * np.pi) - np.pi
        world_pts = _boat_world(bx, by, byaw, bheel)
        proj = _project_f(world_pts, R_out, cam_out, focal_out, cx, cy)
        err = float(np.linalg.norm(proj - boats_meas[k],
                                       axis=1).mean())
        boats_out.append({"x": bx, "y": by, "yaw": byaw,
                            "heel_rad": bheel,
                            "heel_deg": float(np.degrees(bheel)),
                            "err": err})

    rms = float(result.cost ** 0.5 / max(n, 1))
    return {
        "pitch": pitch_out, "h": h_out, "yaw": yaw_out,
        "cam_x": cam_x_out, "cam_y": cam_y_out,
        "focal": focal_out,
        "boats": boats_out,
        "rms": rms,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "params": p,
    }


def solve_iterative_drop(boats_meas: list, img_shape: tuple,
                            err_thresh: float = 15.0,
                            err_thresh_bow_stern: float | None
                                = None,
                            min_active_per_boat: int = 2,
                            heel_prior_weight: float = 0.0,
                            heel_deadzone_rad: float = 0.0,
                            **solve_kwargs):
    """Solve, then iteratively drop the worst-fit keypoint and
    refit until every active keypoint's pixel error is <= err_thresh.

    The drop loop runs with heel_prior_weight=0 (purely
    reprojection-driven decisions, so we don't drop a tip just
    because the prior wants heel=0). After convergence, a final
    refit applies the requested heel_prior_weight — that final pass
    only collapses heels that were already weakly constrained by
    the surviving data.
    """
    n = len(boats_meas)
    active_mask = np.ones((n, 4), dtype=bool)
    dropped_log: list[dict] = []
    landmark_names = ["bow", "stern", "base", "tip"]

    # Per-landmark threshold vector: [bow, stern, base, tip].
    # If err_thresh_bow_stern is given, bow/stern use it; else
    # all four landmarks share err_thresh.
    bs = (err_thresh if err_thresh_bow_stern is None
              else err_thresh_bow_stern)
    thresh_per_kpt = np.array(
        [bs, bs, err_thresh, err_thresh], dtype=np.float64)

    drop_kwargs = dict(solve_kwargs)
    drop_kwargs["heel_prior_weight"] = 0.0
    p_warm = None
    max_drops = 4 * n - 2 * n
    for it in range(max_drops + 1):
        result = solve_frame_free_focal(
            boats_meas, img_shape,
            active_mask=active_mask,
            p_init=p_warm,
            **drop_kwargs,
        )
        p_warm = result["params"]
        errs = _per_point_errors(
            result["params"], boats_meas,
            img_shape[1] / 2.0, img_shape[0] / 2.0,
        )
        excess = errs - thresh_per_kpt[None, :]
        masked = np.where(active_mask, excess, -np.inf)
        worst_excess = float(masked.max())
        if worst_excess <= 0.0:
            worst_err = float(np.where(active_mask, errs,
                                                    -np.inf).max())
            print(f"  iter {it+1}: max active err {worst_err:.2f}"
                  f" px <= per-kpt thresh "
                  f"(bow/stern={bs}, base/tip={err_thresh})"
                  f" — done")
            break
        bi, lmi = np.unravel_index(masked.argmax(), masked.shape)
        bi = int(bi); lmi = int(lmi)
        worst = float(errs[bi, lmi])
        if active_mask[bi].sum() <= min_active_per_boat:
            print(f"  iter {it+1}: worst is boat {bi}.{landmark_names[lmi]}"
                  f" ({worst:.2f} px) but boat already at "
                  f"min_active_per_boat={min_active_per_boat}; "
                  f"stopping")
            break
        active_mask[bi, lmi] = False
        dropped_log.append({
            "order": len(dropped_log) + 1,
            "boat": bi,
            "landmark": landmark_names[lmi],
            "err_px": worst,
        })
        print(f"  iter {it+1}: dropped boat{bi}.{landmark_names[lmi]} "
              f"(err {worst:.2f} px > "
              f"thresh {thresh_per_kpt[lmi]:.0f}); "
              f"refitting "
              f"({active_mask.sum()} active points remain)")

    if heel_prior_weight > 0:
        print(f"  final refit with heel_prior_weight="
              f"{heel_prior_weight}, deadzone="
              f"{np.degrees(heel_deadzone_rad):.1f}°")
        final_kwargs = dict(solve_kwargs)
        final_kwargs["heel_prior_weight"] = heel_prior_weight
        final_kwargs["heel_deadzone_rad"] = heel_deadzone_rad
        result = solve_frame_free_focal(
            boats_meas, img_shape,
            active_mask=active_mask,
            p_init=p_warm,
            **final_kwargs,
        )
        errs = _per_point_errors(
            result["params"], boats_meas,
            img_shape[1] / 2.0, img_shape[0] / 2.0,
        )

    result["active_mask"] = active_mask
    result["dropped_log"] = dropped_log
    result["per_point_errs"] = errs
    return result


def to_meas(boat: dict) -> np.ndarray | None:
    """Convert a {bow, stern, tip, base, clew} dict to the
    (4, 2) boats_meas row solver_clean expects: [bow, stern,
    mast_base, mast_top]."""
    needed = ["bow", "stern", "base", "tip"]
    if not all(k in boat for k in needed):
        return None
    return np.array([
        boat["bow"], boat["stern"], boat["base"], boat["tip"],
    ], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",
                    default="anchor_4fps_2min_predictions.json")
    ap.add_argument("--frames-dir",
                    default="anchor_4fps_2min_frames")
    ap.add_argument("--frame", default=None,
                    help="frame stem to solve (e.g. "
                          "frame_0500_f007620). Defaults to the "
                          "frame with the most boats.")
    ap.add_argument("--img-w", type=int, default=None)
    ap.add_argument("--img-h", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-viz", default=None)
    ap.add_argument("--free-focal", action="store_true", default=True,
                    help="estimate focal length as a 6th camera DOF "
                          "(default on; --fix-focal to disable)")
    ap.add_argument("--fix-focal", dest="free_focal",
                    action="store_false")
    ap.add_argument("--focal-init", type=float, default=FIXED_F,
                    help="initial focal length in pixels")
    ap.add_argument("--iterative", action="store_true", default=True,
                    help="iteratively drop the worst-fit keypoint "
                          "and refit until max active error is "
                          "below --err-thresh (default on)")
    ap.add_argument("--no-iterative", dest="iterative",
                    action="store_false")
    ap.add_argument("--err-thresh", type=float, default=15.0,
                    help="px error threshold for the iterative-drop "
                          "loop")
    ap.add_argument("--heel-deadzone-deg", type=float, default=20.0,
                    help="heel prior is zero within ±this many "
                          "degrees, then linear pull above. 0 = "
                          "classic pull-toward-zero prior. Default "
                          "20° lets boats heel naturally up to 20° "
                          "without penalty.")
    ap.add_argument("--heel-prior-weight", type=float, default=10.0,
                    help="soft prior pulling each boat's heel toward "
                          "0 (vertical mast). Units: px-equivalent "
                          "penalty per radian of heel. Default 10 → "
                          "~0.17 px / degree, so a 30° real heel "
                          "costs only ~5 px and stays solved if the "
                          "image data wants it. Set to 0 to disable.")
    ap.add_argument(
        "--all-tips", default="",
        help="optional inference detections JSON (e.g. "
              "oos_tipbase_detections.json from "
              "infer_tip_base_oos.py). Any tip in there for "
              "this frame that is missing bow/stern/base or "
              "marked dropped_reason is rendered as a gray X "
              "on the solve overlay.")
    args = ap.parse_args()

    cache = json.loads(Path(args.predictions).read_text())
    preds: dict = cache["predictions"]
    if not preds:
        print("no predictions in cache")
        sys.exit(1)

    if args.frame:
        stem = args.frame
        if stem not in preds:
            print(f"frame {stem!r} not in predictions; "
                  f"available: {list(preds)[:5]} ...")
            sys.exit(1)
    else:
        stem = max(preds, key=lambda k: len(preds[k]))
        print(f"auto-picked frame with most boats: "
              f"{stem} ({len(preds[stem])} boats)")

    if args.out_json is None:
        args.out_json = f"camera_solve_{stem}.json"
    if args.out_viz is None:
        args.out_viz = f"camera_solve_{stem}.jpg"

    img_path = Path(args.frames_dir) / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"image not found at {img_path}")
        sys.exit(1)
    H_img, W_img = img.shape[:2]
    if args.img_w:
        W_img = args.img_w
    if args.img_h:
        H_img = args.img_h
    print(f"image {W_img}x{H_img}")

    boats_meas = []
    for boat in preds[stem]:
        m = to_meas(boat)
        if m is not None:
            boats_meas.append(m)
    print(f"using {len(boats_meas)} boats with full 4-keypoint set")
    if len(boats_meas) < 1:
        print("no usable boats")
        sys.exit(1)

    dropped_log: list[dict] = []
    active_mask = np.ones((len(boats_meas), 4), dtype=bool)
    heel_deadzone_rad = np.radians(args.heel_deadzone_deg)
    if args.free_focal and args.iterative:
        print(f"\nrunning iterative-drop free-focal solver "
              f"(focal init {args.focal_init}, "
              f"err_thresh {args.err_thresh} px, "
              f"heel_prior_weight {args.heel_prior_weight}, "
              f"heel_deadzone {args.heel_deadzone_deg}°)...")
        result = solve_iterative_drop(
            boats_meas, img_shape=(H_img, W_img),
            err_thresh=args.err_thresh,
            focal_init=args.focal_init,
            heel_prior_weight=args.heel_prior_weight,
            heel_deadzone_rad=heel_deadzone_rad,
        )
        dropped_log = result["dropped_log"]
        active_mask = result["active_mask"]
        focal_solved = result["focal"]
    elif args.free_focal:
        print(f"\nrunning free-focal solver "
              f"(focal initialized at {args.focal_init})...")
        result = solve_frame_free_focal(
            boats_meas, img_shape=(H_img, W_img),
            focal_init=args.focal_init,
            heel_prior_weight=args.heel_prior_weight,
            heel_deadzone_rad=heel_deadzone_rad,
        )
        focal_solved = result["focal"]
    else:
        print(f"\nrunning solver_clean.solve_frame "
              f"(focal fixed at {FIXED_F})...")
        result = solve_frame_fixed_focal(
            boats_meas, pin_uv=None, marks_uv=[],
            img_shape=(H_img, W_img),
        )
        focal_solved = FIXED_F
    pitch = result["pitch"]
    h = result["h"]
    yaw = result["yaw"]
    cam_x = result["cam_x"]
    cam_y = result["cam_y"]
    boats_out = result["boats"]
    rms = result.get("rms", float("nan"))

    print("\nCAMERA POSE:")
    print(f"  pitch  = {np.degrees(pitch):7.2f}°  "
          f"({pitch:.4f} rad)")
    print(f"  yaw    = {np.degrees(yaw):7.2f}°  "
          f"({yaw:.4f} rad)")
    print(f"  height = {h:7.2f} m")
    print(f"  cam_xy = ({cam_x:.2f}, {cam_y:.2f}) m")
    if args.free_focal:
        print(f"  focal  = {focal_solved:7.1f} px (estimated)")
    else:
        print(f"  focal  = {focal_solved:7.1f} px (fixed)")
    print(f"  reprojection RMS = {rms:.2f} px")
    print(f"\nWORLD MODEL: hull_L = {HULL_L} m, mast_H = {MAST_H} m")

    print("\nPER-BOAT POSE:")
    for i, b in enumerate(boats_out):
        print(f"  boat {i}: xy=({b['x']:7.2f}, {b['y']:7.2f}) m  "
              f"yaw={np.degrees(b['yaw']):7.2f}°  "
              f"heel={b['heel_deg']:6.2f}°  "
              f"err={b['err']:.2f} px")

    if dropped_log:
        print("\nDROPPED POINTS (in drop order):")
        for rec in dropped_log:
            print(f"  #{rec['order']}: boat {rec['boat']} "
                  f"{rec['landmark']:<5}  "
                  f"err {rec['err_px']:.2f} px")
    Path(args.out_json).write_text(json.dumps({
        "frame": stem,
        "image_size": [W_img, H_img],
        "dropped_log": dropped_log,
        "active_mask": active_mask.tolist(),
        "camera": {
            "pitch_rad": pitch, "pitch_deg": float(np.degrees(pitch)),
            "yaw_rad": yaw, "yaw_deg": float(np.degrees(yaw)),
            "height_m": h,
            "cam_x_m": cam_x, "cam_y_m": cam_y,
            "focal_px": focal_solved,
            "focal_estimated": bool(args.free_focal),
        },
        "world_model": {
            "hull_length_m": HULL_L,
            "mast_height_m": MAST_H,
        },
        "boats": [
            {"x_m": b["x"], "y_m": b["y"],
              "yaw_rad": b["yaw"],
              "yaw_deg": float(np.degrees(b["yaw"])),
              "heel_deg": b["heel_deg"],
              "reproj_err_px": b["err"]}
            for b in boats_out
        ],
        "rms_px": rms,
    }, indent=2))
    print(f"\nwrote {args.out_json}")

    cx_pp, cy_pp = W_img / 2.0, H_img / 2.0
    R_cam = _R(pitch, yaw)
    cam_pos = np.array([cam_x, cam_y, h])

    drop_order_lookup: dict[tuple[int, str], int] = {
        (rec["boat"], rec["landmark"]): rec["order"]
        for rec in dropped_log
    }

    overlay = img.copy()
    for i, (meas, b) in enumerate(zip(boats_meas, boats_out)):
        world_pts = _boat_world(b["x"], b["y"], b["yaw"],
                                  b["heel_rad"])
        proj = _project_f(world_pts, R_cam, cam_pos,
                            focal_solved, cx_pp, cy_pp)
        for k, name in enumerate(["bow", "stern", "base", "tip"]):
            mx, my = meas[k]
            px, py = proj[k]
            is_dropped = not bool(active_mask[i, k])
            drop_order = drop_order_lookup.get((i, name))
            if is_dropped:
                ring_col = (60, 60, 220)
                size = 10
                cv2.line(overlay,
                          (int(mx) - size, int(my) - size),
                          (int(mx) + size, int(my) + size),
                          ring_col, 2, cv2.LINE_AA)
                cv2.line(overlay,
                          (int(mx) - size, int(my) + size),
                          (int(mx) + size, int(my) - size),
                          ring_col, 2, cv2.LINE_AA)
                if drop_order is not None:
                    cv2.putText(overlay,
                                  f"x{drop_order}",
                                  (int(mx) + 12, int(my) + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  ring_col, 1, cv2.LINE_AA)
            else:
                cv2.circle(overlay, (int(mx), int(my)), 8,
                            (220, 220, 60), 2, cv2.LINE_AA)
                cv2.circle(overlay, (int(px), int(py)), 6,
                            (60, 200, 245), -1, cv2.LINE_AA)
                cv2.line(overlay, (int(mx), int(my)),
                          (int(px), int(py)),
                          (180, 180, 180), 1, cv2.LINE_AA)
        bow_pix = proj[0].astype(int)
        stern_pix = proj[1].astype(int)
        base_pix = proj[2].astype(int)
        tip_pix = proj[3].astype(int)
        cv2.line(overlay, tuple(bow_pix), tuple(stern_pix),
                  (60, 200, 245), 2, cv2.LINE_AA)
        cv2.line(overlay, tuple(base_pix), tuple(tip_pix),
                  (60, 200, 245), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"#{i}",
                      (int(proj[3, 0]) + 6, int(proj[3, 1]) - 6),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                      (255, 255, 255), 1, cv2.LINE_AA)

    n_extra_tips = 0
    if args.all_tips and Path(args.all_tips).exists():
        all_dets = json.loads(Path(args.all_tips).read_text())
        candidates = (all_dets.get(f"{stem}.jpg")
                          or all_dets.get(stem)
                          or [])
        # Build a set of tip coords already drawn so we don't double-draw.
        drawn_tips: set[tuple[int, int]] = set()
        for meas in boats_meas:
            tx, ty = meas[3]   # tip is index 3
            drawn_tips.add((int(round(tx)), int(round(ty))))
        for d in candidates:
            tip = d.get("tip")
            if tip is None:
                continue
            tx, ty = float(tip[0]), float(tip[1])
            if (int(round(tx)), int(round(ty))) in drawn_tips:
                continue
            ix, iy = int(round(tx)), int(round(ty))
            cv2.drawMarker(overlay, (ix, iy),
                                  (160, 160, 160),
                                  cv2.MARKER_TILTED_CROSS,
                                  18, 2, cv2.LINE_AA)
            cv2.putText(overlay, "?", (ix + 8, iy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, "?", (ix + 8, iy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (200, 200, 200), 1, cv2.LINE_AA)
            n_extra_tips += 1

    cv2.rectangle(overlay, (0, 0), (W_img, 56), (0, 0, 0), -1)
    cv2.putText(overlay,
                  f"{stem}   pitch={np.degrees(pitch):.1f}deg  "
                  f"yaw={np.degrees(yaw):.1f}deg  h={h:.1f}m  "
                  f"f={focal_solved:.0f}px",
                  (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                  (255, 255, 255), 1, cv2.LINE_AA)
    extra_hud = (f"   +{n_extra_tips} dropped tips (gray X)"
                       if n_extra_tips else "")
    cv2.putText(overlay,
                  f"detected (cyan ring) vs reprojected "
                  f"(yellow filled)   RMS={rms:.2f}px"
                  f"{extra_hud}",
                  (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                  (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(args.out_viz, overlay)
    print(f"wrote {args.out_viz}")


if __name__ == "__main__":
    main()
