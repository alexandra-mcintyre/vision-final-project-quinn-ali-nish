"""Single-frame solver with fixed focal length.

Camera: (pitch, h, yaw, cam_x, cam_y) — roll fixed at 0, f fixed at 1500.
Per boat: (x, y, yaw, heel)
Per mark: (x, y)
Pin: (x, y)

Regularization:
  - Camera velocity (pitch, h, yaw, cam_x, cam_y) — resist change from prev
  - Boat velocity — anchor at prev + velocity prediction
  - Boat speed limit — quadratic above max_speed
  - Boat heading-velocity agreement — scaled by speed²
  - Boat yaw rate — quadratic above 90°/s
  - Boat heel — linear pull to 0 + quadratic above 35°
  - Mark/pin position — resist movement (they're stationary)
  - Mark speed limit — quadratic above 0.1 m/s
"""

import numpy as np
from scipy.optimize import least_squares
import cv2

FIXED_F = 1500.0
HULL_L = 4.2
MAST_H = 5.2

# Use only mast-base + mast-top keypoints (ignore bow/stern detections).
# Yaw becomes unobservable from geometry alone when heel≈0 — relies on
# the heading-velocity regularizer to recover yaw from motion.
USE_MAST_ONLY = True

# Base rotation: world (+x right, +y forward, +z up) → camera (+x right, +y down, +z forward)
_R_BASE = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

BOAT_TEMPLATE = np.array([
    [-HULL_L / 3, 0, 0],       # bow
    [2 * HULL_L / 3, 0, 0],    # stern
    [0, 0, 0.7],               # mast base (boom height)
    [0, 0, 0.7 + MAST_H],     # mast top
], dtype=np.float64)


def _R(pitch, yaw):
    """World→camera rotation. Roll fixed at 0."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_pitch @ _R_BASE @ R_yaw


def _project(world_pts, R, cam_pos, cx, cy):
    """Pinhole projection. Returns (N, 2) pixel coords."""
    tvec = -R @ cam_pos
    pts_cam = (R @ world_pts.T).T + tvec
    z = pts_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xy = pts_cam[:, :2] / z
    return np.column_stack([FIXED_F * xy[:, 0] + cx, FIXED_F * xy[:, 1] + cy])


def _boat_world(x, y, yaw, heel):
    """Place boat template in world coords with yaw + heel."""
    tmpl = BOAT_TEMPLATE
    if abs(heel) > 1e-6:
        ch, sh = np.cos(heel), np.sin(heel)
        Rx = np.array([[1, 0, 0], [0, ch, -sh], [0, sh, ch]])
        tmpl = (Rx @ tmpl.T).T
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (Rz @ tmpl.T).T + np.array([x, y, 0])


def _bs_weight(meas):
    """Bow/stern weight based on geometric ratio."""
    mast_px = np.linalg.norm(meas[3] - meas[2])
    bs_px = np.linalg.norm(meas[0] - meas[1])
    if mast_px < 5:
        return 0.1
    ratio = bs_px / mast_px
    if ratio < 0.25 or ratio > 2.0:
        return 0.05
    return 0.3


def residuals(params, boats_meas, pin_uv, marks_uv,
              cx, cy, prev, dt=1.0 / 15):
    """Compute all residuals for one frame.

    params layout: [pitch, h, yaw, cam_x, cam_y,
                    pin_x?, pin_y?,
                    boat0_x, boat0_y, boat0_yaw, boat0_heel, ...,
                    mark0_x, mark0_y, ...]
    """
    pitch, h, yaw = params[0], params[1], params[2]
    cam_x, cam_y = params[3], params[4]
    R_mat = _R(pitch, yaw)
    cam_pos = np.array([cam_x, cam_y, h])
    idx = 5

    blocks = []

    pin_x = pin_y = None
    if pin_uv is not None:
        pin_x, pin_y = params[idx], params[idx + 1]
        idx += 2
        proj = _project(np.array([[pin_x, pin_y, 0]]), R_mat, cam_pos, cx, cy)[0]
        blocks.append(proj - pin_uv)

    boat_states = []
    for meas in boats_meas:
        bx, by, byaw, bheel = params[idx], params[idx+1], params[idx+2], params[idx+3]
        idx += 4
        boat_states.append((bx, by, byaw, bheel))
        world_pts = _boat_world(bx, by, byaw, bheel)
        if USE_MAST_ONLY:
            proj = _project(world_pts[2:], R_mat, cam_pos, cx, cy)
            blocks.append((proj - meas[2:]).ravel())
        else:
            proj = _project(world_pts, R_mat, cam_pos, cx, cy)
            diff = proj - meas
            w = _bs_weight(meas)
            weights = np.array([[w, w], [w, w], [1, 1], [1, 1]])
            blocks.append((diff * weights).ravel())

    mark_xys = []
    for uv in marks_uv:
        mx, my = params[idx], params[idx + 1]
        idx += 2
        mark_xys.append((mx, my))
        proj = _project(np.array([[mx, my, 0]]), R_mat, cam_pos, cx, cy)[0]
        blocks.append(proj - uv)

    if prev is not None:
        BLEND = 0.2
        CAM_BLEND = 1.0

        # Camera velocity — anchor at prev + full velocity → penalizes acceleration only
        for key, val, w in [("pitch", pitch, 250),
                            ("h", h, 25),
                            ("yaw", yaw, 500),
                            ("cam_x", cam_x, 50),
                            ("cam_y", cam_y, 50)]:
            pv = prev.get(key)
            if pv is not None:
                vel = prev.get(f"_{key}_vel", 0)
                anchor = pv + CAM_BLEND * vel
                blocks.append(np.array([(val - anchor) * w]))

        prev_boats = prev.get("boats", [])
        prev_boat_match = prev.get("_boat_match", [])
        for i, (bx, by, byaw, bheel) in enumerate(boat_states):
            if i >= len(prev_boat_match) or prev_boat_match[i] < 0:
                continue
            j = prev_boat_match[i]
            if j >= len(prev_boats):
                continue
            pb = prev_boats[j]

            # Constant-velocity prior: anchor at pb + full velocity = predicted
            # position; residual is acceleration. Heavy weight so boats keep
            # their current heading/speed unless detection strongly says otherwise.
            pvx = prev.get("_boat_vels", {}).get(j, (0, 0))
            anchor_x = pb["x"] + pvx[0]
            anchor_y = pb["y"] + pvx[1]
            blocks.append(np.array([(bx - anchor_x) * 40,
                                    (by - anchor_y) * 40]))

            vx, vy = bx - pb["x"], by - pb["y"]
            speed = np.hypot(vx, vy)
            prev_speed = np.hypot(pvx[0], pvx[1])
            blocks.append(np.array([(speed - prev_speed) * 20]))

            max_disp = 5.0 * dt
            excess = max(0, speed - max_disp)
            blocks.append(np.array([excess * excess * 50]))

            cross = np.sin(byaw) * vx - np.cos(byaw) * vy
            blocks.append(np.array([cross * speed * 5]))

            # Soft yaw-continuity anchor (wrap-safe: sin(Δ)=0 iff yaw≡prev mod 2π).
            # Essential in mast-only mode where yaw has no reprojection signal.
            prev_yaw = pb["yaw"]
            blocks.append(np.array([np.sin(byaw - prev_yaw) * 3]))

            dyaw = byaw - prev_yaw
            dyaw -= 2 * np.pi * round(dyaw / (2 * np.pi))
            max_dyaw = np.radians(90) * dt
            yaw_excess = max(0, abs(dyaw) - max_dyaw)
            blocks.append(np.array([yaw_excess * yaw_excess * 5]))

            prev_heel = pb.get("heel_rad", 0.0)
            blocks.append(np.array([(bheel - prev_heel) * 2.0]))

        for bx, by, byaw, bheel in boat_states:
            blocks.append(np.array([bheel * 1.0]))
            heel_excess = max(0, abs(bheel) - np.radians(45))
            blocks.append(np.array([heel_excess * heel_excess * 20]))

        # Pin: very heavy pull. At Huber's linear regime, loss per meter ≈
        # 2*W*disp, so W=5000 gives 10,000 loss/m — far above any reprojection
        # cost from a single bad detection.
        if pin_x is not None and prev.get("pin") is not None:
            pp = prev["pin"]
            blocks.append(np.array([(pin_x - pp[0]) * 5000,
                                    (pin_y - pp[1]) * 5000]))
            if prev.get("pin_mean") is not None:
                pm_a = prev["pin_mean"]
                blocks.append(np.array([(pin_x - pm_a[0]) * 2000,
                                        (pin_y - pm_a[1]) * 2000]))

        prev_marks = prev.get("mark_xys", [])
        prev_mark_means = prev.get("mark_means", [])
        for i, (mx, my) in enumerate(mark_xys):
            if i < len(prev_marks) and prev_marks[i] is not None:
                pm = prev_marks[i]
                blocks.append(np.array([(mx - pm[0]) * 5000,
                                        (my - pm[1]) * 5000]))
                if i < len(prev_mark_means) and prev_mark_means[i] is not None:
                    ma = prev_mark_means[i]
                    blocks.append(np.array([(mx - ma[0]) * 2000,
                                            (my - ma[1]) * 2000]))

    return np.concatenate(blocks)


def solve_frame(boats_meas, pin_uv, marks_uv, img_shape,
                prev=None, yaw_delta=0.0):
    """Solve one frame. Returns result dict + prev state for next frame."""
    H_img, W_img = img_shape
    cx, cy = W_img / 2.0, H_img / 2.0

    if prev is not None:
        pitch0 = prev["pitch"]
        h0 = max(prev["h"], 10.0)
        yaw0 = prev["yaw"] + yaw_delta
        cx0 = prev["cam_x"]
        cy0 = prev["cam_y"]
    else:
        pitch0 = 0.3
        h0 = 20.0
        yaw0 = 0.0
        cx0, cy0 = 0.0, 0.0

    R0 = _R(pitch0, yaw0)
    cam0 = np.array([cx0, cy0, h0])
    K0 = np.array([[FIXED_F, 0, cx], [0, FIXED_F, cy], [0, 0, 1]])
    K0_inv = np.linalg.inv(K0)
    rvec0 = cv2.Rodrigues(R0)[0].ravel()
    tvec0 = (-R0 @ cam0).ravel()

    p0 = [pitch0, h0, yaw0, cx0, cy0]

    if pin_uv is not None:
        if prev and prev.get("pin"):
            p0.extend(prev["pin"])
        else:
            ray = R0.T @ (K0_inv @ np.array([pin_uv[0], pin_uv[1], 1]))
            if abs(ray[2]) > 1e-6:
                s = -cam0[2] / ray[2]
                gnd = cam0 + s * ray
                p0.extend([float(gnd[0]), float(gnd[1])])
            else:
                p0.extend([cx0, cy0 + 50])

    prev_boats = prev.get("boats", []) if prev else []
    prev_meas = prev.get("_boats_meas", []) if prev else []
    boat_match = []
    used = set()
    match_idxs = (2, 4) if USE_MAST_ONLY else (0, 4)
    for meas in boats_meas:
        matched = -1
        if prev_boats and prev_meas:
            dists = []
            for j, pm in enumerate(prev_meas):
                if j in used or j >= len(prev_boats):
                    dists.append(float("inf"))
                else:
                    d = float(np.mean([np.linalg.norm(np.array(pm[k]) - meas[k])
                                       for k in range(*match_idxs)]))
                    dists.append(d)
            j = int(np.argmin(dists))
            if dists[j] < 30:
                pb = prev_boats[j]
                p0.extend([pb["x"], pb["y"], pb["yaw"], pb.get("heel_rad", 0)])
                matched = j
                used.add(j)
        if matched < 0:
            ray = R0.T @ (K0_inv @ np.array([meas[2][0], meas[2][1], 1]))
            if abs(ray[2]) > 1e-6:
                s = -cam0[2] / ray[2]
                gnd = cam0 + s * ray
                p0.extend([float(gnd[0]), float(gnd[1]), 0.0, 0.0])
            else:
                p0.extend([cx0, cy0 + 50, 0.0, 0.0])
        boat_match.append(matched)

    for uv in marks_uv:
        ray = R0.T @ (K0_inv @ np.array([uv[0], uv[1], 1]))
        if abs(ray[2]) > 1e-6:
            s = -cam0[2] / ray[2]
            gnd = cam0 + s * ray
            p0.extend([float(gnd[0]), float(gnd[1])])
        else:
            p0.extend([cx0, cy0 + 50])

    p0 = np.array(p0, dtype=np.float64)

    lb = np.full_like(p0, -np.inf)
    ub = np.full_like(p0, np.inf)
    lb[0] = 0.01; ub[0] = np.pi / 2 - 0.01
    lb[1] = 10.0; ub[1] = 100.0
    lb[2] = yaw0 - np.radians(5); ub[2] = yaw0 + np.radians(5)

    p0 = np.clip(p0, lb, ub)

    prev_for_reg = None
    if prev is not None:
        prev_for_reg = dict(prev)
        prev_for_reg["_boat_match"] = boat_match

    result = least_squares(
        residuals, p0,
        args=(boats_meas, pin_uv, marks_uv, cx, cy, prev_for_reg),
        bounds=(lb, ub), loss="huber", f_scale=2.0,
        max_nfev=50, verbose=0,
    )

    p = result.x
    pitch, h, yaw_s = float(p[0]), float(p[1]), float(p[2])
    cam_x_s, cam_y_s = float(p[3]), float(p[4])
    idx = 5

    pin_xy = None
    if pin_uv is not None:
        pin_xy = (float(p[idx]), float(p[idx + 1]))
        idx += 2

    boats = []
    R_out = _R(pitch, yaw_s)
    cam_out = np.array([cam_x_s, cam_y_s, h])
    for i, meas in enumerate(boats_meas):
        bx, by, byaw, bheel = float(p[idx]), float(p[idx+1]), float(p[idx+2]), float(p[idx+3])
        idx += 4
        byaw = (byaw + np.pi) % (2 * np.pi) - np.pi
        bheel = (bheel + np.pi) % (2 * np.pi) - np.pi
        world_pts = _boat_world(bx, by, byaw, bheel)
        if USE_MAST_ONLY:
            proj = _project(world_pts[2:], R_out, cam_out, cx, cy)
            err = float(np.linalg.norm(proj - meas[2:], axis=1).mean())
        else:
            proj = _project(world_pts, R_out, cam_out, cx, cy)
            err = float(np.linalg.norm(proj - meas, axis=1).mean())
        boats.append({"x": bx, "y": by, "yaw": byaw,
                      "heel_rad": bheel, "heel_deg": float(np.degrees(bheel)),
                      "err": err})

    mark_xys_out = []
    for uv in marks_uv:
        mx, my = float(p[idx]), float(p[idx + 1])
        idx += 2
        mark_xys_out.append((mx, my))

    cam_vels = {}
    if prev:
        for key, val in [("pitch", pitch), ("h", h), ("yaw", yaw_s),
                         ("cam_x", cam_x_s), ("cam_y", cam_y_s)]:
            pv = prev.get(key)
            if pv is not None:
                cam_vels[f"_{key}_vel"] = val - pv

    boat_vels = {}
    for i, b in enumerate(boats):
        j = boat_match[i]
        if j >= 0 and prev_boats and j < len(prev_boats):
            pb = prev_boats[j]
            boat_vels[j] = (b["x"] - pb["x"], b["y"] - pb["y"])

    rms = float(result.cost ** 0.5 / max(len(boats_meas), 1))

    # Running-mean accumulators (EMA). Carry forward prev's means even when
    # this frame has no detection — means are session-level, not frame-level.
    EMA = 0.02
    prev_pin_mean = prev.get("pin_mean") if prev else None
    if pin_xy is not None:
        if prev_pin_mean is None:
            pin_mean = pin_xy
        else:
            pin_mean = (prev_pin_mean[0] * (1 - EMA) + pin_xy[0] * EMA,
                        prev_pin_mean[1] * (1 - EMA) + pin_xy[1] * EMA)
    else:
        pin_mean = prev_pin_mean

    prev_mmeans = prev.get("mark_means", []) if prev else []
    mark_means: list[tuple[float, float] | None] = []
    for i, (mx, my) in enumerate(mark_xys_out):
        pm = prev_mmeans[i] if i < len(prev_mmeans) else None
        if pm is None:
            mark_means.append((mx, my))
        else:
            mark_means.append((pm[0] * (1 - EMA) + mx * EMA,
                               pm[1] * (1 - EMA) + my * EMA))
    # Carry forward any prev means beyond current count (marks that disappeared)
    for i in range(len(mark_xys_out), len(prev_mmeans)):
        mark_means.append(prev_mmeans[i])

    out = {
        "pitch": pitch, "h": h, "yaw": yaw_s,
        "cam_x": cam_x_s, "cam_y": cam_y_s,
        "pin": pin_xy,
        "pin_mean": pin_mean,
        "boats": boats,
        "marks": [{"x": mx, "y": my} for mx, my in mark_xys_out],
        "mark_xys": mark_xys_out,
        "mark_means": mark_means,
        "rms": rms,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "_boats_meas": [m.tolist() for m in boats_meas],
        "_boat_vels": boat_vels,
        **cam_vels,
    }
    return out
