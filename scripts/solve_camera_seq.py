"""Per-frame camera + boat solver.

Each frame is solved independently with a free-focal least-squares
fit + iterative drop of high-error keypoints. No chain warm-start,
no global bundle adjustment.

For each frame writes to --out-dir:
    frame_NNNN.solve.json     camera + per-boat parameters
    frame_NNNN.solve.jpg      image-space overlay with heading arrows
    frame_NNNN.topdown.jpg    world-space topdown of boats + camera
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

from infer_tip_base_oos import (
    resize_for_dinov3, dinov3_features, tip_heatmap,
    cluster_tips, base_for_tip, bowstern_for_pair,
    cell_to_src,
)
from models import TileHead, TipAttnBaseHead, TipBaseBowSternAttnHead
from solver_clean import _R as solver_R, _boat_world, HULL_L
from solve_camera_from_frame import (
    _project_f, to_meas, solve_iterative_drop,
)


def load_models(device, tip_ckpt, base_ckpt, bowstern_ckpt,
                dinov3_model, dv3_stride):
    model = AutoModel.from_pretrained(dinov3_model).eval().to(device).half()
    model.embeddings.patch_embeddings.stride = (dv3_stride, dv3_stride)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)

    tk = torch.load(tip_ckpt, map_location=device, weights_only=False)
    tip = TileHead(in_dim=int(tk["in_dim"]),
                   hidden=int(tk.get("hidden", 64))).to(device).eval()
    tip.load_state_dict(tk["state_dict"])
    tip_win = int(tk.get("win", 16))

    bk = torch.load(base_ckpt, map_location=device, weights_only=False)
    base = TipAttnBaseHead(
        in_dim=int(bk["in_dim"]),
        hidden=int(bk.get("hidden", 128)),
        n_heads=int(bk["n_heads"]),
        n_layers=int(bk["n_layers"]),
        win_h=int(bk["win_h"]),
        win_w=int(bk["win_w"]),
        tip_row=int(bk["win_top"]),
        tip_col=int(bk["win_w"]) // 2).to(device).eval()
    base.load_state_dict(bk["state_dict"])

    bs = torch.load(bowstern_ckpt, map_location=device, weights_only=False)
    bsh = TipBaseBowSternAttnHead(
        in_dim=int(bs["in_dim"]),
        hidden=int(bs.get("hidden", 128)),
        n_heads=int(bs["n_heads"]),
        n_layers=int(bs["n_layers"]),
        win_h=int(bs["win_h"]),
        win_w=int(bs["win_w"]),
        tip_row=int(bs["win_top"]),
        tip_col=int(bs["win_w"]) // 2,
        n_out=int(bs.get("n_out", 2))).to(device).eval()
    bsh.load_state_dict(bs["state_dict"])
    return model, n_special, tip, tip_win, base, bk, bsh, bs


@torch.no_grad()
def detect_frame(img, model, n_special, tip_head, tip_win,
                 base_head, base_meta, bs_head, device,
                 max_long_side, dv3_stride, tile_stride,
                 sig_thresh, cluster_eps, min_peak_dist,
                 peak_rel_thresh, min_tip_base_px):
    rgb, sx_src, sy_src = resize_for_dinov3(img, max_long_side)
    feat = dinov3_features(model, rgb, device, dv3_stride, n_special)
    heat = tip_heatmap(feat, tip_head, device, tip_win, tile_stride)
    tips_g = cluster_tips(
        heat, sig_thresh=sig_thresh, cluster_eps=cluster_eps,
        min_peak_dist=min_peak_dist, peak_rel_thresh=peak_rel_thresh)
    win_w = int(base_meta["win_w"])
    win_h = int(base_meta["win_h"])
    win_top = int(base_meta["win_top"])
    boats, dropped = [], []
    for tx_g, ty_g, _ in tips_g:
        bx_g, by_g, bp = base_for_tip(
            feat, base_head, device, tx_g, ty_g, win_w, win_h, win_top)
        tx_s, ty_s = cell_to_src(tx_g, ty_g, dv3_stride, sx_src, sy_src)
        bx_s, by_s = cell_to_src(bx_g, by_g, dv3_stride, sx_src, sy_src)
        d_px = ((bx_s - tx_s) ** 2 + (by_s - ty_s) ** 2) ** 0.5
        if d_px < min_tip_base_px:
            dropped.append((tx_s, ty_s))
            continue
        bow_g, stn_g = bowstern_for_pair(
            feat, bs_head, device,
            tx_g, ty_g, bx_g, by_g, win_w, win_h, win_top)
        bow_xs, bow_ys = cell_to_src(bow_g[0], bow_g[1], dv3_stride, sx_src, sy_src)
        stn_xs, stn_ys = cell_to_src(stn_g[0], stn_g[1], dv3_stride, sx_src, sy_src)
        boats.append({
            "tip": [tx_s, ty_s], "base": [bx_s, by_s],
            "bow": [bow_xs, bow_ys], "stern": [stn_xs, stn_ys],
            "base_p": float(bp),
            "bow_p": float(bow_g[2]), "stern_p": float(stn_g[2]),
        })
    return boats, dropped


def render_topdown(p, n_boats, hud, out_path,
                   half_extent_m=80.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam_x, cam_y = float(p[3]), float(p[4])
    fig, ax = plt.subplots(figsize=(6, 6))

    # Boats: position dot + heading arrow
    for i in range(n_boats):
        bx = float(p[6 + 4 * i])
        by = float(p[6 + 4 * i + 1])
        byaw = float(p[6 + 4 * i + 2])
        # Bow direction in world space (BOAT_TEMPLATE puts bow at -x).
        forward = np.array([-np.cos(byaw), -np.sin(byaw)])
        ax.scatter(bx, by, color="tab:orange", s=80, zorder=4)
        ax.arrow(bx, by, 4.0 * forward[0], 4.0 * forward[1],
                 head_width=1.5, head_length=1.5,
                 fc="tab:green", ec="tab:green", zorder=5,
                 length_includes_head=True)
        ax.annotate(f"#{i}", (bx, by),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="black")

    # Camera marker
    ax.scatter(cam_x, cam_y, color="black", marker="^", s=140, zorder=6)
    ax.annotate("cam", (cam_x, cam_y),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, color="black")

    # Center the plot on the boats so it stays readable when the camera
    # is far away.
    if n_boats > 0:
        bxs = [float(p[6 + 4 * i]) for i in range(n_boats)]
        bys = [float(p[6 + 4 * i + 1]) for i in range(n_boats)]
        cx_, cy_ = np.mean(bxs), np.mean(bys)
    else:
        cx_, cy_ = cam_x, cam_y
    ax.set_xlim(cx_ - half_extent_m, cx_ + half_extent_m)
    ax.set_ylim(cy_ - half_extent_m, cy_ + half_extent_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(hud, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=80)
    plt.close(fig)


def render_solve_overlay(img, p, boats_meas, dropped, hud_top, hud_bot,
                         out_path, active_mask=None):
    H_img, W_img = img.shape[:2]
    cx, cy = W_img / 2.0, H_img / 2.0
    pitch, h_cam, yaw_cam = p[0], p[1], p[2]
    cam_x, cam_y = p[3], p[4]
    focal = p[5]
    R = solver_R(pitch, yaw_cam)
    cam_pos = np.array([cam_x, cam_y, h_cam])
    overlay = img.copy()

    for i, meas in enumerate(boats_meas):
        bx, by = p[6 + 4 * i], p[6 + 4 * i + 1]
        byaw, bheel = p[6 + 4 * i + 2], p[6 + 4 * i + 3]
        wp = _boat_world(bx, by, byaw, bheel)
        proj = _project_f(wp, R, cam_pos, focal, cx, cy)
        for k in range(4):
            mx, my = meas[k]
            px, py = proj[k]
            is_active = active_mask is None or active_mask[i, k]
            if is_active:
                cv2.circle(overlay, (int(mx), int(my)), 8,
                           (220, 220, 60), 2, cv2.LINE_AA)
                cv2.circle(overlay, (int(px), int(py)), 6,
                           (60, 200, 245), -1, cv2.LINE_AA)
                cv2.line(overlay, (int(mx), int(my)),
                         (int(px), int(py)),
                         (180, 180, 180), 1, cv2.LINE_AA)
            else:
                sz = 9
                cv2.line(overlay, (int(mx) - sz, int(my) - sz),
                         (int(mx) + sz, int(my) + sz),
                         (60, 60, 220), 2, cv2.LINE_AA)
                cv2.line(overlay, (int(mx) - sz, int(my) + sz),
                         (int(mx) + sz, int(my) - sz),
                         (60, 60, 220), 2, cv2.LINE_AA)

        bow = proj[0]
        stern = proj[1]
        base = proj[2]
        tip = proj[3]
        cv2.line(overlay, tuple(bow.astype(int)), tuple(stern.astype(int)),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, tuple(base.astype(int)), tuple(tip.astype(int)),
                 (60, 200, 245), 2, cv2.LINE_AA)

        # Heading arrow: project the +x boat-frame axis into image space and
        # draw a long arrow from the mast base in that direction.
        head_world = _boat_world(bx, by, byaw, bheel)
        bow_w = head_world[0]
        forward = bow_w - np.array([bx, by, 0.7])
        forward_len = np.linalg.norm(forward[:2])
        if forward_len > 1e-6:
            forward_unit = forward[:2] / forward_len
            tip3 = np.array([bx + 6.0 * forward_unit[0],
                             by + 6.0 * forward_unit[1], 0.7])
            arrow_pts = _project_f(
                np.stack([np.array([bx, by, 0.7]), tip3]),
                R, cam_pos, focal, cx, cy)
            a0 = arrow_pts[0].astype(int)
            a1 = arrow_pts[1].astype(int)
            cv2.arrowedLine(overlay, tuple(a0), tuple(a1),
                            (60, 255, 60), 3, cv2.LINE_AA, tipLength=0.25)

        cv2.putText(overlay, f"#{i}",
                    (int(proj[3, 0]) + 6, int(proj[3, 1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    for tx, ty in dropped:
        ix, iy = int(round(tx)), int(round(ty))
        cv2.drawMarker(overlay, (ix, iy), (160, 160, 160),
                       cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(overlay, "?", (ix + 8, iy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(overlay, (0, 0), (W_img, 56), (0, 0, 0), -1)
    cv2.putText(overlay, hud_top, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, hud_bot, (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="frame_0000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--src-fps", type=int, default=4)
    ap.add_argument("--target-fps", type=int, default=4)
    ap.add_argument("--frames-dir", default="frames")
    ap.add_argument("--out-dir", default="seq_solve")
    ap.add_argument("--tip-ckpt", default="tile_tip_frame_dinov3_s8_v2.pt")
    ap.add_argument("--base-ckpt", default="base_head_attn.pt")
    ap.add_argument("--bowstern-ckpt", default="bowstern_head_attn.pt")
    ap.add_argument("--dinov3-model",
                    default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--max-long-side", type=int, default=1808)
    ap.add_argument("--dv3-stride", type=int, default=8)
    ap.add_argument("--tile-stride", type=int, default=2)
    ap.add_argument("--sig-thresh", type=float, default=0.2)
    ap.add_argument("--cluster-eps", type=float, default=2.5)
    ap.add_argument("--min-peak-dist", type=float, default=2.0)
    ap.add_argument("--peak-rel-thresh", type=float, default=0.5)
    ap.add_argument("--min-tip-base-px", type=float, default=10.0)
    ap.add_argument("--err-thresh", type=float, default=15.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stride = max(1, args.src_fps // args.target_fps)
    if not args.start.startswith("frame_"):
        raise SystemExit("--start must be like 'frame_0000'")
    base_idx = int(args.start.split("_")[1])
    stems = [f"frame_{base_idx + k * stride:04d}" for k in range(args.n)]
    print(f"frames: {len(stems)}  stride={stride}  "
          f"first={stems[0]}  last={stems[-1]}")

    print("loading models …")
    (model, n_special, tip_head, tip_win,
     base_head, base_meta, bs_head, _) = load_models(
        device, args.tip_ckpt, args.base_ckpt, args.bowstern_ckpt,
        args.dinov3_model, args.dv3_stride)

    rms_log = []
    for i, stem in enumerate(stems):
        fp = Path(args.frames_dir) / f"{stem}.jpg"
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  [{i+1}/{len(stems)}] missing {fp}")
            continue

        boats, dropped = detect_frame(
            img, model, n_special, tip_head, tip_win,
            base_head, base_meta, bs_head, device,
            args.max_long_side, args.dv3_stride,
            args.tile_stride, args.sig_thresh,
            args.cluster_eps, args.min_peak_dist,
            args.peak_rel_thresh, args.min_tip_base_px)

        bm = [m for m in (to_meas(b) for b in boats) if m is not None]
        if len(bm) < 2:
            print(f"  [{i+1}/{len(stems)}] {stem}: only {len(bm)} "
                  f"boats — skipping")
            continue

        try:
            sol = solve_iterative_drop(
                bm, img.shape[:2], err_thresh=args.err_thresh,
                heel_prior_weight=10.0,
                heel_deadzone_rad=np.radians(20.0))
        except Exception as e:
            print(f"  [{i+1}/{len(stems)}] {stem}: solve failed ({e})")
            continue

        p = np.asarray(sol["params"], dtype=np.float64)
        am = np.asarray(sol["active_mask"], dtype=bool)
        rms = float(sol["rms"])
        n_drop = int((~am).sum())

        cam = {
            "pitch_rad": float(sol["pitch"]),
            "pitch_deg": float(np.degrees(sol["pitch"])),
            "yaw_rad": float(sol["yaw"]),
            "yaw_deg": float(np.degrees(sol["yaw"])),
            "height_m": float(sol["h"]),
            "cam_x_m": float(sol["cam_x"]),
            "cam_y_m": float(sol["cam_y"]),
            "focal_px": float(sol["focal"]),
            "focal_estimated": True,
        }
        boats_out = [{
            "x_m": float(b["x"]), "y_m": float(b["y"]),
            "yaw_rad": float(b["yaw"]),
            "yaw_deg": float(np.degrees(b["yaw"])),
            "heel_deg": float(b["heel_deg"]),
            "track_id": k,
            "reproj_err_px": 0.0,
            "meas_kpts": bm[k].tolist(),
        } for k, b in enumerate(sol["boats"])]
        json_out = {
            "frame": stem,
            "image_size": [img.shape[1], img.shape[0]],
            "active_mask": am.tolist(),
            "camera": cam,
            "world_model": {"hull_length_m": float(HULL_L),
                            "mast_height_m": 5.2},
            "boats": boats_out,
            "rms_px": rms,
        }
        (out_dir / f"{stem}.solve.json").write_text(
            json.dumps(json_out, indent=2))

        hud_top = (
            f"{stem}  pitch={cam['pitch_deg']:.1f}deg  "
            f"yaw={cam['yaw_deg']:.1f}deg  "
            f"h={cam['height_m']:.1f}m  f={cam['focal_px']:.0f}px")
        hud_bot = (
            f"detected (cyan ring) vs reprojected (orange)   "
            f"heading=green arrow   RMS={rms:.2f}px   dropped={n_drop}")
        render_solve_overlay(
            img, p, bm, dropped, hud_top, hud_bot,
            out_dir / f"{stem}.solve.jpg",
            active_mask=am)
        render_topdown(
            p, len(bm),
            f"{stem}  cam=({cam['cam_x_m']:.1f},{cam['cam_y_m']:.1f})m  "
            f"h={cam['height_m']:.1f}m  RMS={rms:.2f}px",
            out_dir / f"{stem}.topdown.jpg")

        rms_log.append((stem, rms, len(bm), n_drop))
        print(f"  [{i+1}/{len(stems)}] {stem}: {len(bm)} boats, "
              f"dropped {n_drop}, RMS={rms:.2f}px")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if rms_log:
        rmss = np.array([r[1] for r in rms_log])
        print(f"\n=== summary: {len(rms_log)} frames ===")
        print(f"  rms_px: med={np.median(rmss):.2f}  "
              f"p25={np.percentile(rmss, 25):.2f}  "
              f"p75={np.percentile(rmss, 75):.2f}  "
              f"p90={np.percentile(rmss, 90):.2f}  "
              f"max={rmss.max():.2f}")
        summary = {
            "frames": [{"stem": s, "rms_px": r,
                        "n_boats": n, "n_dropped_kpts": d}
                       for s, r, n, d in rms_log],
            "rms_summary": {
                "median": float(np.median(rmss)),
                "p90": float(np.percentile(rmss, 90)),
                "max": float(rmss.max()),
            },
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
