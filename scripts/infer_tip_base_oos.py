"""End-to-end tip + base inference on out-of-sample frames.

Pipeline:
  1. DINOv3 ViT-S/16 dense features at stride 8 (L2-normalized).
  2. Sliding 16-cell tile head -> per-cell tip sigmoid -> DBSCAN +
     local-maxima clustering -> tip points (source-frame px).
  3. For each tip, extract a 21x34 window of features (tip pinned
     at row=2, col=10) and run TipAttnBaseHead -> per-cell base
     logit -> sigmoid -> peak -> base point (source-frame px).

Renders tips (green crosses), bases (red dots), and connecting
lines on the original frame.

Usage:
  python3 infer_tip_base_oos.py --frame frames_15fps/frame_0099.jpg
  python3 infer_tip_base_oos.py --frames-dir frames_15fps \\
      --random 8 --skip-labeled --out-dir /tmp/oos_viz
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN
from transformers import AutoModel

from models import TileHead, TipAttnBaseHead, TipBaseBowSternAttnHead


PATCH = 16
IMNET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor(
    [0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def resize_for_dinov3(img: np.ndarray, max_long: int
                          ) -> tuple[np.ndarray, float, float]:
    """Resize so long side <= max_long and both sides multiple
    of PATCH. Returns (rgb, sx_src, sy_src) where sx_src maps
    resized -> source x."""
    H, W = img.shape[:2]
    s = max_long / max(H, W)
    new_w = (int(round(W * s)) // PATCH) * PATCH
    new_h = (int(round(H * s)) // PATCH) * PATCH
    rgb = cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rgb, W / float(new_w), H / float(new_h)


@torch.no_grad()
def dinov3_features(model, rgb: np.ndarray, device: str,
                       stride: int, n_special: int
                      ) -> np.ndarray:
    """Compute (Gh, Gw, D) L2-normalized features for one frame."""
    Hi, Wi = rgb.shape[:2]
    Gh = (Hi - PATCH) // stride + 1
    Gw = (Wi - PATCH) // stride + 1
    t = (torch.from_numpy(rgb).float().div(255.0)
            .permute(2, 0, 1).unsqueeze(0).to(device))
    t = ((t - IMNET_MEAN.to(device))
          / IMNET_STD.to(device)).half()
    out = model(pixel_values=t)
    feat = (out.last_hidden_state[0, n_special:]
              .view(Gh, Gw, -1).cpu().numpy()
              .astype(np.float32))
    feat /= np.clip(np.linalg.norm(
        feat, axis=-1, keepdims=True), 1e-6, None)
    return feat


@torch.no_grad()
def tip_heatmap(feat: np.ndarray, head: TileHead,
                  device: str, win: int, tile_stride: int
                 ) -> np.ndarray:
    """Returns Gh x Gw sigmoid heatmap from sliding 16-cell tiles."""
    Gh, Gw, D = feat.shape
    ys_t = list(range(0, max(1, Gh - win + 1), tile_stride))
    if ys_t and ys_t[-1] != Gh - win:
        ys_t.append(Gh - win)
    xs_t = list(range(0, max(1, Gw - win + 1), tile_stride))
    if xs_t and xs_t[-1] != Gw - win:
        xs_t.append(Gw - win)
    BATCH = 256
    tile_buf = np.zeros((BATCH, D, win, win), dtype=np.float32)
    heat = np.zeros((Gh, Gw), dtype=np.float32)
    meta: list[tuple[int, int]] = []
    i = 0

    def flush(n: int) -> None:
        nonlocal meta, i
        tt = torch.from_numpy(tile_buf[:n]).to(device)
        sig = torch.sigmoid(head(tt)[:, 0]).cpu().numpy()
        for k, (yy, xx) in enumerate(meta):
            heat[yy:yy + win, xx:xx + win] = np.maximum(
                heat[yy:yy + win, xx:xx + win], sig[k])
        meta = []
        i = 0

    for y0 in ys_t:
        for x0 in xs_t:
            tile_buf[i] = feat[y0:y0 + win,
                                    x0:x0 + win
                                  ].transpose(2, 0, 1)
            meta.append((y0, x0))
            i += 1
            if i == BATCH:
                flush(BATCH)
    if i > 0:
        flush(i)
    return heat


def cluster_tips(heat: np.ndarray,
                    sig_thresh: float,
                    cluster_eps: float,
                    min_peak_dist: float,
                    peak_rel_thresh: float
                   ) -> list[tuple[float, float, float]]:
    """Cluster heatmap peaks. Returns (gx, gy, strength) in cell
    coords."""
    ys_, xs_ = np.where(heat >= sig_thresh)
    if len(xs_) == 0:
        return []
    cl = DBSCAN(eps=cluster_eps, min_samples=1).fit(
        np.column_stack([xs_, ys_]).astype(np.float64))
    out: list[tuple[float, float, float]] = []
    for lab in set(cl.labels_.tolist()):
        if lab < 0:
            continue
        mem = cl.labels_ == lab
        cxs = xs_[mem]
        cys = ys_[mem]
        pad = max(2, int(min_peak_dist))
        bx0 = max(0, int(cxs.min()) - pad)
        by0 = max(0, int(cys.min()) - pad)
        bx1 = min(heat.shape[1], int(cxs.max()) + pad + 1)
        by1 = min(heat.shape[0], int(cys.max()) + pad + 1)
        patch = heat[by0:by1, bx0:bx1].copy()
        cm = np.zeros_like(patch, dtype=bool)
        for cx, cy in zip(cxs, cys):
            cm[cy - by0, cx - bx0] = True
        pm = np.where(cm, patch, 0.0)
        wsz = max(3, int(round(min_peak_dist * 2 + 1)))
        lm = maximum_filter(pm, size=wsz)
        is_peak = (pm == lm) & (pm > 0.0)
        rt = peak_rel_thresh * float(pm.max())
        py_arr, px_arr = np.where(is_peak & (pm >= rt))
        emitted: list[tuple[float, float]] = []
        for py, px in zip(py_arr, px_arr):
            y0p = max(0, py - 1)
            x0p = max(0, px - 1)
            y1p = min(patch.shape[0], py + 2)
            x1p = min(patch.shape[1], px + 2)
            sub = pm[y0p:y1p, x0p:x1p]
            if sub.sum() <= 0:
                gy_p, gx_p = float(py), float(px)
            else:
                ys_sub, xs_sub = np.meshgrid(
                    np.arange(y0p, y1p),
                    np.arange(x0p, x1p), indexing="ij")
                gy_p = float((ys_sub * sub).sum() / sub.sum())
                gx_p = float((xs_sub * sub).sum() / sub.sum())
            gx_g = bx0 + gx_p
            gy_g = by0 + gy_p
            skip = False
            for (ex, ey) in emitted:
                if ((gx_g - ex) ** 2 + (gy_g - ey) ** 2
                        < min_peak_dist ** 2):
                    skip = True
                    break
            if skip:
                continue
            emitted.append((gx_g, gy_g))
            out.append((gx_g, gy_g,
                            float(pm[py, px])))
    out.sort(key=lambda t: -t[2])
    return out


@torch.no_grad()
def base_for_tip(feat: np.ndarray, head: TipAttnBaseHead,
                    device: str, tx_g: float, ty_g: float,
                    win_w: int, win_h: int, win_top: int
                   ) -> tuple[float, float, float]:
    """Returns (bx_g, by_g, peak_sigmoid) in cell coords."""
    Gh, Gw, D = feat.shape
    hw = win_w // 2
    gx0 = int(round(tx_g - hw))
    gy0 = int(round(ty_g - win_top))
    pad_top = max(0, -gy0)
    pad_left = max(0, -gx0)
    gx0_c = max(0, gx0)
    gy0_c = max(0, gy0)
    gy1_c = min(Gh, gy0 + win_h)
    gx1_c = min(Gw, gx0 + win_w)
    padded = np.zeros((win_h, win_w, D), dtype=np.float32)
    if gy1_c > gy0_c and gx1_c > gx0_c:
        slice_ = feat[gy0_c:gy1_c, gx0_c:gx1_c]
        padded[pad_top:pad_top + slice_.shape[0],
                  pad_left:pad_left + slice_.shape[1]] = slice_
    t = (torch.from_numpy(padded.transpose(2, 0, 1))
            .unsqueeze(0).to(device, dtype=torch.float32))
    logits = head(t)[0, 0].cpu().numpy()
    heat = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    py, px = np.unravel_index(int(np.argmax(heat)),
                                       heat.shape)
    # 3x3 weighted centroid for sub-cell peak
    y0 = max(0, py - 1); x0 = max(0, px - 1)
    y1 = min(heat.shape[0], py + 2)
    x1 = min(heat.shape[1], px + 2)
    sub = heat[y0:y1, x0:x1]
    if sub.sum() > 0:
        ys, xs = np.meshgrid(np.arange(y0, y1),
                                       np.arange(x0, x1),
                                       indexing="ij")
        py_s = float((ys * sub).sum() / sub.sum())
        px_s = float((xs * sub).sum() / sub.sum())
    else:
        py_s, px_s = float(py), float(px)
    return (gx0 + px_s, gy0 + py_s, float(heat[py, px]))


def cell_to_src(gx: float, gy: float, stride: int,
                  sx_src: float, sy_src: float
                 ) -> tuple[float, float]:
    half = (PATCH - 1) / 2.0
    return ((gx * stride + half) * sx_src,
              (gy * stride + half) * sy_src)


def _peak_subcell(heat: np.ndarray
                     ) -> tuple[float, float, float]:
    py, px = np.unravel_index(int(np.argmax(heat)),
                                       heat.shape)
    y0 = max(0, py - 1)
    x0 = max(0, px - 1)
    y1 = min(heat.shape[0], py + 2)
    x1 = min(heat.shape[1], px + 2)
    sub = heat[y0:y1, x0:x1]
    if sub.sum() > 0:
        ys, xs = np.meshgrid(np.arange(y0, y1),
                                       np.arange(x0, x1),
                                       indexing="ij")
        py_s = float((ys * sub).sum() / sub.sum())
        px_s = float((xs * sub).sum() / sub.sum())
    else:
        py_s, px_s = float(py), float(px)
    return px_s, py_s, float(heat[py, px])


@torch.no_grad()
def bowstern_for_pair(feat: np.ndarray,
                          head: TipBaseBowSternAttnHead,
                          device: str,
                          tx_g: float, ty_g: float,
                          bx_g: float, by_g: float,
                          win_w: int, win_h: int,
                          win_top: int
                         ) -> tuple[tuple[float, float, float],
                                            tuple[float, float, float]]:
    """Returns ((bow_gx, bow_gy, bow_p),
                 (stern_gx, stern_gy, stern_p)) in cell coords."""
    Gh, Gw, D = feat.shape
    hw = win_w // 2
    gx0 = int(round(tx_g - hw))
    gy0 = int(round(ty_g - win_top))
    pad_top = max(0, -gy0)
    pad_left = max(0, -gx0)
    gx0_c = max(0, gx0)
    gy0_c = max(0, gy0)
    gy1_c = min(Gh, gy0 + win_h)
    gx1_c = min(Gw, gx0 + win_w)
    padded = np.zeros((win_h, win_w, D), dtype=np.float32)
    if gy1_c > gy0_c and gx1_c > gx0_c:
        slice_ = feat[gy0_c:gy1_c, gx0_c:gx1_c]
        padded[pad_top:pad_top + slice_.shape[0],
                  pad_left:pad_left + slice_.shape[1]] = slice_
    # Base position in window cells (offset for any padding).
    bx_w = hw + (bx_g - tx_g) - (gx0_c - gx0)
    by_w = win_top + (by_g - ty_g) - (gy0_c - gy0)
    br = int(np.clip(round(by_w), 0, win_h - 1))
    bc = int(np.clip(round(bx_w), 0, win_w - 1))
    base_idx_t = torch.tensor([br * win_w + bc],
                                       dtype=torch.int64).to(device)
    feats_t = (torch.from_numpy(padded.transpose(2, 0, 1))
                  .unsqueeze(0).to(device, dtype=torch.float32))
    logits = head(feats_t, base_idx_t)[0].cpu().numpy()
    heat = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    bx_w_s, by_w_s, bp = _peak_subcell(heat[0])
    sx_w_s, sy_w_s, sp = _peak_subcell(heat[1])
    bow_gx = gx0 + bx_w_s
    bow_gy = gy0 + by_w_s
    stn_gx = gx0 + sx_w_s
    stn_gy = gy0 + sy_w_s
    return ((bow_gx, bow_gy, bp), (stn_gx, stn_gy, sp))


def render(img: np.ndarray,
              detections: list[dict]
             ) -> np.ndarray:
    """detections: list of dicts with tip, base, base_p, and
    optionally bow, stern, bow_p, stern_p. All coords in
    source-frame px."""
    canvas = img.copy()
    for k, det in enumerate(detections):
        if det.get("dropped_reason"):
            tx, ty = det["tip"]
            ix, iy = int(round(tx)), int(round(ty))
            cv2.drawMarker(canvas, (ix, iy),
                                  (160, 160, 160),
                                  cv2.MARKER_TILTED_CROSS,
                                  16, 2, cv2.LINE_AA)
            cv2.putText(canvas, "?", (ix + 8, iy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, "?", (ix + 8, iy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (200, 200, 200), 1, cv2.LINE_AA)
            continue
        tx, ty = det["tip"]
        bx, by = det["base"]
        bp = det.get("base_p", 0.0)
        ix, iy = int(round(tx)), int(round(ty))
        jx, jy = int(round(bx)), int(round(by))
        cv2.line(canvas, (ix, iy), (jx, jy),
                       (240, 240, 80), 2, cv2.LINE_AA)
        if det.get("bow") is not None and det.get("stern") is not None:
            wx, wy = det["bow"]
            sxw, syw = det["stern"]
            wx_i = int(round(wx)); wy_i = int(round(wy))
            sx_i = int(round(sxw)); sy_i = int(round(syw))
            cv2.line(canvas, (wx_i, wy_i), (sx_i, sy_i),
                            (200, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (wx_i, wy_i), 8, (0, 0, 0), -1,
                          cv2.LINE_AA)
            cv2.circle(canvas, (wx_i, wy_i), 5,
                          (60, 220, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx_i, sy_i), 8, (0, 0, 0), -1,
                          cv2.LINE_AA)
            cv2.circle(canvas, (sx_i, sy_i), 5,
                          (220, 60, 220), -1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (ix, iy), (50, 220, 50),
                              cv2.MARKER_CROSS, 18, 2,
                              cv2.LINE_AA)
        cv2.circle(canvas, (ix, iy), 7, (50, 220, 50), 2,
                      cv2.LINE_AA)
        cv2.circle(canvas, (jx, jy), 8, (0, 0, 0), -1,
                      cv2.LINE_AA)
        cv2.circle(canvas, (jx, jy), 5, (50, 50, 220), -1,
                      cv2.LINE_AA)
        label = f"{k+1}"
        cv2.putText(canvas, label, (ix + 9, iy - 9),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (ix + 9, iy - 9),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (50, 220, 50), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame", help="single frame path")
    g.add_argument("--frames-dir",
                    help="directory; sample with --random or --all")
    ap.add_argument("--random", type=int, default=0,
                    help="when using --frames-dir, sample N "
                          "frames at random")
    ap.add_argument("--all", action="store_true",
                    help="when using --frames-dir, run all "
                          "frames")
    ap.add_argument(
        "--skip-labeled", action="store_true",
        help="skip frames present in labeled set")
    ap.add_argument(
        "--labels", nargs="+",
        default=["gt_frame_tip_base_labels/labels.json",
                  "gt_frame_tip_base_labels_4fps/labels.json"])
    ap.add_argument("--tip-ckpt",
                    default="tile_tip_frame_dinov3_s8_v2.pt")
    ap.add_argument("--base-ckpt",
                    default="base_head_attn.pt")
    ap.add_argument("--bowstern-ckpt",
                    default="bowstern_head_attn.pt",
                    help="set to '' to skip the bow/stern stage")
    ap.add_argument(
        "--dinov3-model",
        default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--max-long-side", type=int, default=1808)
    ap.add_argument("--dv3-stride", type=int, default=8)
    ap.add_argument("--tile-stride", type=int, default=2)
    ap.add_argument("--sig-thresh", type=float, default=0.2)
    ap.add_argument("--cluster-eps", type=float, default=2.5)
    ap.add_argument("--min-peak-dist", type=float, default=2.0)
    ap.add_argument("--peak-rel-thresh", type=float, default=0.5)
    ap.add_argument(
        "--min-tip-base-px", type=float, default=10.0,
        help="drop detections whose predicted base falls within "
              "this many source-frame px of the tip — bow/stern "
              "stage has no orientation signal in that case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/oos_tip_base")
    ap.add_argument(
        "--out-json", default="",
        help="optional path to dump (frame -> detections) json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.frame:
        frames = [Path(args.frame)]
    else:
        fdir = Path(args.frames_dir)
        all_f = sorted(p for p in fdir.iterdir()
                          if p.suffix.lower() == ".jpg")
        if args.skip_labeled:
            labeled: set[str] = set()
            for lp in args.labels:
                if Path(lp).exists():
                    labeled.update(
                        json.loads(Path(lp).read_text()).keys())
            all_f = [p for p in all_f if p.name not in labeled]
        if args.random:
            rng = random.Random(args.seed)
            frames = rng.sample(all_f,
                                       k=min(args.random, len(all_f)))
        elif args.all:
            frames = all_f
        else:
            frames = all_f[:8]
    print(f"frames to process: {len(frames)}")

    model = (AutoModel.from_pretrained(args.dinov3_model)
                .eval().to(device).half())
    model.embeddings.patch_embeddings.stride = (
        args.dv3_stride, args.dv3_stride)
    n_special = 1 + (model.config.num_register_tokens
                          if hasattr(model.config,
                                          "num_register_tokens")
                          else 0)
    tk = torch.load(args.tip_ckpt, map_location=device,
                        weights_only=False)
    tip_win = int(tk.get("win", 16))
    tip_head = TileHead(in_dim=int(tk["in_dim"]),
                              hidden=int(tk.get("hidden", 64))
                             ).to(device).eval()
    tip_head.load_state_dict(tk["state_dict"])
    print(f"tip head: in_dim={tk['in_dim']} win={tip_win}")
    bk = torch.load(args.base_ckpt, map_location=device,
                        weights_only=False)
    win_w = int(bk["win_w"])
    win_h = int(bk["win_h"])
    win_top = int(bk["win_top"])
    base_head = TipAttnBaseHead(
        in_dim=int(bk["in_dim"]),
        hidden=int(bk.get("hidden", 128)),
        n_heads=int(bk["n_heads"]),
        n_layers=int(bk["n_layers"]),
        win_h=win_h, win_w=win_w,
        tip_row=win_top,
        tip_col=win_w // 2).to(device).eval()
    base_head.load_state_dict(bk["state_dict"])
    print(f"base head: win {win_w}x{win_h} top={win_top} "
          f"in_dim={bk['in_dim']}  best_loss"
          f"@ep{bk.get('epoch','?')}")
    bowstern_head = None
    if args.bowstern_ckpt and Path(args.bowstern_ckpt).exists():
        bs = torch.load(args.bowstern_ckpt,
                              map_location=device,
                              weights_only=False)
        bowstern_head = TipBaseBowSternAttnHead(
            in_dim=int(bs["in_dim"]),
            hidden=int(bs.get("hidden", 128)),
            n_heads=int(bs["n_heads"]),
            n_layers=int(bs["n_layers"]),
            win_h=int(bs["win_h"]),
            win_w=int(bs["win_w"]),
            tip_row=int(bs["win_top"]),
            tip_col=int(bs["win_w"]) // 2,
            n_out=int(bs.get("n_out", 2))).to(device).eval()
        bowstern_head.load_state_dict(bs["state_dict"])
        print(f"bowstern head: win {bs['win_w']}x{bs['win_h']} "
                  f"top={bs['win_top']} in_dim={bs['in_dim']} "
                  f"@ep{bs.get('epoch','?')}")

    results: dict[str, list] = {}
    for fi, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  [{fi+1}/{len(frames)}] skip: {fp}")
            continue
        rgb, sx_src, sy_src = resize_for_dinov3(
            img, args.max_long_side)
        feat = dinov3_features(
            model, rgb, device, args.dv3_stride, n_special)
        heat = tip_heatmap(
            feat, tip_head, device, tip_win, args.tile_stride)
        tips_g = cluster_tips(
            heat,
            sig_thresh=args.sig_thresh,
            cluster_eps=args.cluster_eps,
            min_peak_dist=args.min_peak_dist,
            peak_rel_thresh=args.peak_rel_thresh)
        detections: list[dict] = []
        for (tx_g, ty_g, _strength) in tips_g:
            bx_g, by_g, bp = base_for_tip(
                feat, base_head, device, tx_g, ty_g,
                win_w, win_h, win_top)
            tx_s, ty_s = cell_to_src(
                tx_g, ty_g, args.dv3_stride, sx_src, sy_src)
            bx_s, by_s = cell_to_src(
                bx_g, by_g, args.dv3_stride, sx_src, sy_src)
            d_px = ((bx_s - tx_s) ** 2
                       + (by_s - ty_s) ** 2) ** 0.5
            if d_px < args.min_tip_base_px:
                detections.append({
                    "tip": [round(tx_s, 2), round(ty_s, 2)],
                    "dropped_reason": "tip_base_lt_min",
                    "tip_base_dist_px": round(d_px, 2),
                })
                continue
            det = {
                "tip": [round(tx_s, 2), round(ty_s, 2)],
                "base": [round(bx_s, 2), round(by_s, 2)],
                "base_p": round(bp, 4),
            }
            if bowstern_head is not None:
                (bow_g, stn_g) = bowstern_for_pair(
                    feat, bowstern_head, device,
                    tx_g, ty_g, bx_g, by_g,
                    win_w, win_h, win_top)
                bow_x_s, bow_y_s = cell_to_src(
                    bow_g[0], bow_g[1], args.dv3_stride,
                    sx_src, sy_src)
                stn_x_s, stn_y_s = cell_to_src(
                    stn_g[0], stn_g[1], args.dv3_stride,
                    sx_src, sy_src)
                det.update({
                    "bow": [round(bow_x_s, 2),
                              round(bow_y_s, 2)],
                    "bow_p": round(bow_g[2], 4),
                    "stern": [round(stn_x_s, 2),
                                  round(stn_y_s, 2)],
                    "stern_p": round(stn_g[2], 4),
                })
            detections.append(det)
        canvas = render(img, detections)
        bar = canvas.copy()
        cv2.rectangle(bar, (0, 0), (canvas.shape[1], 30),
                              (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.65, canvas, 0.35, 0, canvas)
        cv2.putText(
            canvas,
            f"{fp.name}  tips={len(detections)}  "
            f"thr={args.sig_thresh:.2f}",
            (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA)
        out_p = out_dir / f"{fp.stem}_tipbase.jpg"
        cv2.imwrite(str(out_p), canvas)
        n_kept = sum(1 for d in detections
                          if not d.get("dropped_reason"))
        n_drop = sum(1 for d in detections
                          if d.get("dropped_reason"))
        print(f"  [{fi+1}/{len(frames)}] {fp.name}  "
                  f"kept={n_kept}  "
                  f"dropped<{args.min_tip_base_px:.0f}px="
                  f"{n_drop}  -> {out_p}")
        results[fp.name] = detections
        del feat, heat
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(results, indent=2))
        print(f"wrote detections json -> {args.out_json}")


if __name__ == "__main__":
    main()
