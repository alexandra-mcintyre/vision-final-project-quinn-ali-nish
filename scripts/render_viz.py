"""Build a `viz/` output folder with the useful visualizations.

Per-frame:
  - overlay.mp4     concatenated solve.jpg overlays (image-space)
  - topdown.mp4     concatenated solve.topdown.jpg (world-space)
  - rms_per_frame.png

Whole-sequence:
  - trajectory.png  topdown plot of all boat tracks + camera trail
  - trajectories.json  per-track world coordinates (joinable by track_id)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solves-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    solves_dir = Path(args.solves_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(solves_dir.glob("frame_*.solve.json"))
    if not files:
        raise SystemExit(f"no solves in {solves_dir}")
    print(f"reading {len(files)} solves")

    cam_path = []
    rms_per_frame = []
    tracks: dict[int, list[dict]] = defaultdict(list)
    for fi, fp in enumerate(files):
        d = json.loads(fp.read_text())
        cam = d["camera"]
        cam_path.append((fi, cam["cam_x_m"], cam["cam_y_m"],
                         cam["height_m"], cam["focal_px"]))
        rms_per_frame.append((fi, d["rms_px"]))
        for b in d["boats"]:
            tid = b.get("track_id")
            if tid is None:
                continue
            tracks[tid].append({
                "frame": fi, "stem": d["frame"],
                "x_m": b["x_m"], "y_m": b["y_m"],
                "yaw_deg": b["yaw_deg"], "heel_deg": b.get("heel_deg", 0.0),
            })

    stems = [json.loads(f.read_text())["frame"] for f in files]
    traj_json = {
        "stems": stems,
        "tracks": {str(tid): rows for tid, rows in tracks.items()},
        "cam": [{"frame": fi, "stem": stems[fi],
                 "cam_x_m": x, "cam_y_m": y, "height_m": h, "focal_px": f}
                for fi, x, y, h, f in cam_path],
    }
    (out_dir / "trajectories.json").write_text(
        json.dumps(traj_json, indent=2))
    print(f"  wrote trajectories.json ({len(tracks)} tracks)")

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.tab20
    for i, (tid, rows) in enumerate(sorted(tracks.items(),
                                            key=lambda r: -len(r[1]))):
        if len(rows) < 3:
            continue
        xs = [r["x_m"] for r in rows]
        ys = [r["y_m"] for r in rows]
        color = cmap(i % 20)
        ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=1.0)
        ax.scatter(xs, ys, color=color, s=4,
                   label=f"t{tid} ({len(rows)})" if len(rows) >= 20
                   else None)
    cam_xs = [c[1] for c in cam_path]
    cam_ys = [c[2] for c in cam_path]
    ax.plot(cam_xs, cam_ys, "k-", linewidth=2, label="camera")
    ax.scatter([cam_xs[0]], [cam_ys[0]], color="green", marker="^",
               s=120, zorder=5, label="cam start")
    ax.scatter([cam_xs[-1]], [cam_ys[-1]], color="red", marker="v",
               s=120, zorder=5, label="cam end")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"World-space trajectories  ({len(files)} frames, "
                 f"{sum(1 for t in tracks.values() if len(t) >= 3)} tracks)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=120)
    plt.close(fig)
    print("  wrote trajectory.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    fr = [r[0] for r in rms_per_frame]
    rms = [r[1] for r in rms_per_frame]
    ax.plot(fr, rms, "b-", linewidth=1)
    ax.axhline(np.median(rms), color="red", linestyle="--",
               label=f"median = {np.median(rms):.2f} px")
    ax.axhline(15, color="orange", linestyle=":",
               label="15 px target")
    ax.set_xlabel("frame index")
    ax.set_ylabel("rms_px")
    ax.set_title(f"Per-frame reprojection RMS  "
                 f"(med={np.median(rms):.2f}, max={max(rms):.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rms_per_frame.png", dpi=120)
    plt.close(fig)
    print("  wrote rms_per_frame.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fr = [c[0] for c in cam_path]
    axes[0, 0].plot(fr, [c[1] for c in cam_path], "b-")
    axes[0, 0].set_title("cam_x (m)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(fr, [c[2] for c in cam_path], "g-")
    axes[0, 1].set_title("cam_y (m)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(fr, [c[3] for c in cam_path], "r-")
    axes[1, 0].set_title("height (m)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(fr, [c[4] for c in cam_path], "purple")
    axes[1, 1].set_title("focal (px)")
    axes[1, 1].grid(True, alpha=0.3)
    for ax in axes.flat:
        ax.set_xlabel("frame")
    fig.suptitle("Camera trajectory")
    fig.tight_layout()
    fig.savefig(out_dir / "cam_trajectory.png", dpi=120)
    plt.close(fig)
    print("  wrote cam_trajectory.png")

    if list(solves_dir.glob("frame_*.solve.jpg")):
        out_mp4 = out_dir / "overlay.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-pattern_type", "glob",
            "-i", str(solves_dir / "frame_*.solve.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-loglevel", "error",
            str(out_mp4),
        ], check=True)
        print(f"  wrote overlay.mp4  ({out_mp4.stat().st_size // 1024}K)")

    topdown_dir = out_dir / "_topdown_frames"
    topdown_dir.mkdir(exist_ok=True)
    all_xs = [r["x_m"] for rows in tracks.values() for r in rows] + cam_xs
    all_ys = [r["y_m"] for rows in tracks.values() for r in rows] + cam_ys
    if all_xs:
        pad = 10
        xmin, xmax = min(all_xs) - pad, max(all_xs) + pad
        ymin, ymax = min(all_ys) - pad, max(all_ys) + pad
        for fi, fp in enumerate(files):
            d = json.loads(fp.read_text())
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(cam_xs[:fi+1], cam_ys[:fi+1],
                    "k-", linewidth=1.5, alpha=0.5)
            ax.scatter(cam_xs[fi], cam_ys[fi], color="black",
                       marker="^", s=100, zorder=5)
            for tid, rows in tracks.items():
                past = [r for r in rows if r["frame"] <= fi]
                if not past:
                    continue
                xs = [r["x_m"] for r in past]
                ys = [r["y_m"] for r in past]
                ax.plot(xs, ys, "-", color=cmap(tid % 20),
                        alpha=0.4, linewidth=0.8)
                if past[-1]["frame"] == fi:
                    ax.scatter(past[-1]["x_m"], past[-1]["y_m"],
                               color=cmap(tid % 20), s=40, zorder=4)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            ax.set_title(f"{d['frame']}  rms={d['rms_px']:.2f}px")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(topdown_dir / f"frame_{fi:04d}.png", dpi=80)
            plt.close(fig)
        out_mp4 = out_dir / "topdown.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-pattern_type", "glob",
            "-i", str(topdown_dir / "frame_*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-loglevel", "error",
            str(out_mp4),
        ], check=True)
        for f in topdown_dir.glob("frame_*.png"):
            f.unlink()
        topdown_dir.rmdir()
        print(f"  wrote topdown.mp4  ({out_mp4.stat().st_size // 1024}K)")

    rms_arr = np.array([r for _, r in rms_per_frame])
    n_tracks_long = sum(1 for t in tracks.values() if len(t) >= 20)
    summary = [
        f"frames solved: {len(files)}",
        f"unique tracks: {len(tracks)} "
        f"(>=20 frames: {n_tracks_long})",
        f"rms_px:  median={np.median(rms_arr):.2f}  "
        f"p25={np.percentile(rms_arr, 25):.2f}  "
        f"p75={np.percentile(rms_arr, 75):.2f}  "
        f"p90={np.percentile(rms_arr, 90):.2f}  "
        f"max={rms_arr.max():.2f}",
        f"camera traversed: "
        f"x={min(cam_xs):.1f}..{max(cam_xs):.1f}  "
        f"y={min(cam_ys):.1f}..{max(cam_ys):.1f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print()
    for line in summary:
        print(f"  {line}")


if __name__ == "__main__":
    main()
