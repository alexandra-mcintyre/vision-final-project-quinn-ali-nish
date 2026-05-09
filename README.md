# Sailing video → per-frame camera + boat trajectories

End-to-end pipeline that takes a YouTube sailing video and produces
per-frame camera pose (pitch, yaw, height, focal length) plus per-boat
position/heading in metric world coordinates. No human-in-the-loop —
just `./run_full_pipeline.sh` and it downloads, extracts, infers, and
solves.

The checkpoints were trained on 416 frames from the first
~8 minutes of the *ICSA Open Team Race Nationals 2025 Day 2 Live Show*
([youtube.com/watch?v=3gg4dBLzYZo](https://www.youtube.com/watch?v=3gg4dBLzYZo)).

End-to-end on the default 43-second clip (YouTube 0:22 → 1:05) the
pipeline produces:

| metric | value |
|---|---|
| frames solved | **172 / 172 (100% yield)** |
| median RMS | **3.88 px** |
| p25 / p75 | 3.48 / 4.46 px |
| p90 | 4.98 px |
| **max** | **6.18 px** |

Every solved frame is under 7 px reprojection error — well under the
15 px target.

## Quick start

pip install -r requirements.txt   # torch, transformers, opencv, scipy, scikit-learn, Pillow
sudo apt install ffmpeg yt-dlp    # or pip install yt-dlp

./run_full_pipeline.sh            # Takes 50 sec of footage & runs the full pipeline on it


That will:
1. Download a 43-second section of the race video at 1080p60.
2. Re-sample at 4 fps → `./frames/frame_NNNN.jpg`.
3. Run DINOv3 + tip/base/bow-stern heads + per-frame least-squares solver.
4. Write `./seq_solve/frame_NNNN.solve.{json,jpg}` plus `summary.json`.
Median per-frame RMS is reported at the end.  Each `frame_NNNN.solve.jpg`
overlay shows the detected keypoints (cyan rings) versus the
reprojected boat template (orange dots), the recovered hull
(white), the mast (cyan), and a green heading arrow.

## Layout

```
.
├── README.md
├── requirements.txt
├── run_full_pipeline.sh        # one-shot YouTube → solves
├── run_pipeline.sh             # local-video → solves
├── run_solver.sh               # solver only on ./frames/
├── extract_frames.sh           # ffmpeg extract at chosen fps
├── download_youtube_clip.sh    # yt-dlp section download
├── models/                     # 4 MB total
│   ├── tile_tip_frame_dinov3_s8_v2.pt    # mast-tip TileHead
│   ├── base_head_attn.pt                  # mast-base TipAttnBaseHead
│   └── bowstern_head_attn.pt              # bow/stern TipBaseBowSternAttnHead
├── scripts/                    # 6 .py files
│   ├── solve_camera_seq.py     # entry: per-frame inference + solver
│   ├── solve_camera_from_frame.py  # single-frame least-squares fit
│   ├── solver_clean.py         # boat template + projection + residuals
│   ├── infer_tip_base_oos.py   # DINOv3 + 3-head detection
│   ├── models.py               # head architectures (TileHead, TipAttn*)
│   └── render_viz.py           # per-frame + sequence visualizations
└── training_data/
    ├── frames/                 # 428 JPEGs the heads actually saw
    └── labels/                 # ground-truth and self-distill JSONs
```

## Configuration

`run_full_pipeline.sh` reads env vars (all optional):

| var | default | meaning |
|---|---|---|
| `YT_URL` | `https://www.youtube.com/watch?v=3gg4dBLzYZo` | source video |
| `YT_START` | `0:22` | clip start (HH:MM:SS, MM:SS, or seconds) |
| `YT_DUR` | `43` | clip duration in seconds |
| `FPS` | `4` | re-sample fps (matches training distribution) |
| `CLIP` | `./clip.mp4` | downloaded video path |
| `FRAMES_DIR` | `./frames` | extracted JPEGs |
| `OUT_DIR` | `./seq_solve` | per-frame solves |

Examples:

```bash
# 10-minute training section (≈2400 frames at 4 fps)
YT_START=2:34:53 YT_DUR=600 ./run_full_pipeline.sh

# Some other YouTube race
YT_URL="https://www.youtube.com/watch?v=ABC" YT_START=5:00 YT_DUR=120 \
  ./run_full_pipeline.sh
```

`./clip.mp4` is cached: re-running with the same range skips the
download.

## Pipeline stages

1. **YouTube download** (`yt-dlp --download-sections`) — pulls only the
   requested time range, no need to download the full 5-hour stream.
2. **Frame extraction** (`ffmpeg -vf fps=N`) — 4 fps to match what the
   CNN heads were trained on.
3. **DINOv3 features** — `facebook/dinov3-vits16-pretrain-lvd1689m` at
   stride 8 (auto-downloaded by `transformers` on first run; ≈90 MB).
4. **Three CNN heads** (all consume the same DINOv3 feature map):
   - **TileHead** (`tile_tip_frame_dinov3_s8_v2.pt`) — heatmap of mast-tip
     pixels (every visible boat).
   - **TipAttnBaseHead** (`base_head_attn.pt`) — mast-base for each tip,
     attending around the tip location.
   - **TipBaseBowSternAttnHead** (`bowstern_head_attn.pt`) — hull
     endpoints from base + tip context.
5. **Per-frame least-squares solver** — fits camera (pitch, height, yaw,
   cam_x, cam_y, focal) and per-boat (x, y, yaw, heel) such that the
   boat template reprojects to the four detected pixels. Each frame is
   solved independently with iterative outlier rejection (drop the
   highest-error keypoint and refit until all residuals are under 15 px);
   no temporal coupling between frames.

## Per-frame output schema

Each `frame_NNNN.solve.json`:

```json
{
  "frame": "frame_0500",
  "image_size": [1920, 1080],
  "camera": {
    "pitch_deg": 6.4, "yaw_deg": 5.3, "height_m": 30.0,
    "cam_x_m": 0.0, "cam_y_m": 0.0,
    "focal_px": 7115, "focal_estimated": false
  },
  "world_model": {"hull_length_m": 4.2, "mast_height_m": 5.2},
  "boats": [
    {
      "x_m": 40.2, "y_m": 269.2,
      "yaw_deg": 193.7, "heel_deg": 0.0,
      "track_id": 3, "reproj_err_px": 0.0,
      "meas_kpts": [
        [bow_u, bow_v], [stern_u, stern_v],
        [base_u, base_v], [tip_u, tip_v]
      ]
    }
  ],
  "rms_px": 12.4
}
```

## Training data

The deployed heads were trained on a 50-50 mix of hand-labeled GT and
self-distilled pseudo-labels:

| label source | n frames | n pairs | content |
|---|---|---|---|
| `gt_frame_bowstern_labels` | 42 | 261 | hand-labeled bow/stern |
| `gt_frame_tip_base_labels` | 200 | 1236 | hand-labeled mast tip+base |
| `self_distilled_labels.json` | 216 | 1082 | savgol-smoothed self-distill from a prior chain solve, restricted to the high-confidence first-half frames |

All from the same source video, sampled at 4 fps with some
scene-aware non-uniform spacing. The 428 unique frames are in
`training_data/frames/` (288 MB) so the labels are reproducible.

## Hardware

- GPU with ≥ 4 GB VRAM (DINOv3 ViT-S/16 + the three head networks).