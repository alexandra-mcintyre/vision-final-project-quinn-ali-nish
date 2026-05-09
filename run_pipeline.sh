#!/bin/bash
# End-to-end pipeline driver: video → frames → DINOv3 + 3 CNN heads
# → per-frame initial camera/boat solves.
#
# Usage:
#   ./run_pipeline.sh <video_path> [start_frame] [n_frames]
# Example:
#   ./run_pipeline.sh race.mp4 frame_0314 449
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CWD="$PWD"
VIDEO="${1:?usage: run_pipeline.sh <video_path> [start_frame] [n_frames]}"
START="${2:-frame_0314}"
N="${3:-449}"
FPS="${FPS:-4}"
FRAMES_DIR="${FRAMES_DIR:-$CWD/frames}"
OUT_DIR="${OUT_DIR:-$CWD/seq_solve}"

echo "=== 1. Extract frames at $FPS fps ==="
"$HERE/extract_frames.sh" "$VIDEO" "$FRAMES_DIR" "$FPS"

echo "=== 2. Run sequential per-frame solver ==="
echo "    (DINOv3 features + tip/base/bowstern heads + per-frame solve)"
cd "$HERE/scripts"
python3 solve_camera_seq.py \
    --frames-dir "$FRAMES_DIR" \
    --out-dir "$OUT_DIR" \
    --start "$START" --n "$N" \
    --src-fps "$FPS" --target-fps "$FPS" \
    --tip-ckpt "$HERE/models/tile_tip_frame_dinov3_s8_v2.pt" \
    --base-ckpt "$HERE/models/base_head_attn.pt" \
    --bowstern-ckpt "$HERE/models/bowstern_head_attn.pt"

echo
echo "Done. Per-frame solves at: $OUT_DIR"
echo "Each frame: $OUT_DIR/frame_NNNN.solve.json + .solve.jpg + .topdown.jpg"
