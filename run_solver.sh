#!/bin/bash
# Run the per-frame solver on ./frames/ (assumed pre-populated).
# Hardcoded to use the staged training-viewpoint frames at 4 fps.
# Outputs to ./seq_solve/.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
FRAMES_DIR="$HERE/frames"
OUT_DIR="$HERE/seq_solve"

if [ ! -d "$FRAMES_DIR" ] || [ -z "$(ls "$FRAMES_DIR" 2>/dev/null)" ]; then
    echo "error: $FRAMES_DIR is missing or empty"
    echo "  populate it (e.g. ./extract_frames.sh <video> frames 4) first"
    exit 1
fi

# Pick start = first frame stem, n = total count
FIRST=$(ls "$FRAMES_DIR" | sort | head -1 | sed 's/\.jpg$//')
N=$(ls "$FRAMES_DIR" | wc -l)

echo "Running solver on $N frames in $FRAMES_DIR (starting at $FIRST)"
cd "$HERE/scripts"
python3 solve_camera_seq.py \
    --frames-dir "$FRAMES_DIR" \
    --out-dir "$OUT_DIR" \
    --start "$FIRST" --n "$N" \
    --src-fps 4 --target-fps 4 \
    --tip-ckpt      "$HERE/models/tile_tip_frame_dinov3_s8_v2.pt" \
    --base-ckpt     "$HERE/models/base_head_attn.pt" \
    --bowstern-ckpt "$HERE/models/bowstern_head_attn.pt"

echo
echo "Done. Per-frame solves at: $OUT_DIR"
