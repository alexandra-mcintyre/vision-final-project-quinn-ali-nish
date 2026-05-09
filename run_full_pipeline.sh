#!/bin/bash
# One-shot end-to-end pipeline:
#   YouTube → MP4 → frames → DINOv3+CNN inference → per-frame solver
#
# Hardcoded to the section of the ICSA Open Team Race Nationals 2025 Day 2
# YouTube video that matches the training distribution — and where we
# previously achieved <15 px median chain-RMS.
#
# Usage:
#   ./run_full_pipeline.sh                       # default 50-sec range
#   YT_START=2:34:38 YT_DUR=600 ./run_full_pipeline.sh   # full ~10-min section
#
# Outputs:
#   ./clip.mp4
#   ./frames/frame_NNNN.jpg
#   ./seq_solve/frame_NNNN.solve.{json,jpg} + topdown.jpg + summary.json
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

# ----- knobs -----
YT_URL="${YT_URL:-https://www.youtube.com/watch?v=3gg4dBLzYZo}"
YT_START="${YT_START:-0:22}"      # 0:22 -> 1:05 of the YouTube video
YT_DUR="${YT_DUR:-43}"             # seconds
FPS="${FPS:-4}"                    # match training distribution
CLIP="${CLIP:-$HERE/clip.mp4}"
FRAMES_DIR="${FRAMES_DIR:-$HERE/frames}"
OUT_DIR="${OUT_DIR:-$HERE/seq_solve}"

# ----- helper: HH:MM:SS or MM:SS or SS -> seconds -----
to_sec() {
  awk -v s="$1" 'BEGIN{
    n=split(s,a,":");
    if(n==1) print a[1]+0;
    else if(n==2) print a[1]*60+a[2];
    else print a[1]*3600+a[2]*60+a[3];
  }'
}
START_S=$(to_sec "$YT_START")
END_S=$(( START_S + YT_DUR ))

echo "==========================================================="
echo "  Pipeline:  YouTube → solves at $FPS fps"
echo "==========================================================="
echo "  source: $YT_URL"
echo "  range:  $YT_START + ${YT_DUR}s   (=> ${START_S}s..${END_S}s)"
echo "  fps:    $FPS"
echo "  clip:   $CLIP"
echo "  frames: $FRAMES_DIR"
echo "  out:    $OUT_DIR"
echo "==========================================================="

# ----- 1. Download (skip if cached) -----
if [ -s "$CLIP" ]; then
    echo "==> [1/3] clip exists, skipping download ($(ls -lh "$CLIP" | awk '{print $5}'))"
else
    echo "==> [1/3] downloading from YouTube"
    yt-dlp \
        --ffmpeg-location /usr/bin/ffmpeg \
        --download-sections "*${START_S}-${END_S}" \
        -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best" \
        --merge-output-format mp4 \
        -o "$CLIP" \
        "$YT_URL"
fi

# ----- 2. Extract frames -----
echo "==> [2/3] extracting frames at $FPS fps"
rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"
ffmpeg -y -i "$CLIP" -vf "fps=$FPS" -q:v 2 -start_number 0 \
       "$FRAMES_DIR/frame_%04d.jpg" 2>&1 | tail -2
N_FRAMES=$(ls "$FRAMES_DIR" | wc -l)
echo "    $N_FRAMES frames"

# ----- 3. Solve -----
echo "==> [3/3] running per-frame solver"
FIRST=$(ls "$FRAMES_DIR" | sort | head -1 | sed 's/\.jpg$//')
rm -rf "$OUT_DIR"
cd "$HERE/scripts"
python3 solve_camera_seq.py \
    --frames-dir "$FRAMES_DIR" \
    --out-dir    "$OUT_DIR" \
    --start "$FIRST" --n "$N_FRAMES" \
    --src-fps "$FPS" --target-fps "$FPS" \
    --tip-ckpt      "$HERE/models/tile_tip_frame_dinov3_s8_v2.pt" \
    --base-ckpt     "$HERE/models/base_head_attn.pt" \
    --bowstern-ckpt "$HERE/models/bowstern_head_attn.pt" \
    > "$HERE/pipeline.log" 2>&1

# ----- summary -----
echo "==> done with solver."
python3 - <<EOF
import glob, json, numpy as np
files = sorted(glob.glob('$OUT_DIR/frame_*.solve.json'))
if not files:
    print('  (no solves produced)')
else:
    rms = [json.load(open(f))['rms_px'] for f in files]
    print(f'  frames solved: {len(rms)}')
    print(f'  rms_px:  med={np.median(rms):5.2f}  p25={np.percentile(rms,25):5.2f}'
          f'  p75={np.percentile(rms,75):5.2f}  p90={np.percentile(rms,90):5.2f}'
          f'  max={max(rms):6.2f}')
EOF

echo
echo "  per-frame solves + overlays: $OUT_DIR/frame_*.solve.{json,jpg}"
echo "  full log:                    $HERE/pipeline.log"
