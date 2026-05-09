#!/bin/bash
# Extract frames from a sailing video at the chosen fps.
# Usage:
#   ./extract_frames.sh <video_path> [out_dir] [fps]
# Defaults: out_dir=frames, fps=4 (matches the training corpus).
set -e
VIDEO="${1:?usage: extract_frames.sh <video_path> [out_dir] [fps]}"
OUT="${2:-frames}"
FPS="${3:-4}"
mkdir -p "$OUT"
ffmpeg -y -i "$VIDEO" -vf "fps=$FPS" -q:v 2 -start_number 0 "$OUT/frame_%04d.jpg"
echo "Wrote $(ls "$OUT" | wc -l) frames to $OUT (at ${FPS} fps)"
