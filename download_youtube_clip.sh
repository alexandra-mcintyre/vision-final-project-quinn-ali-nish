#!/bin/bash
# Download a section of a YouTube video as MP4.
#
# Usage:
#   ./download_youtube_clip.sh <url> <start> [duration] [out_path]
#
# Times accept ffmpeg syntax: "90", "1:30", "00:01:30".
# Default duration: 60 seconds. Default out_path: ./clip.mp4
#
# Examples:
#   ./download_youtube_clip.sh https://youtu.be/abc 90 60
#   ./download_youtube_clip.sh https://youtu.be/abc 1:30 60 race.mp4
set -e
URL="${1:?usage: download_youtube_clip.sh <url> <start> [duration] [out_path]}"
START="${2:?missing <start>}"
DUR="${3:-60}"
OUT="${4:-clip.mp4}"

# yt-dlp's --download-sections syntax: "*START-END"
END=$(awk -v s="$START" -v d="$DUR" 'BEGIN{
  n=split(s,a,":"); if(n==1) ss=a[1]; else if(n==2) ss=a[1]*60+a[2]; else ss=a[1]*3600+a[2]*60+a[3];
  print ss + d
}')

echo "downloading $URL  section ${START}–${END}s  -> $OUT"
# Use system ffmpeg if present — anaconda ffmpeg crashes on some
# --force-keyframes-at-cuts cuts.
FFMPEG_LOC=""
[ -x /usr/bin/ffmpeg ] && FFMPEG_LOC="--ffmpeg-location /usr/bin/ffmpeg"
yt-dlp $FFMPEG_LOC \
    --download-sections "*${START}-${END}" \
    -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best" \
    --merge-output-format mp4 \
    -o "$OUT" \
    "$URL"

echo "  wrote: $(ls -lh "$OUT" | awk '{print $5, $NF}')"
ffprobe -v error -show_entries stream=width,height,r_frame_rate,duration -of default=nw=1:nk=1 "$OUT" 2>&1 | head -4
