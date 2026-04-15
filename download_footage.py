"""Download race footage from YouTube and extract the race clip.

Requirements:
    pip install yt-dlp

Usage:
    python download_footage.py
"""

import subprocess
import sys
import os

VIDEO_URL = "https://www.youtube.com/watch?v=3gg4dBLzYZo"
FULL_VIDEO = "video.mp4"
CLIP_OUTPUT = "clip.mp4"

# Race clip: 10.5 minutes starting at the race broadcast
# Adjust these timestamps for your specific race
CLIP_START = "0:03:17"  # start time in the full video
CLIP_DURATION = "0:10:33"  # duration of the clip


def download():
    """Download the full video from YouTube."""
    if os.path.exists(FULL_VIDEO):
        print(f"{FULL_VIDEO} already exists, skipping download")
        return

    print(f"Downloading {VIDEO_URL}...")
    try:
        subprocess.run(
            ["yt-dlp", "-o", FULL_VIDEO, VIDEO_URL],
            check=True,
        )
        print(f"Downloaded to {FULL_VIDEO}")
    except FileNotFoundError:
        print("yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)


def extract_clip():
    """Extract the race clip from the full video."""
    if os.path.exists(CLIP_OUTPUT):
        print(f"{CLIP_OUTPUT} already exists, skipping extraction")
        return

    if not os.path.exists(FULL_VIDEO):
        print(f"{FULL_VIDEO} not found. Run download first.")
        sys.exit(1)

    print(f"Extracting clip: start={CLIP_START}, duration={CLIP_DURATION}")
    subprocess.run(
        [
            "ffmpeg", "-i", FULL_VIDEO,
            "-ss", CLIP_START,
            "-t", CLIP_DURATION,
            "-c", "copy",
            CLIP_OUTPUT,
        ],
        check=True,
    )
    print(f"Extracted to {CLIP_OUTPUT}")


if __name__ == "__main__":
    download()
    extract_clip()
