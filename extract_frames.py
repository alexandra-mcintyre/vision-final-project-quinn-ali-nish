"""Extract frames from clip.mp4 for labeling and training.

Extracts frames at regular intervals, skipping duplicates and
low-content frames (e.g., overlays, transitions).

Usage:
    python extract_frames.py [--clip clip.mp4] [--output frames/] [--every 15] [--max 200]
"""

import cv2
import os
import argparse
import numpy as np


def extract_frames(clip_path, output_dir, every=15, max_frames=200, min_diff=5.0):
    """
    Extract frames from video at regular intervals.

    Args:
        clip_path: Path to the video file
        output_dir: Directory to save extracted frames
        every: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        min_diff: Minimum mean pixel difference from previous frame to avoid duplicates
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {clip_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {clip_path}")
    print(f"  {w}x{h}, {fps:.0f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")
    print(f"  Extracting every {every} frames, max {max_frames}")

    prev_gray = None
    extracted = 0
    skipped_dup = 0

    for fnum in range(0, total_frames, every):
        if extracted >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = cap.read()
        if not ret:
            break

        # Skip near-duplicate frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff < min_diff:
                skipped_dup += 1
                continue
        prev_gray = gray

        # Save frame
        fname = f"frame_{extracted:04d}_f{fnum:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, fname), frame)
        extracted += 1

        if extracted % 50 == 0:
            t_sec = fnum / fps
            print(f"  {extracted} frames extracted (t={t_sec:.1f}s)")

    cap.release()
    print(f"\nDone: {extracted} frames saved to {output_dir}/")
    print(f"  Skipped {skipped_dup} near-duplicate frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training frames from video")
    parser.add_argument("--clip", default="clip.mp4", help="Input video path")
    parser.add_argument("--output", default="mast_box_labels/images", help="Output directory")
    parser.add_argument("--every", type=int, default=15, help="Extract every N frames")
    parser.add_argument("--max", type=int, default=200, help="Max frames to extract")
    parser.add_argument("--min-diff", type=float, default=5.0, help="Min pixel difference to avoid duplicates")
    args = parser.parse_args()

    extract_frames(args.clip, args.output, args.every, args.max, args.min_diff)
