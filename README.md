# Sailing Race Analysis from Drone Footage

Extracting boat positions, headings, and race tactics from a single drone video of a college sailing race (C420 dinghies, Stanford vs Yale).

## Pipeline Overview

```
YouTube Video → Frame Extraction → Keypoint Labeling → YOLO Pose Training → Per-Frame 3D Solving → Race Visualization
```

### 1. Download footage
```bash
pip install yt-dlp
python download_footage.py
```
Downloads the full race broadcast from YouTube and extracts a 10.5-minute clip (`clip.mp4`, 1920x1080, 60fps).

### 2. Extract training frames
```bash
python extract_frames.py --clip clip.mp4 --output mast_box_labels/images --every 15 --max 200
```
Extracts ~200 frames at regular intervals, skipping near-duplicates. Saves as `frame_XXXX_fYYYYYY.jpg`.

### 3. Label keypoints
```bash
python label_frames.py
```
Interactive OpenCV labeler for marking 4 keypoints per sailboat and 2 per mark:

**Sailboats (class 0):** bow, stern, mast tip, mast base
**Marks (class 1-4):** start boat, finish boat, yellow mark, red mark — tip/base only

Controls:
| Key | Action |
|-----|--------|
| T | Mast tip mode |
| B | Mast base mode |
| W | Bow mode |
| E | Stern mode |
| G | Manual group mode — click 4 points to link them as one boat |
| U | Remove last group |
| P | Auto-pair tips/bases by proximity |
| L | Link bow/stern to a mast base |
| A/D | Previous/next frame |
| S | Save |
| Q | Save and quit |
| 0-4 | Select class |
| Right-click | Delete nearest point |

Labels are saved to `mast_frame_labels.json`.

### 4. Build YOLO training data
```bash
python build_kpt_dataset.py
```
Converts `mast_frame_labels.json` → YOLO pose format with 4 keypoints per detection:
- kpt0 = bow (visibility 0 for marks)
- kpt1 = stern (visibility 0 for marks)
- kpt2 = mast tip
- kpt3 = mast base

Generates horizontally mirrored copies for augmentation. Output: `mast_kpt_data/{train,val}/{images,labels}/`

### 5. Train YOLO pose model
```python
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
model.train(
    data='mast_kpt_data/dataset.yaml',
    epochs=100,
    patience=15,
    imgsz=1280,
    batch=8,
    name='pose_4kpt',
    project='runs/pose',
)
```
Trains a YOLOv8-nano pose model with 5 classes and 4 keypoints. ~7 minutes on a 3060.

### 6. Solve 3D positions

The projection solver estimates real-world positions of all boats from their pixel keypoints using a pinhole camera model.

**Boat model:** Each C420 is modeled as 4 3D points:
- Bow: (-L/3, 0, 0) — 1.4m forward of mast
- Stern: (2L/3, 0, 0) — 2.8m aft of mast
- Mast base: (0, 0, 0) — on the water
- Mast top: (0, 0, H) — 5.89m up

**Camera model:** Pinhole with focal length f, rotation (pitch + yaw), translation (x, y, h).

**Bundle solver** (`solver_bundle.py`):
- Solve 5-frame windows jointly
- Camera moves linearly: (x0+dx\*t, y0+dy\*t, h0+dh\*t)
- Shared pitch, yaw, focal length per window
- More stable but slower (~60s/window)

**Bow-stern weighting:** The solver downweights bow/stern keypoints that are geometrically inconsistent (ratio < 0.25 of mast height or > 2.0). Only the 2 most beam-on boats get full bow/stern weight — the rest are mast-only.

```bash
# Quick per-frame solve
python -c "
from solver_simple import solve_reference_frame, solve_frame_simple
# ... see solver_frames_v14 for full example
"
```

## Known objects

| Object | Count | Keypoints |
|--------|-------|-----------|
| Sailboat (C420) | 6 | bow, stern, mast tip, mast base |
| Start boat (RC) | 1 | tip, base |
| Finish boat | 1 | tip, base |
| Yellow mark (pin) | 2 | tip only (duplicated) |
| Red mark | 4 | tip only (duplicated) |

## C420 dimensions
- Hull length: 4.2m
- Mainsail luff (mast height): 5.89m
- Mast position: ~1/3 from bow

## Camera degeneracy

With a single camera, focal length (f), height (h), and pitch are degenerate — many combinations produce similar projections. To break this:
1. Fix 2 of 3 (f, h, pitch) and solve the third
2. Use mast heights to estimate distance (breaks f/h degeneracy)
3. Bundle multiple frames with linear camera motion

## Files

| File | Description |
|------|-------------|
| `download_footage.py` | Download video from YouTube |
| `extract_frames.py` | Extract training frames |
| `label_frames.py` | Interactive 4-keypoint labeler |
| `build_kpt_dataset.py` | Convert labels to YOLO format |
| `mast_frame_labels.json` | 200 labeled frames |
| `projection_solver.py` | Core 3D projection + residuals |
| `solver_simple.py` | Per-frame solver (fixed camera) |
| `solver_bundle.py` | 5-frame bundle solver |
| `training_data/dataset.yaml` | YOLO pose training config |
| `training_data/labels/` | YOLO format training labels |

## Dependencies

```
pip install ultralytics opencv-python numpy scipy yt-dlp
```

## Quick start

```bash
# 1. Get the video
python download_footage.py

# 2. Extract frames
python extract_frames.py

# 3. Label (or use provided labels)
python label_frames.py

# 4. Build training data
python build_kpt_dataset.py

# 5. Train
python -c "
from ultralytics import YOLO
YOLO('yolov8n-pose.pt').train(data='mast_kpt_data/dataset.yaml', epochs=100, imgsz=1280, batch=8)
"

# 6. Run inference + solve
# See projection_solver.py, solver_simple.py, solver_bundle.py
```
