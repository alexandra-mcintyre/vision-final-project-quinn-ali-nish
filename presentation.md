# Sailboat Pose Estimation — Pretraining Pipeline & Single-Frame Solver

A presentation walkthrough of the per-frame inference pipeline:
DINOv3 features → 3 learned heads → metric per-frame solve.

---

## 0. Big picture (one slide)

```
                         ┌──────────────────────────────┐
                         │   Input: one drone frame     │
                         │   (1920 × 1080 BGR)          │
                         └──────────────┬───────────────┘
                                        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │   FROZEN BACKBONE                                             │
   │   DINOv3 ViT-S/16, no fine-tuning                             │
   │   patch 16×16, stride 8                                       │
   │   → dense feature grid  (Gh, Gw, 384)  L2-normalized          │
   └────────┬───────────────┬──────────────┬───────────────────────┘
            ▼               ▼              ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ Stage 1     │  │ Stage 2     │  │ Stage 3     │
   │ TileHead    │  │ TipAttn     │  │ TipBaseBow  │
   │  (CNN)      │→ │ BaseHead    │→ │ SternAttn   │
   │             │  │ (xformer)   │  │ Head (xfmr) │
   │  → tip px   │  │  → base px  │  │ → bow,stern │
   └─────────────┘  └─────────────┘  └─────────────┘
                                     │
              4 px / boat: bow, stern, base, tip
                                     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │   SINGLE-FRAME SOLVER  (solve_iterative_drop)                 │
   │   scipy.optimize.least_squares + iterative outlier drop       │
   │   → camera (6-DOF) + per-boat (4-DOF) in metric world         │
   └──────────────────────────────────────────────────────────────┘
```

---

## 1. DINOv3 feature extraction (frozen backbone)

**Why this is the foundation.** DINOv3 was pretrained
self-supervised on LVD-1689M (1.69 B images, no labels). Its
dense features already separate boats / sails / water / sky with
no task supervision. Every head we train is a tiny decoder on
top — we never touch the backbone.

**Model.** `facebook/dinov3-vits16-pretrain-lvd1689m` — Vision
Transformer Small / patch 16, 384-D features.

**Per-frame extraction (`infer_tip_base_oos.py`):**

```
Input image  (1920 × 1080 BGR, uint8)
        │
        ▼
[1] resize so long-side ≤ 1808 px and both sides multiple of 16
        → typical (1808, 1008)  RGB
        │
        ▼
[2] normalize with ImageNet mean / std,  cast to FP16
        │
        ▼
[3] DINOv3 forward   (batch=1, on GPU)
    patch_embed stride overridden 16 → 8   (overlapping patches)
        │
        ▼
[4] drop CLS + register tokens, reshape to grid
        → feat shape (Gh, Gw, 384)
          for 1808×1008:  Gh = (1008-16)/8+1 = 125
                          Gw = (1808-16)/8+1 = 225
        │
        ▼
[5] L2-normalize each cell  →  unit-length 384-D vectors
        │
        ▼
Output: dense feature grid  (125, 225, 384)  ≈ 28 k cells
        cached for the rest of the pipeline
```

**Why stride 8 instead of the native 16:** halves the cell pitch
in each axis → ~4× more cells, each tip / base / bow / stern
gets multiple feature samples nearby. Costs one extra forward
pass; everything downstream becomes cheaper because per-cell
predictions are sharper.

![Visual 1 — DINOv3 PCA features](presentation_visuals/01_dinov3_features.png)

---

## 2. Stage 1 — Tip detection (TileHead, CNN)

**Goal.** Locate every mast tip in the frame. This is the seed
for the rest of the pipeline; bases and bow/stern are all
conditioned on a detected tip.

### Architecture

`TileHead(in_dim=384, hidden=64)` — a small fully-convolutional
tower. Translation-equivariant, no positional encoding.

```
Input:  feature window  (B, 384, 12, 12)        ─┐
                                                 │
        Conv 1×1   384 → 64                      │
        BatchNorm + ReLU                         │
                                                 │
        Conv 3×3   64 → 64    (pad 1)            │
        BatchNorm + ReLU                         │   ~106 K params
                                                 │
        Conv 3×3   64 → 64    (pad 1)            │
        BatchNorm + ReLU                         │
                                                 │
        Conv 3×3   64 → 64    (pad 1)            │
        ReLU                                     │
                                                 │
        Conv 1×1   64 → 1                        │
                                                ─┘
Output: per-cell logit  (B, 1, 12, 12)
        sigmoid → tip-or-not heatmap, peak at 1.0 = tip centre
```

### Training data (`train_tip_tiles.py`)

- Sample 12×12 windows from labeled boat crops.
- Target = max-merged Gaussian heatmap (σ ≈ 1.5 cells) over
  every tip that falls inside the window.
- 50 % "negative" tiles (no tip) per batch, plus a hard-negative
  cache of full-frame tiles known to confuse the model.
- BCE loss, optionally focal-weighted.

### Inference (sliding window over the whole frame)

```
For every (y, x) tile origin with stride 2:
    crop  feat[y:y+12, x:x+12]
    →  TileHead  →  sigmoid heat 12×12
    max-merge into a global  (Gh, Gw)  heatmap

Result: dense per-cell tip-probability map  (125, 225)
        │
        ▼
Cluster peaks:
    DBSCAN over cells where heat ≥ 0.20
    in each cluster: local-maxima inside its bbox,
       merge any peak < min_peak_dist of an existing peak
    → list of (cell_x, cell_y, score) tip detections
        │
        ▼
Map cell → source-frame pixel via stride 8 + the original
resize ratio:
    src_x = (cell_x · 8 + 7.5) · sx_src
    src_y = (cell_y · 8 + 7.5) · sy_src
```

**Output of Stage 1:** N tip points in source-frame pixels.
For frame_0500 we get 6 tips, one per boat.

![Visual 2 — TileHead heatmap + peaks](presentation_visuals/02_tip_heatmap.png)

---

## 3. Stage 2 — Base detection (TipAttnBaseHead, transformer)

**Goal.** For each tip, find the corresponding mast base.

**Why a CNN won't cut it:** the mast base can be 6–25 cells away
from the tip (depending on heel + camera distance), and which
direction depends on global context (heel direction, occlusion
by sail). Conv receptive fields would have to be huge; a tiny
transformer gets it natively.

### Architecture

`TipAttnBaseHead(in_dim=384, hidden=128, n_heads=4, n_layers=2,
win_h=34, win_w=21, tip_row=2, tip_col=10)`

```
Input:  feature window  (B, 384, 34, 21)        ←  21·34 = 714 tokens
        tip is PINNED at (row=2, col=10) of every window

[1]  Conv 1×1     384 → 128             projection
[2]  reshape to sequence            (B, 714, 128)
[3]  +  2D sin/cos positional encoding   (per-cell row/col)
[4]  LayerNorm
[5]  Transformer encoder × 2
        d_model = 128
        n_heads = 4
        FFN dim = 256
        activation = GELU
        pre-norm                            ← every cell sees
                                              every other cell
[6]  pull tip-cell feature  x[:, tip_idx, :]   (B, 128)
     concat to every cell                       (B, 714, 256)
[7]  MLP head:  LayerNorm → Linear 256→128 → GELU → Linear 128→1
[8]  reshape to spatial                          (B, 1, 34, 21)
                                                          │
Output: per-cell base logit                              ▼
        sigmoid → base-or-not heatmap, argmax cell = base
```

~410 K parameters — still tiny relative to the backbone.

### Window placement

The 21×34 cell window is **anchored on the tip**, not on the
image. Tip sits at row 2 (near the top), column 10 (centre).
The window extends 32 cells downward → covers the whole boat
under the mast.

```
                 col 0          col 10         col 20
            ┌──────────────────────┬──────────────────────┐
   row  0   │                      │                      │
   row  1   │                      │                      │
   row  2   │          ────────tip ★──────────            │   ← pinned
   row  3   │                                             │
            │                                             │
            │       (boat lives in here, mostly below tip)│
            │                                             │
   row 33   │                                             │
            └─────────────────────────────────────────────┘
                            21 cells wide  ·  34 cells tall
                                ≈ 168 × 272 image px
```

### Training (`train_base_attn.py`)

- Labels: hand-clicked (tip, base) pairs on ~hundreds of frames.
- For each labeled pair: cut a 21×34 cell window around the tip,
  Gaussian target (σ = 1.0 cell) at the base cell, BCE loss.
- 300 epochs, batch 64, Adam (β₁=0.9, β₂=0.999, wd=1e-4),
  random horizontal mirror augmentation.

### Inference

For each Stage-1 tip:
1. Extract 21×34 window from cached features (zero-pad if near
   image edge).
2. Single forward pass → 34×21 sigmoid heatmap.
3. Argmax cell → 3×3 weighted centroid for sub-cell sharpness.
4. Map cell → source-frame pixel.

**Output of Stage 2:** one base point per tip.

---

## 4. Stage 3 — Bow / stern detection (TipBaseBowSternAttnHead, transformer)

**Goal.** For each (tip, base) pair, find bow and stern. This
is what gives the boat its **orientation** in the image.

**Why both signals matter:** the tip vector encodes which way
the boat is heeling; the base vector anchors where the hull is.
Together they pin the boat's heading; either one alone is
ambiguous (a vertical mast tells you nothing about which end is
the bow).

### Architecture

Same backbone as Stage 2, two changes only:

```
Input:  feature window  (B, 384, 34, 21)
        tip pinned at (row=2, col=10)
        base at variable cell  (passed in as base_idx)

[1-5]  same as Stage 2:
       Conv 1×1 → reshape → +pos-enc → LayerNorm → 2 transformer layers
                                                            (B, 714, 128)
[6]    conditioning vector =
            mean( x[:, tip_idx, :],  x[:, base_idx, :] )    (B, 128)
       concat to every cell                                 (B, 714, 256)
[7]    MLP head:  LayerNorm → Linear 256→128 → GELU →
                  Linear 128→2     ← TWO output channels
[8]    reshape                                              (B, 2, 34, 21)
                                                                 │
Output:  channel 0 = bow heatmap                                ▼
         channel 1 = stern heatmap
         peak per channel = bow / stern cell
```

~410 K parameters, same scale as Stage 2.

### Training (`train_bowstern_attn.py`)

- Labels: hand-clicked (tip, base, bow, stern) tuples.
- Filter: drop pairs where tip ≈ bow ≈ stern (the labeller's
  "unsure" sentinel) or tip-base distance < 10 px.
- Window cut around the tip (same as Stage 2) but the trainer
  passes in the **gathered base cell index** so the model knows
  where to take its second conditioning vector from.
- Two-channel BCE loss; otherwise same optimizer / mirror /
  schedule as Stage 2.

### Inference

For each (tip, base) pair from Stages 1+2:
1. Extract 21×34 window, compute base cell index inside it.
2. One forward pass → (2, 34, 21) heatmaps.
3. Argmax + 3×3 sub-cell centroid per channel.
4. Map cells → source-frame pixels.

**Output of Stage 3:** bow + stern per boat.

![Visual 3 — Stage 1/2/3 inside one tip's window](presentation_visuals/03_window_panels.png)

### End of pretraining pipeline

For each detected boat we now have **4 source-frame pixels**:

```
boat_meas_kpts = [  bow,    stern,    mast_base,    mast_top  ]
                   (px)     (px)       (px)          (px)
```

This is the entire input to the geometric solver.

![Visual 4 — Composite per-boat keypoints](presentation_visuals/04_composite_keypoints.png)

---

## 5. Single-frame solver (`solve_camera_from_frame.py`)

Takes the per-boat keypoint quadruples and recovers a metric 3D
scene: **camera pose + per-boat ground positions + heel +
focal length**, all from one image.

### World model

```
Coordinate system:  +x right,  +y forward (away from camera),
                    +z up.   Water plane at z = 0.

Boat template (rigid, in metric world coords):
  bow        (-HULL_L/3,        0,   0)         on water
  stern      ( 2·HULL_L/3,      0,   0)         on water
  mast_base  ( 0,               0,   0.7)       boom height
  mast_top   ( 0,               0,   0.7 + MAST_H)
                                                   │
HULL_L = 4.2 m,  MAST_H = 5.2 m  (known dinghy class).
```

![Visual 5 — Solver geometry](presentation_visuals/05_solver_geometry.png)

### Unknowns

```
Per-frame (6 DOF):    pitch, h, yaw, cam_x, cam_y, focal f
Per boat  (4 DOF):    x, y, yaw, heel
                                                   ↳ N boats per frame
                                                     → 6 + 4N unknowns
                                                       8N reprojection residuals
```

Roll is fixed at 0 (drone keeps the horizon level enough that
this is a fair simplification). Principal point fixed at image
centre.

### Forward model

For each (boat, keypoint):

```
1.  place template into world via yaw + heel + (x, y)
2.  rotate into camera with R(pitch, yaw)
3.  pinhole project with focal f, principal point (cx, cy)
4.  residual = predicted_px − detected_px
```

### Initialization

- `pitch_init = 0.3` rad, `h_init = 20` m, `focal_init = 1500` px,
  `cam = (0, 0, h_init)`.
- For each boat: back-project the **detected base pixel** as a
  ray with the assumed pitch + focal, intersect the water plane
  z = 0 → starting (x, y) for that boat.
- Yaw and heel start at 0.

This avoids cold starts in metric space: every boat begins at a
plausible water-plane location instead of "somewhere".

### Residuals (`_residuals_free_f`)

Three blocks, in order:

**(a) Reprojection residuals — the dominant signal**

```
For every active (boat, keypoint):
    r = ( predicted_px(boat, kpt) − detected_px(boat, kpt) ) · w_kpt
```

`w_kpt` is a per-landmark weight tuple, default `(1, 1, 1, 1)`.
Lets the chain solver downweight noisy bow/stern when needed
(not used in the basic single-frame solve).

**(b) Heel prior with deadzone — soft physics regulariser**

```
For each boat:
    if  |heel|  ≤  deadzone:        r_heel = 0
    else:                            r_heel = w_heel · (heel − sign(heel)·deadzone)

defaults:  w_heel = 10 px / rad,  deadzone = 20°
```

So a 20 ° heel costs nothing; a 30 ° heel costs ~1.7 px-equivalent.
Effect: when the image data underdetermines heel (low boat,
mast occluded), heel collapses to 0. When the image data wants
a real heel, it pays a small price and stays.

**(c) Optional focal prior — soft pull on focal length**

```
if focal_prior_target is set and focal_prior_weight > 0:
    r_focal = w_f · ( focal − focal_prior_target )
```

Off by default in single-frame mode. Used by the multi-frame
chain solver to keep focal smooth across cuts.

### Bounds (hard constraints)

```
Camera
  pitch  ∈ ( 0.01,  π/2 − 0.01 )       must look downward
  h      ∈ [ h_min,   h_max ]           default [5, 200] m
  yaw    ∈ [ −π,    π ]                 unconstrained mod 2π
  cam_x  ∈ [ cam_xy_min,  cam_xy_max ]  default ±200 m
  cam_y  ∈ [ cam_xy_min,  min(cam_xy_max, cam_y_max) ]
                                        cam_y_max = 0 ⇒ camera south of fleet
  focal  ∈ [ focal_min,  focal_max ]    default unbounded positive

Per boat
  heel   ∈ [ −60°,  60° ]               kills the "39° → 80° drift" failure mode
  x, y   unbounded                       the only metric anchor is mast height
```

### Optimization

`scipy.optimize.least_squares`, trust-region Levenberg-Marquardt
(`method="trf"`), `loss="linear"`, `max_nfev=200`.

**Jacobian sparsity pattern** declared explicitly:

```
Each reprojection row depends only on:
   - the 6 camera params
   - that boat's own 4 params
Heel rows depend only on heel of that boat.
Focal-prior row depends only on focal.

→ scipy's FD Jacobian groups columns that don't share rows
  → forward-evaluations per Jacobian drop from O(p) to ~p/3
```

For a 6-boat frame: 30 unknowns, 48 reprojection residuals + 6
heel + 0 focal = 54 residuals. Solves in <0.1 s on CPU.

### Iterative-drop refinement (`solve_iterative_drop`)

Wraps the optimizer with outlier removal:

```
active_mask = all-True  (n_boats × 4)

repeat:
    1. solve  (heel_prior_weight = 0 here — drops driven purely
                                            by reprojection error)
    2. compute per-keypoint pixel error with new params
    3. excess = error − thresh_per_kpt        ← bow/stern and
                                                base/tip can have
                                                different thresholds
    4. if max(excess) ≤ 0 : DONE
    5. drop the worst-offending keypoint
       (unless that boat already has only 2 active points)
    6. refit with the smaller active_mask

final: refit ONCE with heel_prior_weight ON
       — only collapses heels not pinned by surviving data
```

The heel prior is held back during the drop loop on purpose: we
don't want to drop a tip just because the heel prior was fighting
it. Once the surviving keypoints converge cleanly, then we apply
the prior to settle any remaining ambiguity.

### Output

```
{
  pitch, h, yaw, cam_x, cam_y, focal,            ← camera
  rms,                                            ← global RMS px
  boats: [
    { x, y, yaw, heel_rad, heel_deg, err },     ← per boat
    ...
  ],
  active_mask, dropped_log, per_point_errs       ← drop diagnostics
}
```

### Concrete example — frame_0500

```
6 boats in.
iter 1: dropped boat1.base (err 18.35 px > thresh 8); refit
iter 2: dropped boat1.tip  (err 12.79 px > thresh 8); refit
iter 3: dropped boat3.tip  (err  9.10 px > thresh 8); refit
iter 4: max active err 7.28 px ≤ thresh — done
final refit with heel_prior_weight = 10, deadzone = 20°

Result:
  pitch =  7.1°
  h     = 22.2 m
  yaw   =  8.7°
  focal = 4591 px
  RMS   =  2.45 px           ← over 21 surviving keypoints
```

Three keypoints dropped came from one occluded boat (boat 1) and
one partially-hidden mast (boat 3.tip). The remaining 21 land
to within 2.5 px of perfect on a 1920×1080 image.

![Visual 6 — Per-frame solver result](presentation_visuals/06_solver_result.png)

---

## 6. Talking-point recap

| Stage | Role | Trained? | Cost per frame |
|---|---|---|---|
| DINOv3 ViT-S/16 | Dense feature backbone | **No** (frozen, pretrained on 1.69 B images) | 1 forward pass |
| TileHead (CNN) | Per-cell tip probability | Yes (small) | sliding window, ~14 k tiles |
| TipAttnBaseHead | Tip-conditioned base | Yes | 1 forward per tip (~6/frame) |
| TipBaseBowSternAttnHead | (tip, base)-conditioned bow + stern | Yes | 1 forward per (tip, base) |
| Single-frame solver | Reproj. error + heel prior + focal prior; iterative outlier drop | No (geometry) | <0.1 s on CPU |

**Key takeaways**

1. Almost all the visual heavy lifting is done by the **frozen
   DINOv3 backbone**. The trained heads are tiny (~100–400 K
   params each) — possible only because the features they
   consume are already informative.
2. Each stage is **conditioned on the previous one**: tip →
   (tip)-base → (tip, base)-bow/stern. A failure at stage 1
   takes the others with it; a failure at stage 2 still gives
   you a tip.
3. The solver is **deliberately minimal**: just reprojection +
   one soft prior (heel) + iterative outlier drop. The complex
   priors (camera continuity, fleet speed, focal log-acceleration,
   pitch-weighted regularisers, …) live in the **chain solver**
   that wraps this single-frame solver across a sequence — out
   of scope for this presentation.

2-Minute Presentation Script                                                                                                                                                                                                    
                                               
  Total time: ~2:00 (≈ 280 words at conversational pace)                                                                                                                                                                          
  Speakers: Quinn (lead, opener) → Nishani (pretraining) → Ali (solver + close)
                                                                                                                                                                                                                                  
  ---                                                             
  QUINN — opener  (0:00 – 0:40,  ~95 words)                                                                                                                                                                                       
                                                                                                                                                                                                                                  
  [Slide: a regatta drone shot or Visual 4]
                                                                                                                                                                                                                                  
  "Hi everyone, I'm Quinn, this is Ali and Nishani. Ali and I are co-captains of Brown Sailing, which is exactly why we picked this project. At most regattas there's a drone overhead recording the race — and afterwards we have hours of footage to go through. We wanted to turn that drone footage into a real top-down map: every boat's position, every frame. The catch is we get one camera, no GPS on the boats, and no calibration. So we built a pipeline that recovers the full 3D scene from pixels alone."                                                                                                                                                                                                                         
  NISHANI — pretraining pipeline  (0:40 – 1:20,  ~95 words)       
                                                           
  [Slide: Visual 1 — DINOv3 PCA features. Then click through Visuals 2 → 3 → 4 as you talk.]
                                                                                                                                                                                                                                  
  "To recover the scene we first detect four keypoints per boat: bow, stern, mast base, and mast top. The backbone is DINOv3 — a frozen vision transformer pretrained self-supervised on 1.7 billion images. As you can see [click
   — Visual 1], it already separates boats, sails, and water with no fine-tuning. On top of that we train three tiny heads in sequence: [click] a small CNN locates every mast tip; [click] a transformer conditioned on the tip  
  predicts the base; [click] and a third transformer, conditioned on both, predicts bow and stern. Each head is under half a million parameters — possible only because DINOv3 is doing the heavy lifting underneath."            
                                                                  
                                                             
  ALI — single-frame solver + close  (1:20 – 2:00,  ~95 words)    
                                                                                                            
  "Now we have four pixels per boat. To turn pixels into meters we use one fact: we know the boat. These are college 420s and FJs — hull 4.2 m, mast 5.2 m. That gives us absolute scale. We then solve for the camera's pitch, height,   
  yaw, and focal length, plus every boat's x, y, yaw, and heel — all from a single frame — by minimizing reprojection error in a loop with iterative outlier dropping for bad predictions. 
  On this OOS frame we recovered six boats with 2.4-pixel accuracy on a 1080p image — and on the here is the top-down view with 20m grids.                                                                                                                 
                                                                                                                                           
                                                                                                                                                                                                                                  
✻ Worked for 48s                               