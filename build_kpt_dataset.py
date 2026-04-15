"""Convert mast_frame_labels.json to YOLO pose training data.

4 keypoints per object: bow, stern, mast_tip, mast_base
- Sailboats (class 0): all 4 from labels
- Marks (class 1-4): tip/base only, bow/stern set to invisible (vis=0)

YOLO label format per line:
  class cx cy w h  kpt0_x kpt0_y kpt0_vis  kpt1_x kpt1_y kpt1_vis  ...

kpt0=bow, kpt1=stern, kpt2=mast_tip, kpt3=mast_base
"""

import json
import os
import random
import shutil

import cv2
import numpy as np

LABEL_FILE = "mast_frame_labels.json"
IMG_DIR = "mast_box_labels/images"
OUT_DIR = "mast_kpt_data"
VAL_RATIO = 0.15

CLASS_NAMES = ['sailboat', 'start_boat', 'finish_boat', 'yellow_mark', 'red_mark']


def main():
    with open(LABEL_FILE) as f:
        labels = json.load(f)

    frames = []
    for fname, lab in labels.items():
        tips = lab.get('tips', [])
        bases = lab.get('bases', [])
        pairs = lab.get('pairs', [])
        if not tips or not bases or not pairs:
            continue
        frames.append(fname)

    print(f"{len(frames)} frames with paired labels")

    random.seed(42)
    random.shuffle(frames)
    val_count = max(1, int(len(frames) * VAL_RATIO))
    val_frames = set(frames[:val_count])
    train_frames = set(frames[val_count:])
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    for split in ['train', 'val']:
        for sub in ['images', 'labels']:
            d = os.path.join(OUT_DIR, split, sub)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        cache = os.path.join(OUT_DIR, split, 'labels.cache')
        if os.path.exists(cache):
            os.remove(cache)

    total_labels = 0
    n_with_bs = 0
    n_without_bs = 0

    for fname in frames:
        lab = labels[fname]
        tips = lab['tips']
        bases = lab['bases']
        pairs = lab['pairs']
        bows_list = lab.get('bows', [])
        sterns_list = lab.get('sterns', [])
        boat_groups = lab.get('boat_groups', [])

        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        split = 'val' if fname in val_frames else 'train'
        stem = os.path.splitext(fname)[0]

        # Build lookup: for class 0 paired objects, find bow/stern from boat_groups
        # boat_groups: list of {tip, base, bow, stern} index dicts
        group_by_pair = {}
        for g in boat_groups:
            group_by_pair[(g['tip'], g['base'])] = g

        lines = []
        paired_tips = set()

        for ti, bi in pairs:
            tip = tips[ti]
            base = bases[bi]
            cls_id = int(tip[2]) if len(tip) > 2 else 0

            tx, ty = tip[0] / w, tip[1] / h
            bx, by = base[0] / w, base[1] / h

            # Get bow/stern for class 0
            bow_x, bow_y, bow_vis = 0.0, 0.0, 0
            stern_x, stern_y, stern_vis = 0.0, 0.0, 0

            if cls_id == 0:
                g = group_by_pair.get((ti, bi))
                if g and g.get('bow', -1) >= 0 and g['bow'] < len(bows_list):
                    bpt = bows_list[g['bow']]
                    bow_x, bow_y, bow_vis = bpt[0] / w, bpt[1] / h, 2
                if g and g.get('stern', -1) >= 0 and g['stern'] < len(sterns_list):
                    spt = sterns_list[g['stern']]
                    stern_x, stern_y, stern_vis = spt[0] / w, spt[1] / h, 2

                if bow_vis > 0 and stern_vis > 0:
                    n_with_bs += 1
                else:
                    n_without_bs += 1

            # Bounding box from all visible keypoints
            vis_xs = [tx, bx]
            vis_ys = [ty, by]
            if bow_vis > 0:
                vis_xs.append(bow_x); vis_ys.append(bow_y)
            if stern_vis > 0:
                vis_xs.append(stern_x); vis_ys.append(stern_y)

            pad = 0.02
            x1 = max(0, min(vis_xs) - pad)
            y1 = max(0, min(vis_ys) - pad)
            x2 = min(1, max(vis_xs) + pad)
            y2 = min(1, max(vis_ys) + pad)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1

            # 4 keypoints: bow, stern, mast_tip, mast_base
            line = (f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                    f"{bow_x:.6f} {bow_y:.6f} {bow_vis} "
                    f"{stern_x:.6f} {stern_y:.6f} {stern_vis} "
                    f"{tx:.6f} {ty:.6f} 2 "
                    f"{bx:.6f} {by:.6f} 2")
            lines.append(line)
            paired_tips.add(ti)

        # Unpaired tips (marks with single keypoint)
        for ti, tip in enumerate(tips):
            if ti in paired_tips:
                continue
            cls_id = int(tip[2]) if len(tip) > 2 else 0
            if cls_id == 0:
                continue

            tx, ty = tip[0] / w, tip[1] / h
            pad = 0.02
            x1n = max(0, tx - pad)
            y1n = max(0, ty - pad)
            x2n = min(1, tx + pad)
            y2n = min(1, ty + pad)
            cx = (x1n + x2n) / 2
            cy = (y1n + y2n) / 2
            bw_box = x2n - x1n
            bh_box = y2n - y1n

            # 4 keypoints: bow=invisible, stern=invisible, tip=visible, base=same as tip
            line = (f"{cls_id} {cx:.6f} {cy:.6f} {bw_box:.6f} {bh_box:.6f} "
                    f"0.000000 0.000000 0 "
                    f"0.000000 0.000000 0 "
                    f"{tx:.6f} {ty:.6f} 2 "
                    f"{tx:.6f} {ty:.6f} 2")
            lines.append(line)

        if not lines:
            continue

        # Write original
        shutil.copy2(img_path, os.path.join(OUT_DIR, split, 'images', fname))
        with open(os.path.join(OUT_DIR, split, 'labels', stem + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

        # Write mirror
        mirror = cv2.flip(img, 1)
        m_fname = stem + '_m.jpg'
        cv2.imwrite(os.path.join(OUT_DIR, split, 'images', m_fname), mirror)
        m_lines = []
        for line in lines:
            parts = line.split()
            cls_id = parts[0]
            cx = 1.0 - float(parts[1])
            cy = float(parts[2])
            bw_v = float(parts[3])
            bh_v = float(parts[4])
            # Mirror all x coordinates for 4 keypoints (12 values: x,y,v * 4)
            kpts = []
            for ki in range(4):
                kx = float(parts[5 + ki*3])
                ky = float(parts[6 + ki*3])
                kv = parts[7 + ki*3]
                if int(kv) > 0:
                    kx = 1.0 - kx
                kpts.extend([f"{kx:.6f}", f"{ky:.6f}", kv])
            m_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw_v:.6f} {bh_v:.6f} " + " ".join(kpts))

        with open(os.path.join(OUT_DIR, split, 'labels', stem + '_m.txt'), 'w') as f:
            f.write('\n'.join(m_lines) + '\n')

        total_labels += len(lines) * 2

    print(f"Wrote {total_labels} labels (with mirrors)")
    print(f"Sailboats with bow+stern: {n_with_bs}, without: {n_without_bs}")

    for split in ['train', 'val']:
        counts = {i: 0 for i in range(len(CLASS_NAMES))}
        label_dir = os.path.join(OUT_DIR, split, 'labels')
        for lf in os.listdir(label_dir):
            if not lf.endswith('.txt'):
                continue
            with open(os.path.join(label_dir, lf)) as f:
                for line in f:
                    cls = int(line.split()[0])
                    counts[cls] += 1
        print(f"  {split}: {dict(counts)}")


if __name__ == '__main__':
    main()
