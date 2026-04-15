"""Full-frame keypoint labeler (multi-class).
Label keypoints on sailboats (4 pts: mast tip, mast base, bow, stern)
and marks (top/bottom).

Controls:
  0-4: select class (0=sailboat, 1=start, 2=finish, 3=yellow, 4=red)
  T: mast tip mode (red)
  B: mast base mode (green)
  W: bow mode (cyan) — class 0 only
  E: stern mode (yellow) — class 0 only
  L: link mode — click bow/stern, then click mast base to connect
  Right click: delete nearest point
  A/D: prev/next frame
  P: auto-pair tips/bases only (no bow/stern auto-link)
  G: manual group mode — click near tip, base, bow, stern to group them
  U: ungroup — click near any grouped point to remove its group
  S: save
  Q: save and quit
  C: clear all points on current frame
"""

import cv2
import numpy as np
import json
import os
import glob

IMG_DIR = "mast_box_labels/images"
LABEL_FILE = "mast_frame_labels.json"

CLASS_NAMES = ['sailboat', 'start_boat', 'finish_boat', 'yellow_mark', 'red_mark']
CLASS_COLORS = [
    (0, 0, 255),    # sailboat: red
    (255, 0, 128),  # start_boat: magenta
    (255, 128, 0),  # finish_boat: orange
    (0, 255, 255),  # yellow_mark: yellow
    (0, 100, 255),  # red_mark: dark orange
]

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
print(f"{len(images)} frames")

if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE) as f:
        labels = json.load(f)
else:
    labels = {}

current = [0]
mode = ['tip']  # 'tip', 'base', 'bow', 'stern', 'group', 'link'
cur_class = [0]  # current class index
tips = []       # list of (x, y, class_id) in pixels
bases = []
bows = []       # class 0 only
sterns = []     # class 0 only
bow_links = {}  # bow_idx -> base_idx
stern_links = {}  # stern_idx -> base_idx
pairs = []      # list of (tip_idx, base_idx) after pairing
boat_groups = []  # list of {tip, base, bow, stern} index groups
group_build = {}  # partial group being built: keys collected so far
link_sel = [None]  # ('bow'|'stern', idx) — selected point awaiting base click
DISPLAY_SCALE = 1  # full res

# Colors per group for visualization
GROUP_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 50), (50, 200, 150), (150, 50, 200),
    (255, 200, 150), (150, 255, 200), (200, 150, 255),
]


def get_frame_data():
    fname = os.path.basename(images[current[0]])
    if fname in labels:
        lab = labels[fname]
        return (lab.get('tips', []), lab.get('bases', []),
                lab.get('bows', []), lab.get('sterns', []),
                lab.get('bow_links', {}), lab.get('stern_links', {}),
                lab.get('pairs', []), lab.get('boat_groups', []))
    return [], [], [], [], {}, {}, [], []


def save_frame_data():
    fname = os.path.basename(images[current[0]])
    labels[fname] = {
        'tips': [[float(x), float(y), int(c)] for x, y, c in tips],
        'bases': [[float(x), float(y), int(c)] for x, y, c in bases],
        'bows': [[float(x), float(y)] for x, y in bows],
        'sterns': [[float(x), float(y)] for x, y in sterns],
        'bow_links': {str(k): int(v) for k, v in bow_links.items()},
        'stern_links': {str(k): int(v) for k, v in stern_links.items()},
        'pairs': pairs,
        'boat_groups': boat_groups,
    }


def load_frame():
    global tips, bases, bows, sterns, bow_links, stern_links, pairs, boat_groups
    t, b, bw, st, bl, sl, p, bg = get_frame_data()
    tips = [(pt[0], pt[1], pt[2] if len(pt) > 2 else 0) for pt in t]
    bases = [(pt[0], pt[1], pt[2] if len(pt) > 2 else 0) for pt in b]
    bows = [tuple(pt[:2]) for pt in bw]
    sterns = [tuple(pt[:2]) for pt in st]
    bow_links = {int(k): int(v) for k, v in bl.items()}
    stern_links = {int(k): int(v) for k, v in sl.items()}
    pairs = list(p)
    boat_groups = list(bg)
    link_sel[0] = None


def auto_pair():
    """Pair tips/bases per class, then group bow/stern with nearest mast pair for class 0."""
    global pairs, boat_groups
    pairs = []
    boat_groups = []
    if not tips or not bases:
        return

    from scipy.optimize import linear_sum_assignment

    # Pair tips and bases within each class
    all_classes = set(t[2] for t in tips) | set(b[2] for b in bases)
    for cls in sorted(all_classes):
        cls_tips = [(i, t) for i, t in enumerate(tips) if t[2] == cls]
        cls_bases = [(i, b) for i, b in enumerate(bases) if b[2] == cls]
        if not cls_tips or not cls_bases:
            continue

        n = max(len(cls_tips), len(cls_bases))
        cost = np.full((n, n), 1e6)
        for ti_local, (ti_global, tip) in enumerate(cls_tips):
            for bi_local, (bi_global, base) in enumerate(cls_bases):
                dist = np.sqrt((tip[0]-base[0])**2 + (tip[1]-base[1])**2)
                cost[ti_local][bi_local] = dist

        row_idx, col_idx = linear_sum_assignment(cost)
        for ti_local, bi_local in zip(row_idx, col_idx):
            if ti_local < len(cls_tips) and bi_local < len(cls_bases) and cost[ti_local][bi_local] < 500:
                pairs.append([int(cls_tips[ti_local][0]), int(cls_bases[bi_local][0])])

    # Bow/stern links are manual only — not auto-paired


def redraw():
    img = cv2.imread(images[current[0]])
    vis = img.copy()
    h, w = vis.shape[:2]

    fname = os.path.basename(images[current[0]])
    n_labeled = sum(1 for f, l in labels.items() if l.get('tips') and l.get('bases'))

    # Draw tips (class-colored, darker shade)
    for i, (x, y, c) in enumerate(tips):
        color = CLASS_COLORS[int(c)]
        cv2.circle(vis, (int(x), int(y)), 8, color, 2)
        cv2.circle(vis, (int(x), int(y)), 3, color, -1)
        prefix = CLASS_NAMES[int(c)][0].upper()
        cv2.putText(vis, f"{prefix}T{i}", (int(x)+10, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw bases (class-colored, brighter shade)
    for i, (x, y, c) in enumerate(bases):
        color = CLASS_COLORS[int(c)]
        bright = tuple(min(255, v + 80) for v in color)
        cv2.circle(vis, (int(x), int(y)), 8, bright, 2)
        cv2.circle(vis, (int(x), int(y)), 3, bright, -1)
        prefix = CLASS_NAMES[int(c)][0].upper()
        cv2.putText(vis, f"{prefix}B{i}", (int(x)+10, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bright, 1)

    # Draw bows (cyan)
    for i, (x, y) in enumerate(bows):
        cv2.circle(vis, (int(x), int(y)), 8, (255, 255, 0), 2)
        cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 0), -1)
        cv2.putText(vis, f"W{i}", (int(x)+10, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Draw sterns (yellow-green)
    for i, (x, y) in enumerate(sterns):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 255), 2)
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.putText(vis, f"E{i}", (int(x)+10, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw bow->base links
    for wi, bi in bow_links.items():
        if wi < len(bows) and bi < len(bases):
            bx, by = bows[wi]
            mx, my = bases[bi][0], bases[bi][1]
            cv2.line(vis, (int(bx), int(by)), (int(mx), int(my)), (255, 255, 0), 2)

    # Draw stern->base links
    for si, bi in stern_links.items():
        if si < len(sterns) and bi < len(bases):
            sx, sy = sterns[si]
            mx, my = bases[bi][0], bases[bi][1]
            cv2.line(vis, (int(sx), int(sy)), (int(mx), int(my)), (0, 255, 255), 2)

    # Draw link selection highlight
    if link_sel[0]:
        pt_type, pt_idx = link_sel[0]
        if pt_type == 'bow' and pt_idx < len(bows):
            px, py = bows[pt_idx]
            cv2.circle(vis, (int(px), int(py)), 16, (255, 255, 255), 3)
        elif pt_type == 'stern' and pt_idx < len(sterns):
            px, py = sterns[pt_idx]
            cv2.circle(vis, (int(px), int(py)), 16, (255, 255, 255), 3)

    # Draw grouped boats with colored connections
    for gi, g in enumerate(boat_groups):
        gc = GROUP_COLORS[gi % len(GROUP_COLORS)]
        pts = []
        if g['tip'] >= 0 and g['tip'] < len(tips):
            pts.append(('T', tips[g['tip']][:2]))
        if g['base'] >= 0 and g['base'] < len(bases):
            pts.append(('B', bases[g['base']][:2]))
        if g['bow'] >= 0 and g['bow'] < len(bows):
            pts.append(('W', bows[g['bow']]))
        if g['stern'] >= 0 and g['stern'] < len(sterns):
            pts.append(('E', sterns[g['stern']]))

        # Draw lines between all points in group
        for pi in range(len(pts)):
            for pj in range(pi+1, len(pts)):
                p1 = pts[pi][1]; p2 = pts[pj][1]
                cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), gc, 2)

        # Draw group number at centroid
        if pts:
            cx = int(np.mean([p[1][0] for p in pts]))
            cy = int(np.mean([p[1][1] for p in pts]))
            cv2.putText(vis, f"G{gi}", (cx-10, cy-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, gc, 2)

        # Highlight grouped points with colored rings
        for label, pt in pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 12, gc, 2)

    # Draw partial group being built
    if group_build:
        for key, info in group_build.items():
            cv2.circle(vis, (int(info[0]), int(info[1])), 15, (255, 255, 255), 2)
        remaining = [k for k in ['tip','base','bow','stern'] if k not in group_build]
        cv2.putText(vis, f"GROUPING: need {', '.join(remaining)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Status
    cls_color = CLASS_COLORS[cur_class[0]]
    MODE_NAMES = {'tip': 'TIP (T)', 'base': 'BASE (B)', 'bow': 'BOW (W)', 'stern': 'STERN (E)', 'group': 'GROUP (G)', 'link': 'LINK (L)'}
    mode_text = MODE_NAMES.get(mode[0], mode[0])
    cls_text = f"{CLASS_NAMES[cur_class[0]]}({cur_class[0]})"
    cv2.putText(vis, f"[{current[0]+1}/{len(images)}] {fname} | Class: {cls_text} | Mode: {mode_text} | "
                f"{len(tips)}T {len(bases)}B {len(bows)}W {len(sterns)}E | {len(boat_groups)}G {len(pairs)}P | {n_labeled} done",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cls_color, 2)
    cv2.putText(vis, "T/B/W/E=mode | L=link bow/stern->base | G=group | P=auto-pair | RClick=del | A/D=nav | S=save | Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow('Mast Frame Label', vis)


def find_nearest_point(x, y, threshold=40):
    """Find nearest point of any type. Returns (type, index, dist)."""
    best_dist = threshold
    best_type = None
    best_idx = -1

    for i, (px, py, c) in enumerate(tips):
        if c != 0: continue  # only class 0 for grouping
        dist = np.sqrt((x-px)**2 + (y-py)**2)
        if dist < best_dist:
            best_dist = dist; best_type = 'tip'; best_idx = i

    for i, (px, py, c) in enumerate(bases):
        if c != 0: continue
        dist = np.sqrt((x-px)**2 + (y-py)**2)
        if dist < best_dist:
            best_dist = dist; best_type = 'base'; best_idx = i

    for i, (px, py) in enumerate(bows):
        dist = np.sqrt((x-px)**2 + (y-py)**2)
        if dist < best_dist:
            best_dist = dist; best_type = 'bow'; best_idx = i

    for i, (px, py) in enumerate(sterns):
        dist = np.sqrt((x-px)**2 + (y-py)**2)
        if dist < best_dist:
            best_dist = dist; best_type = 'stern'; best_idx = i

    return best_type, best_idx, best_dist


def mouse_cb(event, x, y, flags, param):
    global tips, bases, bows, sterns, pairs, boat_groups, group_build

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode[0] == 'link':
            if link_sel[0] is None:
                # First click: select a bow or stern point
                best_dist = 40
                best_type = None
                best_idx = -1
                for i, (px, py) in enumerate(bows):
                    dist = np.sqrt((x-px)**2 + (y-py)**2)
                    if dist < best_dist:
                        best_dist = dist; best_type = 'bow'; best_idx = i
                for i, (px, py) in enumerate(sterns):
                    dist = np.sqrt((x-px)**2 + (y-py)**2)
                    if dist < best_dist:
                        best_dist = dist; best_type = 'stern'; best_idx = i
                if best_type:
                    link_sel[0] = (best_type, best_idx)
                    print(f"Selected {best_type} {best_idx} — now click a mast base")
            else:
                # Second click: select a mast base to link to
                best_dist = 40
                best_idx = -1
                for i, (px, py, c) in enumerate(bases):
                    dist = np.sqrt((x-px)**2 + (y-py)**2)
                    if dist < best_dist:
                        best_dist = dist; best_idx = i
                if best_idx >= 0:
                    pt_type, pt_idx = link_sel[0]
                    if pt_type == 'bow':
                        bow_links[pt_idx] = best_idx
                    else:
                        stern_links[pt_idx] = best_idx
                    print(f"Linked {pt_type} {pt_idx} -> base {best_idx}")
                link_sel[0] = None
            return

        if mode[0] == 'group':
            # Group mode: click near points to add them to current group
            pt_type, pt_idx, dist = find_nearest_point(x, y)
            if pt_type and pt_type not in group_build:
                # Get the point coordinates
                if pt_type == 'tip':
                    group_build['tip'] = (tips[pt_idx][0], tips[pt_idx][1], pt_idx)
                elif pt_type == 'base':
                    group_build['base'] = (bases[pt_idx][0], bases[pt_idx][1], pt_idx)
                elif pt_type == 'bow':
                    group_build['bow'] = (bows[pt_idx][0], bows[pt_idx][1], pt_idx)
                elif pt_type == 'stern':
                    group_build['stern'] = (sterns[pt_idx][0], sterns[pt_idx][1], pt_idx)

                # If all 4 collected, create the group
                if len(group_build) == 4:
                    new_group = {
                        'tip': group_build['tip'][2],
                        'base': group_build['base'][2],
                        'bow': group_build['bow'][2],
                        'stern': group_build['stern'][2],
                    }
                    # Also create the tip-base pair
                    pair = [new_group['tip'], new_group['base']]
                    if pair not in pairs:
                        pairs.append(pair)
                    boat_groups.append(new_group)
                    group_build = {}
                    print(f"Group created: {new_group}")
            return

        if mode[0] == 'tip':
            tips.append((x, y, cur_class[0]))
        elif mode[0] == 'base':
            bases.append((x, y, cur_class[0]))
        elif mode[0] == 'bow':
            bows.append((x, y))
        elif mode[0] == 'stern':
            sterns.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        best_dist = 30
        best_type = None
        best_idx = -1

        for i, (px, py, c) in enumerate(tips):
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < best_dist:
                best_dist = dist; best_type = 'tip'; best_idx = i

        for i, (px, py, c) in enumerate(bases):
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < best_dist:
                best_dist = dist; best_type = 'base'; best_idx = i

        for i, (px, py) in enumerate(bows):
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < best_dist:
                best_dist = dist; best_type = 'bow'; best_idx = i

        for i, (px, py) in enumerate(sterns):
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < best_dist:
                best_dist = dist; best_type = 'stern'; best_idx = i

        if best_type == 'tip':
            tips.pop(best_idx)
        elif best_type == 'base':
            bases.pop(best_idx)
            # Remove any links pointing to this base and reindex
            bow_links = {k: (v if v < best_idx else v-1) for k, v in bow_links.items() if v != best_idx}
            stern_links = {k: (v if v < best_idx else v-1) for k, v in stern_links.items() if v != best_idx}
        elif best_type == 'bow':
            bows.pop(best_idx)
            # Remove link for this bow and reindex
            bow_links.pop(best_idx, None)
            bow_links = {(k if k < best_idx else k-1): v for k, v in bow_links.items()}
        elif best_type == 'stern':
            sterns.pop(best_idx)
            stern_links.pop(best_idx, None)
            stern_links = {(k if k < best_idx else k-1): v for k, v in stern_links.items()}
        if best_type:
            pairs = []
            boat_groups = []


cv2.namedWindow('Mast Frame Label', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mast Frame Label', 1280, 720)
cv2.setMouseCallback('Mast Frame Label', mouse_cb)

load_frame()

while True:
    redraw()
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        save_frame_data()
        with open(LABEL_FILE, 'w') as f:
            json.dump(labels, f, indent=2)
        n = sum(1 for l in labels.values() if l.get('tips') and l.get('bases'))
        print(f"Saved {n} labeled frames to {LABEL_FILE}")
        break

    elif key == ord('t'):
        mode[0] = 'tip'

    elif key == ord('b'):
        mode[0] = 'base'

    elif key == ord('w'):
        mode[0] = 'bow'
        cur_class[0] = 0  # bow/stern only for sailboats

    elif key == ord('e'):
        mode[0] = 'stern'
        cur_class[0] = 0

    elif key == ord('l'):
        mode[0] = 'link'
        link_sel[0] = None
        print("Link mode: click bow/stern, then click mast base")

    elif key == ord('g'):
        mode[0] = 'group'
        group_build = {}
        print("Group mode: click tip, base, bow, stern for one boat")

    elif key == ord('u'):
        # Ungroup: click near any point, remove its group
        # Just remove the last group for simplicity
        if boat_groups:
            removed = boat_groups.pop()
            # Also remove the pair
            pair = [removed['tip'], removed['base']]
            if pair in pairs:
                pairs.remove(pair)
            print(f"Removed group: {removed}")

    elif key == 27:  # Escape — cancel group build
        group_build = {}
        mode[0] = 'tip'
        print("Group cancelled")

    elif key == ord('p'):
        auto_pair()
        save_frame_data()

    elif key == ord('d'):
        save_frame_data()
        current[0] = min(current[0] + 1, len(images) - 1)
        load_frame()

    elif key == ord('a'):
        save_frame_data()
        current[0] = max(current[0] - 1, 0)
        load_frame()

    elif key == ord('s'):
        save_frame_data()
        with open(LABEL_FILE, 'w') as f:
            json.dump(labels, f, indent=2)
        n = sum(1 for l in labels.values() if l.get('tips') and l.get('bases'))
        print(f"Saved ({n} frames labeled)")

    elif key == ord('c'):
        tips = []
        bases = []
        bows = []
        sterns = []
        bow_links = {}
        stern_links = {}
        pairs = []
        boat_groups = []

    elif ord('0') <= key <= ord('4'):
        cur_class[0] = key - ord('0')
        print(f"Class: {CLASS_NAMES[cur_class[0]]} ({cur_class[0]})")

cv2.destroyAllWindows()
