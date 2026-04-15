"""Bundle solver over 5-frame windows.

Camera moves linearly: pos(t) = (x0 + dx*dt, y0 + dy*dt, h0 + dh*dt)
Camera rotation is derived from the position + a fixed pitch angle.
f is fixed at 500.

Per boat per frame: position (x, y) on water + yaw.
Uses mast heights to estimate distances, then refines with full projection.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from projection_solver import boat_model, project

FIXED_F = 500
MAST_H = 5.89
HULL_L = 4.2


def estimate_distances(boats, f=FIXED_F):
    """Rough distance from mast height in pixels."""
    dists = []
    for b in boats:
        obs = b["obs"]
        mast_px = np.sqrt((obs["mast_top"][0]-obs["mast_base"][0])**2 +
                          (obs["mast_top"][1]-obs["mast_base"][1])**2)
        # Rough: dist ≈ f * MAST_H / mast_px (ignoring viewing angle)
        if mast_px > 5:
            dist = f * MAST_H / mast_px
        else:
            dist = 100  # default
        dists.append(dist)
    return dists


def bundle_residuals(params, frame_boats, frame_times, img_shape):
    """
    params: [x0, y0, h0, dx, dy, dh, pitch, yaw_cam, focal,
             boat0_x, boat0_y, boat0_yaw, ...]  per frame per boat

    Camera at time t: pos = (x0 + dx*dt, y0 + dy*dt, h0 + dh*dt)
    """
    Himg, Wimg = img_shape
    cx, cy = Wimg / 2, Himg / 2

    x0, y0, h0 = params[0], params[1], params[2]
    dx, dy, dh = params[3], params[4], params[5]
    pitch = params[6]
    yaw_cam = params[7]
    focal = params[8]

    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

    residual_list = []
    idx = 9  # start of boat params

    for fi, (boats, t) in enumerate(zip(frame_boats, frame_times)):
        dt = t - frame_times[0]

        # Camera position at this time
        cam_x = x0 + dx * dt
        cam_y = y0 + dy * dt
        cam_h = h0 + dh * dt

        # Camera rotation: pitch down by 'pitch', yaw by 'yaw_cam'
        # Build rotation: first yaw around Z, then pitch around X
        Ryaw = np.array([[np.cos(yaw_cam), -np.sin(yaw_cam), 0],
                         [np.sin(yaw_cam),  np.cos(yaw_cam), 0],
                         [0, 0, 1]])
        Rpitch = np.array([[1, 0, 0],
                           [0, np.cos(pitch), -np.sin(pitch)],
                           [0, np.sin(pitch),  np.cos(pitch)]])
        # Camera looks along -Z in camera frame, so we need to flip
        Rflip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # OpenCV convention

        R = Rflip @ Rpitch @ Ryaw
        rvec, _ = cv2.Rodrigues(R)
        tvec = (-R @ np.array([cam_x, cam_y, cam_h])).reshape(3, 1)

        for b in boats:
            L, H = b["L"], b["H"]
            obs = b["obs"]
            boat = boat_model(L, H)

            bx, by, byaw = params[idx], params[idx+1], params[idx+2]
            idx += 3

            Rz = np.array([[np.cos(byaw), -np.sin(byaw), 0],
                           [np.sin(byaw),  np.cos(byaw), 0],
                           [0, 0, 1]])
            pts_world = (Rz @ boat.T).T + np.array([bx, by, 0.0])
            proj = project(pts_world, K, rvec, tvec)
            meas = np.array([obs["bow"], obs["stern"], obs["mast_base"], obs["mast_top"]], dtype=np.float64)
            diff = (proj - meas).ravel()

            # Weight bow/stern by pixel length
            mast_px = np.sqrt((meas[2][0]-meas[3][0])**2 + (meas[2][1]-meas[3][1])**2)
            bs_px = np.sqrt((meas[0][0]-meas[1][0])**2 + (meas[0][1]-meas[1][1])**2)
            ratio = bs_px / mast_px if mast_px > 5 else 0
            bs_w = 0.3 if ratio >= 0.25 else 0.02

            weights = np.array([bs_w, bs_w, bs_w, bs_w, 1.0, 1.0, 1.0, 1.0])
            residual_list.append(diff * weights)

    # Height must be positive
    h_penalty = max(0, -h0) * 100 + max(0, -(h0 + dh * (frame_times[-1] - frame_times[0]))) * 100
    # Height should be reasonable (10-200m)
    h_range_penalty = max(0, 10 - h0) * 10 + max(0, h0 - 200) * 10
    pitch_penalty = 0.0

    residual_list.append(np.array([h_penalty, h_range_penalty, pitch_penalty]))

    return np.concatenate(residual_list)


def solve_bundle(frame_boats, frame_times, img_shape):
    """Solve a bundle of frames."""

    # Initial estimates from mast heights
    all_dists = []
    all_px = []
    for boats in frame_boats:
        dists = estimate_distances(boats)
        for i, b in enumerate(boats):
            all_dists.append(dists[i])
            all_px.append(b["obs"]["mast_base"])

    avg_dist = np.mean(all_dists) if all_dists else 50

    # Init camera: above the scene, looking down
    # Estimate h from average distance: h ≈ dist * sin(pitch)
    # For pitch ≈ 30° (0.5 rad): h ≈ dist * 0.5
    h_init = avg_dist * 0.4

    best = None; best_c = float('inf')
    for trial in range(3):
        np.random.seed(trial * 13)

        # Camera trajectory: x0, y0, h0, dx, dy, dh, pitch, yaw_cam, focal
        p0 = [
            np.random.randn() * 5,     # x0
            np.random.randn() * 5,     # y0
            20.0,                       # h0 fixed at 20m
            np.random.randn() * 0.5,   # dx
            np.random.randn() * 0.5,   # dy
            0.0,                        # dh = 0
            0.349,                      # pitch fixed at 20°
            np.random.randn() * 0.3,   # yaw_cam
            500 + np.random.randn() * 100,  # focal (free)
        ]

        # Per boat per frame
        for fi, boats in enumerate(frame_boats):
            dists = estimate_distances(boats)
            for i, b in enumerate(boats):
                # Init boat position from pixel + estimated distance
                px, py = b["obs"]["mast_base"]
                # Rough: x ≈ (px - 960) * dist / f, y ≈ dist
                bx = (px - 960) * dists[i] / FIXED_F
                by = dists[i]
                p0 += [bx + np.random.randn(), by + np.random.randn(), np.random.randn() * 3]

        p0 = np.array(p0)

        # Bounds: camera params bounded, boat params free
        n_params = len(p0)
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params, np.inf)

        # x0, y0: free
        # h0: fixed at 20m
        lb[2] = 19.9; ub[2] = 20.1
        # dx, dy: reasonable drift (-5 to 5 m/s)
        lb[3] = -5; ub[3] = 5
        lb[4] = -5; ub[4] = 5
        # dh: zero (height doesn't change)
        lb[5] = -0.01; ub[5] = 0.01
        # pitch: fixed at 20° (0.349 rad)
        lb[6] = 0.348; ub[6] = 0.350
        # yaw_cam: -pi to pi
        lb[7] = -np.pi; ub[7] = np.pi
        # focal: 100-5000
        lb[8] = 100; ub[8] = 5000

        # Clamp init to bounds
        p0 = np.clip(p0, lb, ub)

        r = least_squares(bundle_residuals, p0,
                         args=(frame_boats, frame_times, img_shape),
                         bounds=(lb, ub),
                         loss='soft_l1', verbose=0, max_nfev=500)
        if r.cost < best_c:
            best_c = r.cost; best = r

    return best
