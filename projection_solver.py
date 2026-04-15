import numpy as np
from scipy.optimize import least_squares
import cv2


# -----------------------------
# 3D boat model in local coords
# -----------------------------
def boat_model(L, H):
    # bow, stern, mast base, mast top
    return np.array([
        [-L/3, 0.0, 0.0],
        [ 2*L/3, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, H],
    ], dtype=np.float64)


# -----------------------------
# projection helper
# -----------------------------
def project(points_3d, K, rvec, tvec):
    pts, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    return pts.reshape(-1, 2)


# -----------------------------
# residual function
# -----------------------------
def residuals(params, boats, img_shape):
    Himg, Wimg = img_shape
    cx, cy = Wimg / 2, Himg / 2

    idx = 0

    # ---- camera parameters ----
    f = params[idx]; idx += 1
    rvec = params[idx:idx+3].reshape(3, 1); idx += 3
    tvec = params[idx:idx+3].reshape(3, 1); idx += 3

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    residual_list = []

    # ---- per boat ----
    for b in boats:
        L = b["L"]
        H = b["H"]
        obs = b["obs"]

        # local boat geometry
        boat = boat_model(L, H)

        # boat pose (unknown)
        x, y, yaw = params[idx:idx+3]
        idx += 3

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        t_boat = np.array([x, y, 0.0])

        # transform to world
        pts_world = (Rz @ boat.T).T + t_boat

        # project to image
        proj = project(pts_world, K, rvec, tvec)

        meas = np.array([
            obs["bow"],
            obs["stern"],
            obs["mast_base"],
            obs["mast_top"]
        ], dtype=np.float64)

        diff = (proj - meas).ravel()

        # Geometric validation: check if detected bow-stern makes sense
        # Expected hull pixels ≈ mast pixels * (L/H)
        mast_px = np.sqrt((meas[2][0]-meas[3][0])**2 + (meas[2][1]-meas[3][1])**2)
        bs_px = np.sqrt((meas[0][0]-meas[1][0])**2 + (meas[0][1]-meas[1][1])**2)
        expected_bs = mast_px * (L / H)  # rough expected hull length in pixels

        # If detected bow-stern is way off from mast height, downweight it
        if mast_px > 5:
            bs_mast_ratio = bs_px / mast_px
            # Hull/mast real ratio = 4.2/5.89 = 0.71
            # Beam-on: bs_px ≈ 0.71 * mast_px
            # Head-on: bs_px can be small (0.1-0.3 * mast_px)
            # Below 0.25 of mast height: something is wrong
            if bs_mast_ratio < 0.25:
                bs_weight = 0.05  # almost ignore — likely bad detection
            elif bs_mast_ratio > 2.0:
                bs_weight = 0.05  # way too wide — crossed to another boat
            else:
                bs_weight = 0.3
        else:
            bs_weight = 0.1  # tiny boat, don't trust bow-stern

        # Weight: bow(0,1), stern(2,3), mast_base(4,5), mast_top(6,7)
        weights = np.array([bs_weight, bs_weight, bs_weight, bs_weight, 1.0, 1.0, 1.0, 1.0])
        residual_list.append(diff * weights)

    return np.concatenate(residual_list)


def residuals_constrained(params, boats, img_shape, fixed_height=20.0):
    """Residuals with height pinned and tilt forced positive (looking down)."""
    # Hard clamp rvec[0] to ensure looking down (positive tilt)
    # rvec[0] > 0 generally means looking down for our init convention
    params = params.copy()
    if params[1] < 0.3:
        params[1] = 0.3  # hard floor on pitch

    base = residuals(params, boats, img_shape)

    # Height constraint: strong pin
    height_penalty = (params[6] - fixed_height) * 10.0

    # Focal length constraint: pin to fixed value
    focal_penalty = (params[0] - 500.0) * 5.0

    # Tilt constraint
    rvec = params[1:4]
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    cam_z = R[:, 2]
    tilt = np.arcsin(-cam_z[2])
    tilt_penalty = max(0, -tilt) * 1000.0  # very strongly penalize looking up

    # Depth ordering: boats higher in image (smaller pixel Y) should be further from camera
    # Distance from camera = sqrt((x - cam_x)^2 + (y - cam_y)^2)
    tvec_vec = params[4:7]
    cam_pos = -R.T @ tvec_vec  # camera position in world
    idx = 7
    boat_data = []
    for b in boats:
        pixel_y = b["obs"]["mast_base"][1]
        x, y, yaw = params[idx:idx+3]
        idx += 3
        dist = np.sqrt((x - cam_pos[0])**2 + (y - cam_pos[1])**2)
        boat_data.append((pixel_y, dist))

    ordering_penalty = 0.0
    if len(boat_data) >= 2:
        boat_data.sort(key=lambda bd: bd[0])  # sort by pixel Y ascending (top first)
        for i in range(len(boat_data) - 1):
            py1, d1 = boat_data[i]      # higher in image = should be further
            py2, d2 = boat_data[i + 1]  # lower in image = should be closer
            if d1 < d2:  # violation: further pixel is closer in world
                ordering_penalty += (d2 - d1) * 5.0

    return np.concatenate([base, [height_penalty, focal_penalty, tilt_penalty, ordering_penalty]])


# -----------------------------
# solver wrapper
# -----------------------------
def solve_camera_and_scene(boats, img_shape):

    n_boats = len(boats)

    # ---- initial guesses ----
    f0 = img_shape[1] * 1.2

    rvec0 = np.array([0.0, 0.0, 0.0])
    tvec0 = np.array([0.0, 0.0, 50.0])

    params0 = [f0] + list(rvec0) + list(tvec0)

    # boat initial guesses
    for i in range(n_boats):
        params0 += [i * 2.0, 0.0, 0.0]  # x, y, yaw

    params0 = np.array(params0, dtype=np.float64)

    result = least_squares(
        residuals,
        params0,
        args=(boats, img_shape),
        loss='soft_l1',
        verbose=2,
        max_nfev=200
    )

    return result