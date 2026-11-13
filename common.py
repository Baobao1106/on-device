"""Shared geometry for the ERP challenge.

Conventions
-----------
World:  x = right, y = up, z = forward.  ERP centre column = forward (yaw 0).
ERP:    lon = atan2(x, z) in [-pi, pi), lat = asin(y).
        u = (lon / 2pi + 0.5) * W,  v = (0.5 - lat / pi) * H
Camera: x = right, y = up, z = optical axis (pixel v grows downward).
Pose:   R_cam2world = Ry(yaw) @ Rx(pitch) @ Rz(roll)
        yaw   > 0 turns right, pitch > 0 looks up, roll > 0 rotates image CCW.
"""
import json
import numpy as np
import cv2

# 24-shot capture plan: (yaw_deg, pitch_deg)
def capture_plan():
    plan = [(45.0 * k, 0.0) for k in range(8)]                      # 01-08 horizon
    plan += [(360.0 / 7 * k, 45.0) for k in range(7)]               # 09-15 up
    plan += [(360.0 / 14 + 360.0 / 7 * k, -45.0) for k in range(7)] # 16-22 down
    plan += [(0.0, 90.0), (0.0, -90.0)]                             # 23 zenith, 24 nadir
    return plan


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose_to_R(yaw_deg, pitch_deg, roll_deg=0.0):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    return rot_y(y) @ rot_x(p) @ rot_z(r)


def intrinsics(w, h, hfov_deg):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return f, (w - 1) / 2, (h - 1) / 2


def erp_dirs(W, H):
    """Unit world direction for every ERP pixel centre, shape (H, W, 3)."""
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    lon = (u - 0.5) * 2 * np.pi
    lat = (0.5 - v) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


def dirs_to_erp_uv(d, W, H):
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = (lon / (2 * np.pi) + 0.5) * W - 0.5
    v = (0.5 - lat / np.pi) * H - 0.5
    return u.astype(np.float32), v.astype(np.float32)


def sample_erp(erp, d, interp=cv2.INTER_CUBIC):
    """Sample an ERP image along world directions d (..., 3) with horizontal wrap."""
    H, W = erp.shape[:2]
    pad = 8
    ext = cv2.copyMakeBorder(erp, 0, 0, pad, pad, cv2.BORDER_WRAP)
    u, v = dirs_to_erp_uv(d, W, H)
    return cv2.remap(ext, u + pad, v, interp, borderMode=cv2.BORDER_REFLECT)


def render_pinhole(erp, R, w, h, hfov_deg, interp=cv2.INTER_CUBIC):
    f, cx, cy = intrinsics(w, h, hfov_deg)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    dc = np.stack([(xs - cx) / f, -(ys - cy) / f, np.ones_like(xs, float)], -1)
    dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
    return sample_erp(erp, dc @ R.T, interp)


def rotate_erp(erp, R, interp=cv2.INTER_CUBIC):
    """Return ERP whose direction d shows what `erp` showed at R @ d."""
    H, W = erp.shape[:2]
    return sample_erp(erp, erp_dirs(W, H) @ R.T, interp)


CUBE_FACES = {  # name: R (face camera -> world)
    "front": pose_to_R(0, 0), "right": pose_to_R(90, 0), "back": pose_to_R(180, 0),
    "left": pose_to_R(-90, 0), "top": pose_to_R(0, 90), "bottom": pose_to_R(0, -90),
}


def cube_face(erp, name, size, interp=cv2.INTER_CUBIC):
    return render_pinhole(erp, CUBE_FACES[name], size, size, 90.0, interp)


def face_lat(name, size):
    """Latitude (deg) of every pixel of a cube face, for masking."""
    f, cx, cy = intrinsics(size, size, 90.0)
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    dc = np.stack([(xs - cx) / f, -(ys - cy) / f, np.ones_like(xs, float)], -1)
    d = dc @ CUBE_FACES[name].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(d[..., 1]))


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
