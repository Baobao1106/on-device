"""Build one evaluation set from a ground-truth ERP panorama.

out/
  public/images/<random>.jpg     24 phone-like pinhole shots (864x648), random names/order
  hidden/gt.png                  ground-truth ERP, 2048x1024 (gravity-aligned; yaw is free in scoring)
  hidden/mask.png                255 = scored (lat > nadir_mask_deg)
  hidden/true_poses.json         actual jittered poses + photometric params

Simulated capture: true hfov random 72-77 deg per set (hidden); 8 shots at pitch 0, 7 at +45, 7 at -45, zenith, nadir; hand jitter +-5 deg
(yaw/pitch/roll), vignetting, per-shot gain 0.85-1.15, white balance +-4%, noise, JPEG q90.
The hidden evaluation sets are produced with exactly this script and its default parameters.

Usage: python make_set.py --erp pano.jpg --out sets/scene_a --seed 1
"""
import argparse
import os
import numpy as np
import cv2
from common import capture_plan, pose_to_R, render_pinhole, rotate_erp, erp_dirs, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--erp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--w", type=int, default=864)
ap.add_argument("--h", type=int, default=648)
ap.add_argument("--hfov", type=float, default=None, help="true horizontal FOV (default: random 72-77 deg, not given to teams)")
ap.add_argument("--jitter", type=float, default=5.0, help="max |deg| hand jitter (yaw/pitch/roll)")
ap.add_argument("--ref_jitter", type=float, default=2.0, help="jitter for shot 01 (careful level shot)")
ap.add_argument("--gt_w", type=int, default=2048)
ap.add_argument("--nadir_mask_deg", type=float, default=-60.0)
ap.add_argument("--src_yaw", type=float, default=None, help="yaw of the source ERP that becomes 'forward' (random if omitted)")
ap.add_argument("--jpeg_q", type=int, default=90)
ap.add_argument("--keep_order", action="store_true", help="name files 01..24 in capture order (default: shuffled random names, image-only task)")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
src = cv2.imread(a.erp, cv2.IMREAD_COLOR)
assert src is not None, a.erp
src = src.astype(np.float32) / 255.0

# Choose which direction of the source panorama is "forward" for this set.
src_yaw = rng.uniform(0, 360) if a.src_yaw is None else a.src_yaw
hfov = rng.uniform(72.0, 77.0) if a.hfov is None else a.hfov
R_frame = pose_to_R(src_yaw, 0)          # nominal frame -> source frame

os.makedirs(f"{a.out}/public/images", exist_ok=True)
os.makedirs(f"{a.out}/hidden", exist_ok=True)

# Ground truth in the nominal frame, at evaluation resolution.
gt = rotate_erp(src, R_frame)
gt = cv2.resize(gt, (a.gt_w, a.gt_w // 2), interpolation=cv2.INTER_AREA)
cv2.imwrite(f"{a.out}/hidden/gt.png", np.clip(gt * 255 + 0.5, 0, 255).astype(np.uint8))
lat = np.degrees(np.arcsin(erp_dirs(a.gt_w, a.gt_w // 2)[..., 1]))
cv2.imwrite(f"{a.out}/hidden/mask.png", ((lat > a.nadir_mask_deg) * 255).astype(np.uint8))

# Vignetting map (shared lens model).
xs, ys = np.meshgrid(np.linspace(-1, 1, a.w), np.linspace(-1, 1, a.h) * a.h / a.w)
vig = 1.0 - 0.18 * (xs ** 2 + ys ** 2)

plan = capture_plan()
names = [f"{x:04x}" for x in rng.choice(65536, len(plan), replace=False)]  # random, order-free file names
nominal, truth, encoded = [], [], {}
for i, (yaw, pitch) in enumerate(plan):
    j = a.ref_jitter if i == 0 else a.jitter
    dy, dp, dr = rng.uniform(-j, j, 3)
    t_yaw, t_pitch, t_roll = yaw + dy, np.clip(pitch + dp, -90, 90), dr
    if abs(pitch) == 90:  # at the poles yaw is just a rotation about the axis: make it random
        t_yaw = rng.uniform(0, 360)
    R = R_frame @ pose_to_R(t_yaw, t_pitch, t_roll)
    img = render_pinhole(src, R, a.w, a.h, hfov, cv2.INTER_CUBIC)

    # Phone-like photometric variation.
    gain = rng.uniform(0.85, 1.15)
    wb = rng.uniform(0.96, 1.04, 3)
    img = img * vig[..., None] * gain * wb[None, None, :]
    img = img + rng.normal(0, 2.0 / 255, img.shape)
    img = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    name = f"{i + 1:02d}.jpg" if a.keep_order else f"{names[i]}.jpg"
    encoded[name] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_q])[1].tobytes()

    nominal.append({"image": name, "yaw_deg": round(yaw, 3), "pitch_deg": pitch})
    truth.append({"image": name, "yaw_deg": float(t_yaw), "pitch_deg": float(t_pitch), "roll_deg": float(dr),
                  "gain": float(gain), "wb": wb.tolist()})

if a.keep_order:
  save_json({
    "note": "Nominal capture-plan angles. Shot 01 is level and facing forward = ERP centre. "
            "Actual hand-held angles differ by a few degrees.",
    "image_size": [a.w, a.h],
    "output_size": [a.gt_w, a.gt_w // 2],
    "shots": nominal}, f"{a.out}/public/poses_nominal.json")

# Write the photos in file-name order and give them all the same timestamp, so that neither
# modification times nor creation/listing order leak the capture order.
FIXED_TIME = 1767225600  # 2026-01-01 00:00:00 UTC
for name in sorted(encoded):
    path = f"{a.out}/public/images/{name}"
    with open(path, "wb") as f:
        f.write(encoded[name])
    os.utime(path, (FIXED_TIME, FIXED_TIME))

save_json({"source": os.path.abspath(a.erp), "src_yaw_deg": src_yaw, "hfov_deg": hfov,
           "jitter_deg": a.jitter, "ref_jitter_deg": a.ref_jitter, "seed": a.seed,
           "nadir_mask_deg": a.nadir_mask_deg, "shots": truth}, f"{a.out}/hidden/true_poses.json")
print(f"wrote {a.out}: {len(plan)} shots {a.w}x{a.h}, gt {a.gt_w}x{a.gt_w // 2}, src_yaw {src_yaw:.1f}")
