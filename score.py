"""Score one submitted ERP against a hidden evaluation set.

Checks / metrics
  size        output must be W x H of the set (2048x1024)
  alignment   rotation that best maps the output onto GT (yaw free, pitch/roll searched)
  tilt        angle between the output's up-axis and true up -> score 0 if >= --max_tilt (5 deg)
  photometric one gain per colour channel fitted to GT (absolute exposure / white balance cannot be
              known from the photos, like yaw)
  seam        left/right edge continuity (automatic hint; official check is visual)
  fidelity    PSNR and LPIPS vs GT on 6 cube faces of the aligned output, nadir cap masked
  score       50*(PSNR-15)/20 + 50*(1-LPIPS/0.5), each part clipped to 0..50;
              0 if the size is wrong or tilt >= 5 deg

Usage: python score.py --set sets/scene_a --pred result.png [--json out.json]
"""
import argparse
import json
import warnings
import numpy as np
import cv2
from common import pose_to_R, rotate_erp, cube_face, face_lat, erp_dirs, load_json

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--pred", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--max_tilt", type=float, default=5.0, help="tilt (deg) at or above which the score is 0")
ap.add_argument("--seam_png", default=None, help="save half-rolled image for the visual seam check")
a = ap.parse_args()

gt = cv2.imread(f"{a.set}/hidden/gt.png", cv2.IMREAD_COLOR)
mask = cv2.imread(f"{a.set}/hidden/mask.png", cv2.IMREAD_GRAYSCALE) > 127
nadir_deg = load_json(f"{a.set}/hidden/true_poses.json")["nadir_mask_deg"]
pred = cv2.imread(a.pred, cv2.IMREAD_COLOR)
H, W = gt.shape[:2]
res = {"pred": a.pred, "set": a.set, "size_ok": pred is not None and pred.shape[:2] == (H, W)}
if pred is None:
    raise SystemExit("cannot read prediction")
if not res["size_ok"]:
    res["size_found"] = list(pred.shape[1::-1])
    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- alignment
def edge_map(img, w):
    g = cv2.cvtColor(cv2.resize(img, (w, w // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0) ; gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx ** 2 + gy ** 2)


def ncc(x, y, m):
    x = x[m]; y = y[m]
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8))


SW = 512
Eg = edge_map(gt, SW)
Ep = edge_map(pred, SW)
m_s = cv2.resize(mask.astype(np.uint8), (SW, SW // 2), interpolation=cv2.INTER_NEAREST) > 0
coslat = np.cos(np.radians(90 - (np.arange(SW // 2) + 0.5) * 180 / (SW // 2)))[:, None]

# 1) yaw by circular cross-correlation (pure column shift in ERP)
A = (Eg - Eg[m_s].mean()) * m_s * coslat
B = (Ep - Ep[m_s].mean()) * m_s * coslat
xc = np.real(np.fft.ifft(np.fft.fft(A, axis=1).conj() * np.fft.fft(B, axis=1), axis=1)).sum(0)
yaw0 = (np.argmax(xc) / SW) * 360.0  # pred shows at yaw+yaw0 what GT shows at yaw


def score_rot(yaw, pitch, roll, w=256):
    R = pose_to_R(yaw, pitch, roll)
    Ep_r = rotate_erp(Ep_small[w], R, cv2.INTER_LINEAR)
    return ncc(Ep_r, Eg_small[w], m_small[w])


Ep_small, Eg_small, m_small = {}, {}, {}
for w_ in (256, 512):
    Ep_small[w_] = edge_map(pred, w_); Eg_small[w_] = edge_map(gt, w_)
    m_small[w_] = cv2.resize(mask.astype(np.uint8), (w_, w_ // 2), interpolation=cv2.INTER_NEAREST) > 0

# 2) coarse pitch/roll (+ small yaw) search, then two refinements
best = (-2, yaw0, 0.0, 0.0)
for dy in np.arange(-4, 5, 2):
    for p in np.arange(-12, 13, 2):
        for r in np.arange(-12, 13, 2):
            s = score_rot(yaw0 + dy, p, r)
            if s > best[0]:
                best = (s, yaw0 + dy, p, r)
for step, span, w_ in ((0.5, 2, 256), (0.125, 0.5, 512)):
    _, y_, p_, r_ = best
    grid = np.arange(-span, span + 1e-9, step)
    for dy in grid:
        for dp in grid:
            for dr in grid:
                s = score_rot(y_ + dy, p_ + dp, r_ + dr, w_)
                if s > best[0]:
                    best = (s, y_ + dy, p_ + dp, r_ + dr)
ncc_best, yaw_c, pitch_c, roll_c = best

# 3) sub-pixel refinement on the images themselves (the edge-map grid above is only ~0.1-0.5 deg
#    accurate, and 0.2 deg of residual rotation already costs a lot of PSNR on fine textures)
from scipy.optimize import minimize


def gray_small(img, w):
    g = cv2.cvtColor(cv2.resize(img, (w, w // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), 0.7)


for w_ in (1024, 2048):
    Gp, Gg = gray_small(pred, w_), gray_small(gt, w_)
    m_ = cv2.resize(mask.astype(np.uint8), (w_, w_ // 2), interpolation=cv2.INTER_NEAREST) > 0
    obj = lambda x: -ncc(rotate_erp(Gp, pose_to_R(*x), cv2.INTER_LINEAR), Gg, m_)
    r = minimize(obj, [yaw_c, pitch_c, roll_c], method="Powell",
                 options={"xtol": 0.005, "ftol": 1e-7, "maxfev": 300})
    if -r.fun > ncc_best or w_ == 2048:
        yaw_c, pitch_c, roll_c = r.x
    ncc_best = -r.fun
R_corr = pose_to_R(yaw_c, pitch_c, roll_c)
up_in_pred = R_corr @ np.array([0, 1.0, 0])
tilt = float(np.degrees(np.arccos(np.clip(up_in_pred[1], -1, 1))))
res.update({"align_yaw_deg": round(float(yaw_c % 360), 3), "align_pitch_deg": round(float(pitch_c), 3),
            "align_roll_deg": round(float(roll_c), 3), "align_ncc": round(ncc_best, 4),
            "tilt_deg": round(tilt, 3), "tilt_ok": tilt < a.max_tilt})
aligned = rotate_erp(pred, R_corr)

# ---------------------------------------------------------------- photometric normalisation
# One gain per colour channel (least squares on the scored area, clipped/black pixels ignored).
af = aligned.astype(np.float64); gf = gt.astype(np.float64)
valid = mask & (af.max(2) > 5) & (af.max(2) < 250) & (gf.max(2) < 250)
gains = []
for c in range(3):
    p_, t_ = af[..., c][valid], gf[..., c][valid]
    gains.append(float(np.clip((p_ * t_).sum() / ((p_ * p_).sum() + 1e-9), 0.5, 2.0)) if p_.size > 1000 else 1.0)
aligned = np.clip(af * np.array(gains)[None, None, :] + 0.5, 0, 255).astype(np.uint8)
res["gain_bgr"] = [round(g_, 4) for g_ in gains]

# ---------------------------------------------------------------- seam (left/right edges)
g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float32)
rows = mask[:, 0]
seam = np.abs(g[rows, 0] - g[rows, -1]).mean()
inner = np.median([np.abs(g[rows, c] - g[rows, c + 1]).mean() for c in range(0, W - 1, 16)])
res["seam_ratio"] = round(float(seam / (inner + 1e-6)), 3)
res["seam_hint_ok"] = res["seam_ratio"] < 2.0
if a.seam_png:
    cv2.imwrite(a.seam_png, np.roll(pred, W // 2, axis=1))

# ---------------------------------------------------------------- fidelity (full reference)
# PSNR and LPIPS on 6 cube faces of the aligned output; nadir cap masked out.
# SSIM is reported for reference only (it is too lenient on blurry fills of plain walls).
from skimage.metrics import structural_similarity
import torch
import lpips
torch.set_grad_enabled(False)
lsp = lpips.LPIPS(net="alex", spatial=True, verbose=False)
to_t = lambda x: torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
F = W // 4
se, n_px, lp_sum, ss_sum, faces = 0.0, 0, 0.0, 0.0, {}
for name in ["front", "right", "back", "left", "top", "bottom"]:
    fp = cube_face(aligned, name, F); fg = cube_face(gt, name, F)
    fm = face_lat(name, F) > nadir_deg
    if fm.sum() < 100:
        continue
    d = (fp.astype(np.float64) - fg.astype(np.float64))[fm]
    f_se = float((d ** 2).sum()); f_n = int(fm.sum())
    lm = lsp(to_t(fp), to_t(fg))[0, 0].numpy()
    _, smap = structural_similarity(cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY),
                                    data_range=255, full=True, gaussian_weights=True)
    se += f_se; n_px += f_n; lp_sum += float(lm[fm].sum()); ss_sum += float(smap[fm].sum())
    faces[name] = {"psnr": round(10 * np.log10(255 ** 2 / (f_se / (3 * f_n) + 1e-10)), 2),
                   "lpips": round(float(lm[fm].mean()), 4)}
psnr = float(10 * np.log10(255 ** 2 / (se / (3 * n_px) + 1e-10)))
lp = lp_sum / n_px
res.update({"psnr": round(psnr, 3), "lpips": round(lp, 4), "ssim_ref": round(ss_sum / n_px, 4), "faces": faces})
res["passed"] = bool(res["size_ok"] and res["tilt_ok"])
# Quality score (0-100): 50 x PSNR part (15 dB -> 0, 35 dB -> 50) + 50 x LPIPS part (0.5 -> 0, 0 -> 50).
s_psnr = float(np.clip(50 * (psnr - 15) / 20, 0, 50))
s_lpips = float(np.clip(50 * (1 - lp / 0.5), 0, 50))
res.update({"score_psnr": round(s_psnr, 2), "score_lpips": round(s_lpips, 2),
            "score": round(s_psnr + s_lpips, 2) if res["passed"] else 0.0})
print(json.dumps(res, indent=2, ensure_ascii=False))
if a.json:
    with open(a.json, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
