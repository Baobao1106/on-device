# Jetson ERP: 360° Panorama Stitching

> **Challenge:** From Perspective Images to 360° Panorama  
> **Target Device:** NVIDIA Jetson Orin Nano Super 8GB (MAXN Mode)  
> **Input:** 24 unordered perspective photos (`864x648` JPG)  
> **Output:** 1 Equirectangular Projection (ERP) panorama (`2048x1024` PNG)

---

## 📌 Project Overview

This repository contains the codebase and pipeline for reconstructing a full $360^\circ$ Equirectangular Projection (ERP) panorama from 24 handheld pinhole perspective photos.

### Key Constraints & Rules:
- **FOV:** The horizontal field of view ($hfov$) is identical for all 24 photos in a set, randomly chosen between $72^\circ$ and $77^\circ$.
- **Capture Pattern (24 shots):**
  - Horizon ($0^\circ$ pitch): 8 shots, spaced $45^\circ$ apart
  - Up ($+45^\circ$ pitch): 7 shots, spaced $\approx 51.4^\circ$ apart
  - Down ($-45^\circ$ pitch): 7 shots, spaced $\approx 51.4^\circ$ apart
  - Zenith ($+90^\circ$ pitch): 1 shot
  - Nadir ($-90^\circ$ pitch): 1 shot
- **Perturbations:** Hand-held jitter $\pm 5^\circ$ (yaw/pitch/roll), vignetting, exposure variation $\pm 15\%$, white balance shift $\pm 4\%$, Gaussian sensor noise, JPEG compression.
- **Evaluation Criteria:**
  - Size check: exactly $2048 \times 1024$ PNG.
  - Tilt threshold: up-axis deviation must be $< 5^\circ$ (otherwise score is 0).
  - Nadir exclusion: floor area below $-60^\circ$ is excluded from scoring.
  - Quality score (0–100): $50 \times \frac{\text{PSNR} - 15}{20} + 50 \times \left(1 - \frac{\text{LPIPS}}{0.5}\right)$.
  - Time limit: 300 seconds (5 minutes) per set on Jetson Orin Nano.

---

## 🚀 Getting Started

### 1. Installation

Create a virtual environment and install dependencies:

```bash
# Optional: create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. File Structure

```text
├── common.py                # Coordinate math, spherical sampling, ERP transformations
├── make_set.py              # Synthetic dataset generator from ground-truth ERP
├── score.py                 # Ground-truth alignment and quality scoring (PSNR, LPIPS, Tilt)
├── run_eval.py              # Batch benchmark & timing harness for Jetson
├── rough_classifier.py      # Rule-based topology classifier (Zenith, Nadir, Horizon, Up, Down)
├── requirements.txt         # Dependencies list
├── sets/                    # Training and validation sets (t01 - t10)
└── panos/                   # High-resolution source panoramas
```

### 3. Running Scoring & Evaluation

To evaluate a predicted panorama against ground truth:

```bash
python score.py --set sets/t01 --pred path/to/result.png
```

To run a method across all available sets with a 300-second timeout:

```bash
python run_eval.py --sets sets --out runs/my_run --cmd "python my_method.py --images {images} --out {out}" --score
```

---

## 👥 Team & Responsibilities

- **Thành viên 1:** Pipeline Architecture & Pose Topology Classification
- **Thành viên 2:** Feature Matching (SIFT / LightGlue) & Bundle Adjustment
- **Thành viên 3:** Gravity Alignment (Leveling, Tilt < 5°), Exposure Compensation & Multi-band Blending
- **Thành viên 4:** On-device Optimization (Jetson Orin Nano), Benchmark & Data Augmentation
