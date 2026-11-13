"""Rough ring classification for 24 unordered pinhole images.

Phân loại thô 24 bức ảnh ngẫu nhiên thành 5 nhóm dựa trên đặc trưng thị giác
và tính chất hình học không gian (Pose Topology):
  - Nhóm 1: Trần nhà (Zenith, 1 ảnh, Pitch ≈ +90°)
  - Nhóm 2: Sàn nhà (Nadir, 1 ảnh, Pitch ≈ -90°)
  - Nhóm 3: Vành trên (Up ring, 7 ảnh, Pitch ≈ +45°)
  - Nhóm 4: Vành ngang (Horizon ring, 8 ảnh, Pitch ≈ 0°)
  - Nhóm 5: Vành dưới (Down ring, 7 ảnh, Pitch ≈ -45°)
"""

import os
import argparse
import json
import cv2
import numpy as np


def extract_visual_features(img_path):
    """Trích xuất các đặc trưng thị giác thô từ một bức ảnh.
    
    Returns:
        dict chứa các chỉ số:
          - brightness: độ sáng trung bình (0-255)
          - lap_var: phương sai Laplacian (độ nét, vân bề mặt / texture)
          - v_lines: số lượng đường thẳng đứng phát hiện được (HoughLinesP)
          - tilted_lines: số lượng đường xiên góc (tilted lines)
          - v_bias: độ chênh lệch ánh sáng giữa 1/3 trên và 1/3 dưới
    """
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")
        
    h, w = gray.shape
    
    # 1. Độ sáng trung bình
    brightness = float(np.mean(gray))
    
    # 2. Độ chi tiết / vân bề mặt (Laplacian Variance)
    # Ảnh trơn (như trần nhà) sẽ có giá trị này rất thấp
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 3. Phát hiện đường thẳng (Canny + HoughLinesP)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=10)
    
    v_lines = 0
    tilted_lines = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Đường thẳng đứng: góc từ 80 đến 100 độ
            if 80 <= ang <= 100:
                v_lines += 1
            # Đường xiên: góc từ 25-65 độ hoặc 115-155 độ
            elif 25 <= ang <= 65 or 115 <= ang <= 155:
                tilted_lines += 1
                
    # 4. Gradient độ sáng dọc (Phần trên sáng hơn phần dưới bao nhiêu)
    top_br = float(np.mean(gray[:h // 3, :]))
    bot_br = float(np.mean(gray[2 * h // 3:, :]))
    v_bias = top_br - bot_br
    
    return {
        "brightness": brightness,
        "lap_var": lap_var,
        "v_lines": v_lines,
        "tilted_lines": tilted_lines,
        "v_bias": v_bias
    }


def classify_rough_rings(images_dir):
    """Phân loại thô 24 ảnh trong thư mục images_dir vào 5 nhóm.
    
    Args:
        images_dir (str): Đường dẫn đến thư mục chứa 24 ảnh jpg.
        
    Returns:
        dict: {
            "zenith": "filename_zenith.jpg",       # 1 ảnh trần (~ +90°)
            "nadir": "filename_nadir.jpg",         # 1 ảnh sàn (~ -90°)
            "horizon": ["img1.jpg", ...],          # 8 ảnh vành ngang (~ 0°)
            "up": ["img1.jpg", ...],               # 7 ảnh vành trên (~ +45°)
            "down": ["img1.jpg", ...]              # 7 ảnh vành dưới (~ -45°)
        }
    """
    valid_exts = (".jpg", ".jpeg", ".png")
    all_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)])
    
    if len(all_files) != 24:
        print(f"Cảnh báo: Thư mục có {len(all_files)} ảnh (chuẩn capture_plan là 24 ảnh).")

    # Bước 1: Trích xuất đặc trưng cho tất cả các ảnh
    features = {}
    for f in all_files:
        path = os.path.join(images_dir, f)
        features[f] = extract_visual_features(path)
        
    remaining = set(all_files)
    
    # Bước 2: Tìm ảnh Trần (Zenith - 1 ảnh)
    # Đặc điểm: Trần thường sáng nhất, ít chi tiết nhất (lap_var thấp), ít đường đứng
    zenith_scores = {
        f: features[f]["brightness"] / (np.sqrt(features[f]["lap_var"]) + 5.0) - features[f]["v_lines"] * 2.0
        for f in remaining
    }
    zenith = max(zenith_scores, key=zenith_scores.get)
    remaining.remove(zenith)
    
    # Bước 3: Tìm ảnh Sàn (Nadir - 1 ảnh)
    # Đặc điểm: Chụp thẳng xuống đất, hầu như không có đường thẳng đứng (v_lines rất thấp),
    # nhưng có vân gạch/thảm (lap_var cao), độ sáng trung bình/thấp hơn trần
    nadir_scores = {
        f: np.log(features[f]["lap_var"] + 1.0) * 10.0 - features[f]["v_lines"] * 5.0 - features[f]["brightness"] * 0.2
        for f in remaining
    }
    nadir = max(nadir_scores, key=nadir_scores.get)
    remaining.remove(nadir)
    
    # Bước 4: Tìm 8 ảnh Vành ngang (Horizon - 8 ảnh)
    # Đặc điểm: Camera nằm ngang (pitch ≈ 0°), các góc tường, cạnh cửa, chân tủ
    # bảo toàn tính chất thẳng đứng -> số lượng v_lines cao nhất và tỉ lệ v_lines / tilted_lines lớn nhất
    horizon_scores = {
        f: features[f]["v_lines"] * 2.0 + (features[f]["v_lines"] / (features[f]["tilted_lines"] + 1.0)) * 5.0
        for f in remaining
    }
    sorted_horizon_candidates = sorted(remaining, key=lambda f: horizon_scores[f], reverse=True)
    horizon = set(sorted_horizon_candidates[:8])
    remaining -= horizon
    
    # Bước 5: Phân chia 14 ảnh còn lại thành Vành trên (7 ảnh) và Vành dưới (7 ảnh)
    # Đặc điểm: Vành trên nhìn hướng lên trần/nguồn sáng nên tổng thể sáng hơn
    # và có gradient ánh sáng phía trên lớn hơn (v_bias cao hơn).
    # Vành dưới nhìn hướng xuống sàn nên tối hơn và có nhiều đường xiên (tilted_lines) do phối cảnh sàn.
    up_down_scores = {
        f: features[f]["brightness"] + features[f]["v_bias"] * 1.5 - features[f]["tilted_lines"] * 0.5
        for f in remaining
    }
    sorted_up_down = sorted(remaining, key=lambda f: up_down_scores[f], reverse=True)
    up = set(sorted_up_down[:7])
    down = set(sorted_up_down[7:])
    
    return {
        "zenith": zenith,
        "nadir": nadir,
        "horizon": sorted(list(horizon)),
        "up": sorted(list(up)),
        "down": sorted(list(down))
    }


def evaluate_against_ground_truth(set_dir):
    """Hàm phụ trợ giúp đánh giá độ chính xác khi có thư mục hidden/true_poses.json"""
    poses_path = os.path.join(set_dir, "hidden", "true_poses.json")
    if not os.path.exists(poses_path):
        print(f"Không tìm thấy {poses_path} để kiểm tra ground truth.")
        return
        
    poses = json.load(open(poses_path))
    gt_groups = {}
    for s in poses["shots"]:
        p = round(s["pitch_deg"])
        if p > 70:
            grp = "zenith"
        elif p > 20:
            grp = "up"
        elif p > -20:
            grp = "horizon"
        elif p > -70:
            grp = "down"
        else:
            grp = "nadir"
        gt_groups[s["image"]] = grp
        
    images_dir = os.path.join(set_dir, "public", "images")
    res = classify_rough_rings(images_dir)
    
    pred_map = {res["zenith"]: "zenith", res["nadir"]: "nadir"}
    for f in res["horizon"]:
        pred_map[f] = "horizon"
    for f in res["up"]:
        pred_map[f] = "up"
    for f in res["down"]:
        pred_map[f] = "down"
        
    correct = 0
    print(f"\n{'Ảnh':<12} | {'Nhóm đoán':<10} | {'Ground Truth':<12} | {'Kết quả'}")
    print("-" * 50)
    for f in sorted(gt_groups.keys()):
        p_grp = pred_map.get(f, "unknown")
        g_grp = gt_groups[f]
        is_ok = (p_grp == g_grp)
        if is_ok:
            correct += 1
        mark = "✓ Đúng" if is_ok else "✗ Sai"
        print(f"{f:<12} | {p_grp:<10} | {g_grp:<12} | {mark}")
        
    print("-" * 50)
    print(f"Tổng kết: {correct}/24 ảnh chính xác ({correct / 24 * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân loại thô 24 ảnh thành 5 nhóm theo pitch")
    parser.add_argument("--images", default="sets/t01/public/images", help="Đường dẫn thư mục ảnh")
    parser.add_argument("--eval_set", default=None, help="Đường dẫn set để đánh giá (ví dụ sets/t01)")
    args = parser.parse_args()
    
    if args.eval_set:
        evaluate_against_ground_truth(args.eval_set)
    else:
        results = classify_rough_rings(args.images)
        print("Kết quả phân loại 5 nhóm:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
