import cv2
import numpy as np
import os
from harris import HarrisCornerDetector
from descriptor import get_descriptors
from matcher import match_features
from ransac import ransac_homography


def draw_matches(img1, kp1, img2, kp2, matches, inliers=None, output_path=None):
    # 绘制匹配线图
    # img1, img2: 两幅输入图
    # kp1, kp2: 关键点
    # matches: 所有匹配对
    # inliers: 内点对列表
    # output_path: 输出图片路径
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:w1+w2] = img2

    inlier_set = set(inliers) if inliers is not None else None

    for i, j in matches:
        x1, y1 = kp1[i]
        x2, y2 = kp2[j]
        x2_shifted = x2 + w1

        # 内点绿色，外点红色
        color = (0, 255, 0) if (inlier_set is None or (i, j) in inlier_set) else (0, 0, 255)

        cv2.circle(out_img, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(out_img, (int(x2_shifted), int(y2)), 3, color, -1)
        cv2.line(out_img, (int(x1), int(y1)), (int(x2_shifted), int(y2)), color, 1)

    if output_path is not None:
        cv2.imwrite(output_path, out_img)
    return out_img


def draw_inliers_only(img1, kp1, img2, kp2, inliers, output_path=None):
    # 只绘制内点匹配线
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:w1+w2] = img2

    for i, j in inliers:
        x1, y1 = kp1[i]
        x2, y2 = kp2[j]
        x2_shifted = x2 + w1

        color = (0, 255, 0)

        cv2.circle(out_img, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(out_img, (int(x2_shifted), int(y2)), 3, color, -1)
        cv2.line(out_img, (int(x1), int(y1)), (int(x2_shifted), int(y2)), color, 1)

    if output_path is not None:
        cv2.imwrite(output_path, out_img)
    return out_img


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "inputs")  # 输入图文件夹
    output_dir = os.path.join(current_dir, "outputs")  # 输出结果文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 定义10个场景
    scenes = [
        {"name": "Translation", "prefix": "scenario_1_translation",
         "img1": "scenario_1_translation_inputA.jpg", "img2": "scenario_1_translation_inputB.jpg"},
        {"name": "Scale", "prefix": "scenario_2_scale_zoom",
         "img1": "scenario_2_scale_zoom_inputA.jpg", "img2": "scenario_2_scale_zoom_inputB.jpg"},
        {"name": "Rotation", "prefix": "scenario_3_rotation",
         "img1": "scenario_3_rotation_inputA.jpg", "img2": "scenario_3_rotation_inputB.jpg"},
        {"name": "Perspective", "prefix": "scenario_4_perspective_viewpoint",
         "img1": "scenario_4_perspective_viewpoint_inputA.jpg", "img2": "scenario_4_perspective_viewpoint_inputB.jpg"},
        {"name": "Illumination", "prefix": "scenario_5_illumination_variation",
         "img1": "scenario_5_illumination_variation_inputA.jpg", "img2": "scenario_5_illumination_variation_inputB.jpg"},
        {"name": "Blur", "prefix": "scenario_6_focus_blur",
         "img1": "scenario_6_focus_blur_inputA.jpg", "img2": "scenario_6_focus_blur_inputB.jpg"},
        {"name": "Repetitive Patterns", "prefix": "scenario_7_repetitive_patterns",
         "img1": "scenario_7_repetitive_patterns_inputA.jpg", "img2": "scenario_7_repetitive_patterns_inputB.jpg"},
        {"name": "Low Texture", "prefix": "scenario_8_low_texture",
         "img1": "scenario_8_low_texture_inputA.jpg", "img2": "scenario_8_low_texture_inputB.jpg"},
        {"name": "Heavy Clutter", "prefix": "scenario_9_heavy_clutter",
         "img1": "scenario_9_heavy_clutter_inputA.jpg", "img2": "scenario_9_heavy_clutter_inputB.jpg"},
        {"name": "Partial Occlusion", "prefix": "scenario_10_partial_occlusion",
         "img1": "scenario_10_partial_occlusion_inputA.jpg", "img2": "scenario_10_partial_occlusion_inputB.jpg"},
    ]

    # 初始化角点检测器
    detector = HarrisCornerDetector(
        k=0.04, window_size=5, sigma=1.0,
        threshold_ratio=0.015, nms_size=3,
        num_octaves=2, num_scales=3, scale_factor=1.3
    )

    results = []

    # 处理每个场景
    for scene in scenes:
        print(f"\n{scene['name']}")

        img1_path = os.path.join(input_dir, scene["img1"])
        img2_path = os.path.join(input_dir, scene["img2"])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"  无法读取图片")
            continue

        # 1. 检测角点
        print("  检测角点...")
        kp1 = detector.detect(img1)
        kp2 = detector.detect(img2)
        print(f"  img1: {len(kp1)}个, img2: {len(kp2)}个")

        # 2. 提取描述符
        print("  提取描述符...")
        desc1, used_kp1 = get_descriptors(img1, kp1)
        desc2, used_kp2 = get_descriptors(img2, kp2)
        print(f"  img1: {len(desc1)}个, img2: {len(desc2)}个")

        # 3. 特征匹配
        print("  匹配特征...")
        matches = match_features(desc1, desc2, ratio_thresh=0.75)
        print(f"  找到{len(matches)}对匹配")

        # 4. RANSAC估计单应矩阵并剔除外点
        print("  RANSAC筛选...")
        H_est, inliers = ransac_homography(used_kp1, used_kp2, matches, threshold=20.0, max_iter=8000)
        print(f"  保留{len(inliers)}个内点 ({len(inliers)/len(matches)*100:.1f}%)")

        results.append({
            "name": scene["name"],
            "num_matches": len(matches),
            "num_inliers": len(inliers),
            "inlier_ratio": len(inliers)/len(matches)*100 if len(matches) > 0 else 0
        })

        # 5. 保存结果图
        print("  保存结果...")
        output_full = os.path.join(output_dir, f"{scene['prefix']}_result_full.jpg")
        output_inliers = os.path.join(output_dir, f"{scene['prefix']}_result_inliers.jpg")

        draw_matches(img1, used_kp1, img2, used_kp2, matches, inliers, output_full)
        draw_inliers_only(img1, used_kp1, img2, used_kp2, inliers, output_inliers)

        print(f"  保存到{output_full}")

    # 输出结果汇总
    print(f"\n{'='*50}")
    print("  结果汇总")
    print(f"{'='*50}")
    print(f"{'场景':<20} {'匹配':<8} {'内点':<8} {'内点率':<10}")
    print(f"{'-'*48}")
    for r in results:
        print(f"{r['name']:<20} {r['num_matches']:>8} {r['num_inliers']:>8} {r['inlier_ratio']:>8.1f}%")
    print(f"{'-'*48}")

    avg_inlier_ratio = np.mean([r["inlier_ratio"] for r in results])
    print(f"{'平均':<20} {'':>8} {'':>8} {avg_inlier_ratio:>8.1f}%")

    print(f"\n结果保存在{output_dir}")


if __name__ == "__main__":
    main()
