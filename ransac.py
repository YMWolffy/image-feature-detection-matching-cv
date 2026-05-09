import cv2
import numpy as np
import random


def ransac_homography(kp1, kp2, matches, threshold=20.0, max_iter=8000):
    # 使用RANSAC算法鲁棒估计单应矩阵
    # kp1: 图1的关键点
    # kp2: 图2的关键点
    # matches: 匹配对列表
    # threshold: 内点判定的距离阈值
    # max_iter: RANSAC最大迭代次数
    if len(matches) < 4:
        return None, []

    # 提取匹配点的坐标
    pts1 = []
    pts2 = []
    for i, j in matches:
        pts1.append(kp1[i])
        pts2.append(kp2[j])

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    best_H = None
    best_inlier_mask = None
    max_inliers = 0

    num_matches = len(matches)

    # RANSAC主循环
    for _ in range(max_iter):
        # 随机采样4个点
        sample_indices = random.sample(range(num_matches), 4)
        sample_pts1 = pts1[sample_indices]
        sample_pts2 = pts2[sample_indices]

        # 计算单应矩阵
        try:
            H, _ = cv2.findHomography(sample_pts1, sample_pts2, 0)
        except:
            continue

        if H is None:
            continue

        # 用当前H变换所有点
        pts1_h = np.column_stack((pts1, np.ones(num_matches)))
        pts2_proj_h = pts1_h @ H.T

        # 齐次坐标转欧氏坐标
        z = pts2_proj_h[:, 2]
        valid = np.abs(z) > 1e-10
        pts2_proj = np.zeros_like(pts2)
        pts2_proj[valid] = pts2_proj_h[valid, :2] / z[valid, np.newaxis]

        # 计算投影误差，判断内点
        errors = np.sqrt(np.sum((pts2_proj - pts2) ** 2, axis=1))
        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)

        # 更新最优模型
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inlier_mask = inlier_mask

    # 用所有内点重新计算一个更精确的H
    if best_H is not None and max_inliers >= 4:
        inlier_indices = np.where(best_inlier_mask)[0]
        inlier_pts1 = pts1[inlier_indices]
        inlier_pts2 = pts2[inlier_indices]

        try:
            best_H, _ = cv2.findHomography(inlier_pts1, inlier_pts2, 0)
        except:
            pass

    # 收集内点匹配
    inlier_matches = []
    if best_inlier_mask is not None:
        for i, is_inlier in enumerate(best_inlier_mask):
            if is_inlier:
                inlier_matches.append(matches[i])

    return best_H, inlier_matches
