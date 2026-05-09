import numpy as np


def match_features(desc1, desc2, ratio_thresh=0.75):
    # 特征匹配，使用SSD距离和Lowe比率测试，带双向匹配验证
    # desc1: 图1的描述符列表
    # desc2: 图2的描述符列表
    # ratio_thresh: Lowe比率阈值
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    desc1_np = np.array(desc1)
    desc2_np = np.array(desc2)

    # 从图1往图2匹配
    matches_1to2 = []
    for i in range(len(desc1_np)):
        d1 = desc1_np[i]
        # 计算所有描述符之间的SSD距离
        dists = np.sum((desc2_np - d1) ** 2, axis=1)
        idx_sorted = np.argsort(dists)
        best_idx = idx_sorted[0]  # 最近邻
        second_idx = idx_sorted[1]  # 次近邻
        best_dist = dists[best_idx]
        second_dist = dists[second_idx]

        # Lowe比率测试：最近邻距离要显著小于次近邻
        if second_dist > 1e-10 and best_dist < ratio_thresh * second_dist:
            matches_1to2.append((i, best_idx, best_dist))

    # 双向匹配：再从图2往图1匹配一次
    matches_2to1 = []
    for j in range(len(desc2_np)):
        d2 = desc2_np[j]
        dists = np.sum((desc1_np - d2) ** 2, axis=1)
        idx_sorted = np.argsort(dists)
        best_idx = idx_sorted[0]
        second_idx = idx_sorted[1]
        best_dist = dists[best_idx]
        second_dist = dists[second_idx]

        if second_dist > 1e-10 and best_dist < ratio_thresh * second_dist:
            matches_2to1.append((best_idx, j, best_dist))

    # 只保留双向一致的匹配
    set_2to1_dict = {(i, j) for i, j, d in matches_2to1}
    consistent = []
    for (i, j, d) in matches_1to2:
        if (i, j) in set_2to1_dict:
            consistent.append((i, j, d))

    # 去重，确保每个点只参与一次匹配
    used1 = set()
    used2 = set()
    final = []
    consistent_sorted = sorted(consistent, key=lambda x: x[2])
    for i, j, d in consistent_sorted:
        if i not in used1 and j not in used2:
            final.append((i, j))
            used1.add(i)
            used2.add(j)

    return final
