import cv2
import numpy as np


def get_descriptors(img, kp, patch_size=16, num_bins=8):
    # 提取SIFT风格的描述符
    # img: 输入图像
    # kp: 关键点列表
    # patch_size: 每个关键点周围的patch大小
    # num_bins: 每个子区域的方向直方图bins数量
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    descriptors = []
    used_kp = []  # 记录实际被使用的关键点（边界的会被跳过）
    half = patch_size // 2
    height, width = gray.shape

    for (x, y) in kp:
        # 检查patch是否在图像边界内
        x0 = int(round(x)) - half
        y0 = int(round(y)) - half
        x1 = x0 + patch_size
        y1 = y0 + patch_size

        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue

        used_kp.append((x, y))
        patch = gray[y0:y1, x0:x1].astype(np.float32)

        # 计算patch内梯度的幅值和方向
        Ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(Ix ** 2 + Iy ** 2)
        ang = np.arctan2(Iy, Ix) * 180 / np.pi

        # 高斯加权，中心区域梯度影响更大
        sigma = 1.5
        Y, X = np.mgrid[0:patch_size, 0:patch_size]
        Yc, Xc = patch_size / 2 - 0.5, patch_size / 2 - 0.5
        weights = np.exp(-((X - Xc) ** 2 + (Y - Yc) ** 2) / (2 * sigma ** 2))
        mag_weighted = mag * weights

        # 计算主方向（使用36-bin方向直方图）
        main_dir_hist = np.zeros(36, dtype=np.float32)
        for i in range(patch_size):
            for j in range(patch_size):
                m = mag_weighted[i, j]
                if m < 1e-6:
                    continue
                a = ang[i, j]
                if a < 0:
                    a += 360  # 方向转成0-360度
                bin_idx = int(a / 10) % 36
                main_dir_hist[bin_idx] += m

        # 取直方图最大值对应的方向作为主方向
        main_dir = np.argmax(main_dir_hist) * 10

        # 把所有梯度方向减去主方向（实现旋转不变性）
        mag_rot = mag_weighted
        ang_rot = (ang - main_dir) % 360

        # 构建描述符：4x4子区域，每个区域8-bin方向直方图
        desc = np.zeros((4, 4, num_bins), dtype=np.float32)
        cell_size = patch_size // 4

        for i in range(patch_size):
            for j in range(patch_size):
                m = mag_rot[i, j]
                if m < 1e-6:
                    continue

                a = ang_rot[i, j] % 360

                # 方向bin的三线性插值
                bin_f = a / (360 / num_bins)
                bin0 = int(np.floor(bin_f)) % num_bins
                bin1 = (bin0 + 1) % num_bins
                frac = bin_f - bin0

                # 子区域坐标的三线性插值
                ci_f = (i + 0.5) / cell_size - 0.5
                cj_f = (j + 0.5) / cell_size - 0.5

                ci0 = int(np.floor(ci_f))
                ci1 = ci0 + 1
                cj0 = int(np.floor(cj_f))
                cj1 = cj0 + 1

                # 把梯度投票到相邻的4个子区域和2个方向bin
                for ci in [ci0, ci1]:
                    for cj in [cj0, cj1]:
                        if 0 <= ci < 4 and 0 <= cj < 4:
                            w_ci = 1 - abs(ci_f - ci)
                            w_cj = 1 - abs(cj_f - cj)
                            w_space = w_ci * w_cj

                            w_bin0 = 1 - frac
                            w_bin1 = frac

                            desc[ci, cj, bin0] += m * w_space * w_bin0
                            desc[ci, cj, bin1] += m * w_space * w_bin1

        # 归一化（RootSIFT）
        desc_flat = desc.flatten()

        # 第一次归一化
        norm = np.linalg.norm(desc_flat)
        if norm > 1e-6:
            desc_flat = desc_flat / norm

        # 平方根操作
        desc_flat = np.sqrt(np.abs(desc_flat))

        # 第二次归一化
        norm = np.linalg.norm(desc_flat)
        if norm > 1e-6:
            desc_flat = desc_flat / norm

        descriptors.append(desc_flat)

    return descriptors, used_kp
