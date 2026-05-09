import cv2
import numpy as np


class HarrisCornerDetector:
    def __init__(self, k=0.04, window_size=5, sigma=1.0, threshold_ratio=0.015, nms_size=3, num_octaves=2, num_scales=3, scale_factor=1.3):
        # k: Harris响应公式中的经验常数
        # window_size: 高斯窗口大小
        # sigma: 高斯模糊的标准差
        # threshold_ratio: 阈值比例（相对于最大响应值）
        # nms_size: 非极大值抑制的窗口大小
        # num_octaves: 高斯金字塔的层数
        # num_scales: 每层金字塔的尺度数量
        # scale_factor: 尺度缩放因子
        self.k = k
        self.window_size = window_size
        self.sigma = sigma
        self.threshold_ratio = threshold_ratio
        self.nms_size = nms_size
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.scale_factor = scale_factor

    def _build_gaussian_pyramid(self, img):
        # 构建高斯金字塔
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        gray = gray.astype(np.float32)

        pyramid = []
        current_img = gray
        current_sigma = self.sigma

        for octave_idx in range(self.num_octaves):
            octave = []
            for scale_idx in range(self.num_scales):
                # 计算当前尺度的sigma值
                sigma_octave = current_sigma * (self.scale_factor ** scale_idx)
                # 确保卷积核大小是奇数
                ksize = int(2 * np.ceil(3 * sigma_octave) + 1)
                if ksize % 2 == 0:
                    ksize += 1
                blurred = cv2.GaussianBlur(current_img, (ksize, ksize), sigma_octave)
                octave.append((blurred, current_sigma * (self.scale_factor ** scale_idx)))

            pyramid.append(octave)

            # 下一层金字塔：将当前层最后一张图降采样
            if octave_idx < self.num_octaves - 1:
                last_blurred = octave[-1][0]
                new_h = max(20, last_blurred.shape[0] // 2)
                new_w = max(20, last_blurred.shape[1] // 2)
                current_img = cv2.resize(last_blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
                current_sigma *= 2

        return pyramid

    def _detect_single_scale(self, img):
        # 计算单一尺度上的Harris响应
        # 计算图像梯度
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度平方和叉积
        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        # 高斯窗口加权
        Sxx = cv2.GaussianBlur(Ixx, (self.window_size, self.window_size), self.sigma)
        Syy = cv2.GaussianBlur(Iyy, (self.window_size, self.window_size), self.sigma)
        Sxy = cv2.GaussianBlur(Ixy, (self.window_size, self.window_size), self.sigma)

        # 计算Harris响应
        det_M = Sxx * Syy - Sxy ** 2
        trace_M = Sxx + Syy
        R = det_M - self.k * (trace_M ** 2)

        return R

    def detect(self, img):
        # 多尺度检测角点
        all_corners = []
        height, width = img.shape[:2]

        pyramid = self._build_gaussian_pyramid(img)

        for octave_idx, octave in enumerate(pyramid):
            for scale_idx, (scaled_img, _) in enumerate(octave):
                current_scale = (2 ** octave_idx) * (self.scale_factor ** scale_idx)

                # 计算当前尺度的Harris响应
                R = self._detect_single_scale(scaled_img)

                # 阈值过滤，去除弱小响应
                if np.max(R) <= 1e-10:
                    continue
                threshold = self.threshold_ratio * np.max(R)
                R_thresholded = np.zeros_like(R)
                mask = (R > threshold) & (R > 0)
                R_thresholded[mask] = R[mask]

                # 非极大值抑制，保留局部最大值
                kernel = np.ones((self.nms_size, self.nms_size), np.uint8)
                R_dilated = cv2.dilate(R_thresholded, kernel)
                R_local_max = (R_thresholded == R_dilated) & (R_thresholded > 0)

                # 将角点坐标映射回原图尺度
                y_coords, x_coords = np.where(R_local_max)
                for y, x in zip(y_coords, x_coords):
                    orig_x = int(round(x * current_scale))
                    orig_y = int(round(y * current_scale))
                    all_corners.append((orig_x, orig_y))

        # 在原图尺度上，去除重叠的角点
        if len(all_corners) > 0:
            corners_np = np.array(all_corners)
            unique_corners = []
            used = np.zeros(len(corners_np), dtype=bool)
            min_dist = 5  # 5像素以内的角点合并保留一个
            for i in range(len(corners_np)):
                if used[i]:
                    continue
                unique_corners.append(tuple(corners_np[i]))
                dists = np.sqrt(np.sum((corners_np - corners_np[i]) ** 2, axis=1))
                used[dists < min_dist] = True
            all_corners = unique_corners

        return all_corners
