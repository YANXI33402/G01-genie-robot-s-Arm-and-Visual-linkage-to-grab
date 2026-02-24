#!/usr/bin/env python3
"""
沙袋检测 - 使用a2d_sdk的CosineCamera接口获取head和head_depth图像
实时检测红色沙袋并输出相机坐标系位置
"""
import cv2
import numpy as np
import time
from a2d_sdk.robot import CosineCamera as Camera


class SandbagDetector:
    def __init__(self):
        # ===== 相机内参 =====
        self.fx = 644.0
        self.fy = 644.0
        self.cx = 636.0
        self.cy = 410.0

        # ===== 初始化相机 =====
        try:
            camera_group = ["head", "head_depth"]
            self.camera = Camera(camera_group)
            print('CosineCamera initialized successfully')
            self._first_frame_skipped = False
        except Exception as e:
            print(f'Failed to initialize camera: {e}')
            raise

        self.rgb = None
        self.depth = None
        self.depth_dtype = None

        # ===== 时间滤波 =====
        self.has_last = False
        self.Xf = self.Yf = self.Zf = 0.0

        # ===== HSV颜色阈值（红色检测）=====
        self.h_low1 = 0
        self.h_high1 = 10
        self.h_low2 = 160
        self.h_high2 = 179
        self.s_min = 160
        self.v_min = 30
        self.min_area = 300
        self.max_area = 2000
        self.max_depth_std = 0.03
        self.min_width_m = 0.05
        self.max_width_m = 0.4

        # ===== 滤波参数 =====
        self.alpha = 0.3

        # ===== 可视化与深度采样 =====
        self.visualize = True
        self.depth_patch = 1

    def acquire_images(self):
        """获取RGB-D图像"""
        try:
            # 获取最新RGB图像
            rgb_image, rgb_timestamp = self.camera.get_latest_image("head")

            # 跳过第一帧
            if not self._first_frame_skipped:
                if rgb_image is None:
                    return False
                self._first_frame_skipped = True
                print('First frame skipped, starting image acquisition')

            if rgb_image is None:
                return False

            # 获取对齐的深度图像
            depth_result = self.camera.get_image_nearest("head_depth", rgb_timestamp)

            # get_image_nearest 返回的是 (image, timestamp) 元组
            if depth_result is None:
                return False

            if isinstance(depth_result, tuple):
                depth_image, _ = depth_result
            else:
                depth_image = depth_result

            if depth_image is None:
                return False

            # 处理RGB图像格式
            if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
                self.rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            else:
                self.rgb = rgb_image

            # 处理深度图像格式
            self.depth = depth_image.copy()
            if self.depth.dtype == np.uint16:
                self.depth_dtype = np.uint16
            elif self.depth.dtype == np.float32:
                self.depth_dtype = np.float32
            else:
                self.depth_dtype = self.depth.dtype
                print(f'Unknown depth format: {self.depth.dtype}, assuming meters')

            return True

        except Exception as e:
            print(f'Error acquiring images: {e}')
            return False

    def process(self):
        """处理图像，检测红色沙袋"""
        if self.rgb is None or self.depth is None:
            return

        frame = self.rgb.copy()

        # ==================== HSV 双阈值分割（红色检测）====================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([self.h_low1, self.s_min, self.v_min]),
                           np.array([self.h_high1, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([self.h_low2, self.s_min, self.v_min]),
                           np.array([self.h_high2, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        # 亮度过滤
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = gray > 20
        mask = cv2.bitwise_and(mask, mask, mask=bright.astype(np.uint8) * 255)

        # 去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 可视化 mask（已注释）
        if self.visualize:
            cv2.imshow("rgb_mask", mask)

        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检测最大连通区域
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            x0, y0, w0, h0 = cv2.boundingRect(largest)

            # 计算中心像素坐标
            u0 = int(x0 + w0 / 2)
            v0 = int(y0 + h0 / 2)
            Zb = None
            p0 = self.depth_patch

            # 深度采样
            if (u0 >= p0 and v0 >= p0 and
                u0 < self.depth.shape[1] - p0 and
                v0 < self.depth.shape[0] - p0):
                patch0 = self.depth[v0-p0:v0+p0+1, u0-p0:u0+p0+1].astype(np.float32)
                patch0 = patch0[patch0 > 0]
                if patch0.size >= 3:
                    # 深度单位转换
                    if self.depth_dtype == np.uint16:
                        patchm0 = patch0 / 1000.0
                    else:
                        patchm0 = patch0.copy()
                    Zb = float(np.median(patchm0))

            if Zb is not None and not np.isnan(Zb) and Zb > 0.0:
                Xb = (u0 - self.cx) * Zb / self.fx
                Yb = (v0 - self.cy) * Zb / self.fy

                # 时间滤波
                if not self.has_last:
                    self.Xf, self.Yf, self.Zf = Xb, Yb, Zb
                    self.has_last = True
                else:
                    self.Xf = self.alpha * Xb + (1 - self.alpha) * self.Xf
                    self.Yf = self.alpha * Yb + (1 - self.alpha) * self.Yf
                    self.Zf = self.alpha * Zb + (1 - self.alpha) * self.Zf

                # 输出位置
                print(f"[Sandbag] X={self.Xf:.3f}, Y={self.Yf:.3f}, Z={self.Zf:.3f} m")

                # 可视化（已注释mask可视化部分，保留result窗口显示）
                if self.visualize:
                    cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 0, 255), 2)
                    text = f"X:{self.Xf:.3f} Y:{self.Yf:.3f} Z:{self.Zf:.3f} m"
                    cv2.putText(frame, text, (x0, y0 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("result", frame)
                    cv2.waitKey(1)
                return

        # 无检测结果时显示空画面
        if self.visualize:
            cv2.imshow("result", frame)
            cv2.waitKey(1)

    def run(self):
        """主循环"""
        print("Starting sandbag detection...")
        try:
            while True:
                if self.acquire_images():
                    self.process()
                time.sleep(0.001)  # 避免CPU占用过高
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.camera.close()
            print("Camera closed")
            cv2.destroyAllWindows()


def main():
    detector = SandbagDetector()
    detector.run()


if __name__ == '__main__':
    main()
