# Environments/Sandbox_Vision/dumb_vision_sandbox.py
import numpy as np
import random


class DumbVisionSandbox:
    def __init__(self, grid_size=64):
        self.grid_size = grid_size

    def generate_frame(self):
        # 1. Tạo màn hình nhiễu siêu mờ (Noise)
        frame = np.random.uniform(0.0, 0.1, (self.grid_size, self.grid_size))

        # 2. Sinh ra Vùng Chú Ý (ROI) - Khối sáng đại diện cho cửa sổ/chuyển động
        roi_w = random.randint(8, 16)
        roi_h = random.randint(8, 16)
        roi_x = random.randint(2, self.grid_size - roi_w - 2)
        roi_y = random.randint(2, self.grid_size - roi_h - 2)

        frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = 0.8

        # 3. Tọa độ mục tiêu AI cần nội suy ra (Tâm của vùng sáng)
        center_x = roi_x + roi_w / 2.0
        center_y = roi_y + roi_h / 2.0

        return frame, (center_x, center_y)