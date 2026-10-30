# Environments/Sandbox_Vision/smart_focus_sandbox.py
import numpy as np
import random


class SmartFocusSandbox:
    def __init__(self, size=32):
        self.size = size  # Ô điểm vàng siêu nét

    def generate_sample(self):
        # Nền nhiễu
        frame = np.random.uniform(0.0, 0.1, (self.size, self.size))

        # ID 0: Khoảng trống (Empty)
        # ID 1: Nút Close (Dấu X/Cộng)
        # ID 2: Khối Vuông (Icon/Viền)
        label = random.choice([0, 1, 2])

        if label == 1:
            # Vẽ hình chữ thập đại diện cho Nút Close
            cx, cy = self.size // 2, self.size // 2
            frame[cy - 8 : cy + 8, cx - 1 : cx + 2] = 1.0  # Trục dọc
            frame[cy - 1 : cy + 2, cx - 8 : cx + 8] = 1.0  # Trục ngang

        elif label == 2:
            # Vẽ khối vuông đại diện cho Icon/Cửa sổ
            frame[8:24, 8:24] = 1.0
            frame[10:22, 10:22] = 0.0  # Rỗng ruột

        # Ép thành mảng 1D (1024 pixel)
        return frame.flatten(), label