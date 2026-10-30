# Kazuki_Rin_Project/Environments/Sandbox_Vision/vision_sandbox.py
import numpy as np
import random
import matplotlib.pyplot as plt

class VisionSandbox:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size

    def generate_frame(self):
        # 1. Nền nhiễu tối (Desktop ảo)
        frame = np.random.uniform(0.0, 0.2, (self.grid_size, self.grid_size))

        # 2. Sinh ra Cửa sổ phần mềm (Window) ngẫu nhiên kích thước và vị trí
        win_w = random.randint(10, 20)
        win_h = random.randint(10, 20)
        win_x = random.randint(2, self.grid_size - win_w - 2)
        win_y = random.randint(2, self.grid_size - win_h - 2)

        # Tô màu xám cho toàn bộ phần mềm
        frame[win_y : win_y + win_h, win_x : win_x + win_w] = 0.5

        # Vẽ viền cửa sổ sáng hơn một chút (Border)
        frame[win_y, win_x : win_x + win_w] = 0.7               # Viền trên
        frame[win_y + win_h - 1, win_x : win_x + win_w] = 0.7   # Viền dưới
        frame[win_y : win_y + win_h, win_x] = 0.7               # Viền trái
        frame[win_y : win_y + win_h, win_x + win_w - 1] = 0.7   # Viền phải

        # 3. Mục tiêu: Vẽ Nút Close (Góc trên cùng bên phải cửa sổ)
        # Pixel này sẽ sáng rực nhất
        btn_x = win_x + win_w - 2
        btn_y = win_y + 1
        frame[btn_y, btn_x] = 1.0

        return frame, (btn_x, btn_y)

    def test_visualization(self, num_samples=5):
        print(f"🛠️ Đang sinh ra {num_samples} bức ảnh test Cửa sổ Giao diện...")
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            frame, (x, y) = self.generate_frame()
            axes[i].imshow(frame, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f"Close Button: ({x}, {y})")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    env = VisionSandbox()
    env.test_visualization()