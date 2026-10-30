# Environments/Sandbox_Pygame/mouse_physics_env.py
import math


class MousePhysicsEnv:
    def __init__(self):
        self.start_x, self.start_y = 0.0, 0.0
        self.target_x, self.target_y = 100.0, 100.0

    def reset(self):
        self.cur_x, self.cur_y = self.start_x, self.start_y
        self.vel_x, self.vel_y = 0.0, 0.0
        self.path = [(self.cur_x, self.cur_y)]
        return self.get_state()

    def get_state(self):
        # Giác quan cơ bắp: Tọa độ hiện tại, Vận tốc, Khoảng cách tới đích
        dx = self.target_x - self.cur_x
        dy = self.target_y - self.cur_y
        dist = math.sqrt(dx**2 + dy**2)
        return (self.cur_x, self.cur_y, self.vel_x, self.vel_y, dx, dy, dist)

    def step(self, force_x, force_y):
        # 1. Cập nhật Vận tốc (có ma sát 0.8 để tránh trượt vô tận)
        self.vel_x = (self.vel_x + force_x) * 0.8
        self.vel_y = (self.vel_y + force_y) * 0.8

        # 2. Cập nhật Tọa độ
        self.cur_x += self.vel_x
        self.cur_y += self.vel_y
        self.path.append((self.cur_x, self.cur_y))

        # 3. Tính khoảng cách mới
        dist = math.sqrt((self.target_x - self.cur_x)**2 + (self.target_y - self.cur_y)**2)
        return self.get_state(), dist