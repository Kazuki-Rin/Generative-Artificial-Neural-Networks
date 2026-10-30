#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, Union, List

class VPetEnvironment:
    """
    Môi trường VPet (Custom Gym-like).
    - Không kế thừa Gymnasium -> Không gọi super().
    - Render frame RGB thực tế cho M1 Vision.
    - Tính toán Motor Feedback chính xác (actual_dx/dy).
    - Phát hiện Hazard (Boundary, Collision, Unsafe Command).
    """
    def __init__(self, 
                 screen_width: int = 1920, 
                 screen_height: int = 1080, 
                 render_width: int = 320,   # P1-15: Frame nhẹ cho M1 (320x180 ~ 16:9)
                 render_height: int = 180,
                 seed: Optional[int] = None):
        
        self.width = float(screen_width)
        self.height = float(screen_height)
        
        # Render config (Low-res cho M1 foveate)
        self.render_width = render_width
        self.render_height = render_height
        self._scale_x = render_width / screen_width
        self._scale_y = render_height / screen_height
        
        # RNG
        self._np_random = np.random.default_rng(seed)
        
        # State
        self.asuna_x = self.width / 2.0
        self.asuna_y = self.height / 2.0
        self.target_x: Optional[float] = None
        self.target_y: Optional[float] = None
        
        # Metadata
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
        self._step_count = 0
        self._max_steps = 10000 # Truncated limit

    # ==========================================
    # GYM-LIKE INTERFACE (No Inheritance)
    # ==========================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Khởi tạo lại môi trường. Không gọi super()."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        
        self.asuna_x = self.width / 2.0
        self.asuna_y = self.height / 2.0
        self.target_x = None
        self.target_y = None
        self._step_count = 0
        
        # Spawn target ban đầu
        self.spawn_target(
            self._np_random.uniform(0, self.width),
            self._np_random.uniform(0, self.height)
        )
        
        obs = self._get_obs()
        # FIX: Truyền hazards mặc định cho reset (tránh missing 5 args)
        default_hazards = {
            "boundary_danger": 0.0,
            "target_collision_risk": 0.0,
            "unsafe_command": 0.0,
            "is_danger": 0
        }
        info = self._get_info(0.0, 0.0, default_hazards, 0.0, 0.0)
        return obs, info

    def step(self, action: Union[np.ndarray, list, Tuple[float, float, float]]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Thực hiện một bước.
        Return: (obs, reward, terminated, truncated, info)
        info chứa: actual_dx, actual_dy, hazards, distance...
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if len(action) < 3:
            action = np.pad(action, (0, max(0, 3 - len(action))))
            
        cmd_dx, cmd_dy = action[0], action[1]
        click = action[2]
        
        old_x, old_y = self.asuna_x, self.asuna_y
        
        # 1. Cập nhật vật lý
        self.update_position(cmd_dx, cmd_dy)
        
        # P1-17: Tính actual delta SAU khi update
        actual_dx = self.asuna_x - old_x
        actual_dy = self.asuna_y - old_y
        
        # 2. Tính Reward
        reward, is_success = self._compute_reward(old_x, old_y, click)
        if is_success:
            self.spawn_target(self._np_random.uniform(0, self.width), self._np_random.uniform(0, self.height))
        
        # 3. Hazard Detection (P1-16)
        hazards = self._detect_hazards(cmd_dx, cmd_dy, click)
        
        # 4. Termination checks
        terminated = self._check_terminated()
        self._step_count += 1
        truncated = self._step_count >= self._max_steps
        
        obs = self._get_obs()
        info = self._get_info(actual_dx, actual_dy, hazards, cmd_dx, cmd_dy)
        info["task_success"] = 1.0 if is_success else 0.0
        
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Trả về frame RGB cho M1 (Đã được get_visual_observation gọi)."""
        if mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self):
        pass

    # ==========================================
    # PHYSICS
    # ==========================================
    def update_position(self, dx: float, dy: float, speed_multiplier: float = 10.0) -> Tuple[float, float]:
        """Di chuyển Asuna, clamp ở viền màn hình."""
        self.asuna_x += dx * speed_multiplier
        self.asuna_y += dy * speed_multiplier
        
        # Clamp
        self.asuna_x = max(0.0, min(self.width, self.asuna_x))
        self.asuna_y = max(0.0, min(self.height, self.asuna_y))
        
        return self.asuna_x, self.asuna_y

    def spawn_target(self, x: float, y: float):
        """Tạo mục tiêu mới."""
        self.target_x = max(0.0, min(self.width, float(x)))
        self.target_y = max(0.0, min(self.height, float(y)))

    # ==========================================
    # OBSERVATION GETTERS
    # ==========================================
    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "visual": self.get_visual_observation(),
            "audio": self.get_audio_observation(),
            "spatial": self.get_spatial_observation(),
            "proprioception": self.get_proprioception()
        }

    def get_visual_observation(self) -> np.ndarray:
        """
        P1-15: Render frame RGB thực tế (Low-res).
        M1 sẽ nhận frame này -> FoveatedVisionSystem.resize -> Foveate.
        """
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """Vẽ frame RGB đơn giản bằng OpenCV (Numpy)."""
        frame = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        
        if self.target_x is not None and self.target_y is not None:
            tx = int(self.target_x * self._scale_x)
            ty = int(self.target_y * self._scale_y)
            cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1) 
            cv2.circle(frame, (tx, ty), 8, (0, 0, 100), 1) 
        
        ax = int(self.asuna_x * self._scale_x)
        ay = int(self.asuna_y * self._scale_y)
        cv2.circle(frame, (ax, ay), 6, (0, 255, 0), -1) 
        cv2.circle(frame, (ax, ay), 10, (0, 100, 0), 1)  
        return frame

    def get_audio_observation(self) -> np.ndarray:
        """
        [BẢN VÁ P0]: Cung cấp âm thanh định hướng. 
        Mảng 20D: 10 bins trái, 10 bins phải dựa trên tọa độ tương đối của mục tiêu.
        Tín hiệu to dần khi Asuna di chuyển tới gần.
        """
        audio = np.zeros(20, dtype=np.float32)
        if self.target_x is not None and self.target_y is not None:
            dx = (self.target_x - self.asuna_x) / self.width
            dy = (self.target_y - self.asuna_y) / self.height
            dist = np.sqrt(dx**2 + dy**2)
            intensity = np.clip(1.0 - dist, 0.0, 1.0)
            
            pan_left = np.clip(0.5 - dx, 0.0, 1.0)
            pan_right = np.clip(0.5 + dx, 0.0, 1.0)
            
            audio[:10] = intensity * pan_left + np.random.normal(0, 0.05, 10)
            audio[10:] = intensity * pan_right + np.random.normal(0, 0.05, 10)
            
        return np.clip(audio, 0.0, 1.0).astype(np.float32)

    def get_spatial_observation(self) -> np.ndarray:
        """Vector khoảng cách tương đối Asuna -> Target (Cho M3). Normalized [-1, 1]."""
        if self.target_x is None or self.target_y is None:
            return np.array([0.0, 0.0], dtype=np.float32)
            
        rel_x = (self.target_x - self.asuna_x) / self.width
        rel_y = (self.target_y - self.asuna_y) / self.height
        return np.array([rel_x, rel_y], dtype=np.float32)

    def get_proprioception(self) -> np.ndarray:
        """Vị trí tuyệt đối Asuna (Normalized 0-1)."""
        return np.array([self.asuna_x / self.width, self.asuna_y / self.height], dtype=np.float32)

    # ==========================================
    # HAZARD DETECTION (P1-16)
    # ==========================================
    def _detect_hazards(self, cmd_dx: float, cmd_dy: float, click: float) -> Dict[str, Any]:
        """
        Phát hiện các nguy hiểm:
        1. Boundary Danger: Gần viền màn hình (< 5% width/height).
        2. Target Collision: Đè lên target (distance < threshold) mà không click.
        3. Unsafe Motor Command: Command magnitude > 1.0 (saturation) hoặc NaN.
        """
        hazards = {
            "boundary_danger": 0.0,
            "target_collision_risk": 0.0,
            "unsafe_command": 0.0,
            "is_danger": 0 # Binary flag cho M5
        }
        
        # 1. Boundary Danger
        margin_x = self.width * 0.05
        margin_y = self.height * 0.05
        dist_left = self.asuna_x
        dist_right = self.width - self.asuna_x
        dist_top = self.asuna_y
        dist_bottom = self.height - self.asuna_y
        min_dist_boundary = min(dist_left, dist_right, dist_top, dist_bottom)
        min_margin = min(margin_x, margin_y)
        
        if min_dist_boundary < min_margin:
            hazards["boundary_danger"] = 1.0 - (min_dist_boundary / min_margin)
        
        # 2. Target Collision Risk
        if self.target_x is not None:
            dist_target = np.sqrt((self.target_x - self.asuna_x)**2 + (self.target_y - self.asuna_y)**2)
            collision_radius = 30.0
            if dist_target < collision_radius and click < 0.5:
                hazards["target_collision_risk"] = 1.0 - (dist_target / collision_radius)
        
        # 3. Unsafe Command
        cmd_mag = np.sqrt(cmd_dx**2 + cmd_dy**2)
        if np.isnan(cmd_mag) or np.isinf(cmd_mag):
            hazards["unsafe_command"] = 1.0
        elif cmd_mag > 1.0:
            hazards["unsafe_command"] = min(1.0, (cmd_mag - 1.0) / 2.0)
            
        # Binary flag cho M5 (Amygdala)
        if any(v > 0.5 for v in hazards.values()):
            hazards["is_danger"] = 1
            
        return hazards

    # ==========================================
    # REWARD & TERMINATION
    # ==========================================
    def _compute_reward(self, old_x: float, old_y: float, click: float) -> Tuple[float, bool]:
        if self.target_x is None or self.target_y is None:
            return 0.0, False
        
        dist_before = np.sqrt((self.target_x - old_x)**2 + (self.target_y - old_y)**2)
        dist_after = np.sqrt((self.target_x - self.asuna_x)**2 + (self.target_y - self.asuna_y)**2)
        
        progress_reward = (dist_before - dist_after) / max(self.width, self.height) * 10.0
        
        click_reward = 0.0
        is_success = False
        
        if click > 0.5 and dist_after < 50.0:
            click_reward = 10.0
            is_success = True
            
        return float(progress_reward + click_reward), is_success

    def _check_terminated(self) -> bool:
        return False # Continuous task

    # FIX: Tất cả tham số optional với default
    def _get_info(self, 
                  actual_dx: float = 0.0, 
                  actual_dy: float = 0.0, 
                  hazards: Optional[Dict] = None, 
                  cmd_dx: float = 0.0, 
                  cmd_dy: float = 0.0) -> Dict[str, Any]:
        """
        Trả về info dict. 
        Khi gọi từ reset(): actual_dx/dy=0, hazards=rỗng, cmd_dx/dy=0.
        Khi gọi từ step(): truyền đầy đủ 5 tham số.
        """
        if hazards is None:
            hazards = {
                "boundary_danger": 0.0,
                "target_collision_risk": 0.0,
                "unsafe_command": 0.0,
                "is_danger": 0
            }
            
        dist = -1.0
        if self.target_x is not None:
            dist = np.sqrt((self.target_x - self.asuna_x)**2 + (self.target_y - self.asuna_y)**2)
            
        return {
            "asuna_pos": (self.asuna_x, self.asuna_y),
            "target_pos": (self.target_x, self.target_y),
            "distance": dist,
            # P1-17: Motor Feedback chính xác
            "actual_dx": actual_dx,
            "actual_dy": actual_dy,
            "command_dx": cmd_dx,
            "command_dy": cmd_dy,
            # P1-16: Hazards
            "hazards": hazards,
            "step_count": self._step_count
        }

# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    print("🌍 [TEST] VPetEnvironment Fixed (P0-8, P1-15, P1-16, P1-17)...")
    env = VPetEnvironment(screen_width=1920, screen_height=1080, render_width=320, render_height=180)
    
    obs, info = env.reset(seed=42)
    print(f"✅ Reset OK. Keys: {list(obs.keys())}")
    print(f"   Visual (Frame) shape: {obs['visual'].shape} (dtype: {obs['visual'].dtype})")
    print(f"   Spatial shape: {obs['spatial'].shape}")
    print(f"   Audio shape: {obs['audio'].shape}")
    print(f"   Proprioception shape: {obs['proprioception'].shape}")
    print(f"   Info keys: {list(info.keys())}")
    print(f"   Hazards: {info['hazards']}")
    
    # Test Step
    for i in range(3):
        spatial = obs['spatial']
        action = [np.clip(spatial[0]*5, -1, 1), np.clip(spatial[1]*5, -1, 1), 0.0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n   Step {i+1}: Reward={reward:.4f}, Dist={info['distance']:.1f}px")
        print(f"   Cmd: ({action[0]:.2f}, {action[1]:.2f}) -> Actual: ({info['actual_dx']:.2f}, {info['actual_dy']:.2f})")
        print(f"   Hazards: {info['hazards']}")
        
        if i == 0:
            env.asuna_x = 10.0; env.asuna_y = 10.0
            obs, reward, terminated, truncated, info = env.step([0,0,0])
            print(f"   [Corner Test] Hazards: {info['hazards']}")
            assert info['hazards']['boundary_danger'] > 0.5, "Boundary hazard failed!"

    print("\n✅ Tất cả tests PASSED!")
    print("   - P0-8: No super() call.")
    print("   - P1-15: Visual is RGB Frame (320x180x3).")
    print("   - P1-16: Hazards dict in info (boundary, collision, unsafe_cmd).")
    print("   - P1-17: actual_dx/dy calculated from position delta.")