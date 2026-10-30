"""
Foveated Vision System - Shared preprocessing for M1a_Dorsal (Where pathway) and M1b_Ventral (What pathway).

This module provides a biologically-inspired foveated vision system that splits visual input
into:
- Peripheral vision (low resolution, wide field) -> 20D vector for M1a (Where pathway)
- Foveal vision (high resolution, narrow field) -> 20D vector for M1b (What pathway)

The system mimics the human retina's non-uniform photoreceptor density.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FoveatedVisionSystem:
    """
    Foveated vision system that processes images into peripheral and foveal feature vectors.
    
    Architecture:
    - Input: RGB image (H, W, 3) or grayscale (H, W)
    - Peripheral pathway: Low-res global features (20D) -> M1a_Dorsal (Where)
    - Foveal pathway: High-res local features (20D) -> M1b_Ventral (What)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (240, 320),
        fovea_size: Tuple[int, int] = (64, 64),
        peripheral_downsample: int = 8,
        fovea_downsample: int = 2,
        output_dim: int = 20
    ):
        """
        Initialize the foveated vision system.
        
        Args:
            input_shape: Input image shape (height, width)
            fovea_size: Size of foveal region (height, width)
            peripheral_downsample: Downsample factor for peripheral vision
            fovea_downsample: Downsample factor for foveal vision
            output_dim: Output dimension for both pathways (default 20)
        """
        self.input_shape = input_shape
        self.fovea_size = fovea_size
        self.peripheral_downsample = peripheral_downsample
        self.fovea_downsample = fovea_downsample
        self.output_dim = output_dim
        
        # Calculate output shapes
        self.peripheral_shape = (
            input_shape[0] // peripheral_downsample,
            input_shape[1] // peripheral_downsample
        )
        self.fovea_shape = (
            fovea_size[0] // fovea_downsample,
            fovea_size[1] // fovea_downsample
        )
        
        # Feature extractors (simple but effective)
        self._init_feature_extractors()
        
        # Fovea position (center by default, can be moved by attention)
        self.fovea_center = (input_shape[1] // 2, input_shape[0] // 2)
        
        logger.info(f"FoveatedVisionSystem initialized: input={input_shape}, "
                   f"peripheral={self.peripheral_shape}, fovea={self.fovea_shape}, "
                   f"output_dim={output_dim}")
    
    def _init_feature_extractors(self):
        """Initialize feature extraction kernels."""
        # Gabor-like filters for edge/orientation detection
        self.edge_kernels = self._create_edge_kernels()
        
        # Color opponent channels
        self.color_weights = np.array([
            [0.299, 0.587, 0.114],      # Luminance
            [0.5, -0.5, 0.0],           # Red-Green
            [0.25, 0.25, -0.5]          # Blue-Yellow
        ], dtype=np.float32)
    
    def _create_edge_kernels(self) -> np.ndarray:
        """Create oriented edge detection kernels."""
        kernels = []
        kernel_size = 5
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            # Gabor-like kernel
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    x = i - kernel_size // 2
                    y = j - kernel_size // 2
                    x_rot = x * np.cos(theta) + y * np.sin(theta)
                    y_rot = -x * np.sin(theta) + y * np.cos(theta)
                    kernel[i, j] = np.exp(-(x_rot**2 + y_rot**2) / 8) * np.cos(2 * np.pi * x_rot / 4)
            kernels.append(kernel)
        return np.stack(kernels)  # (4, 5, 5)
    
    def set_fovea_position(self, x: int, y: int, image_shape: Optional[Tuple[int, int]] = None):
        """Set fovea center position (for attention-driven saccades) bounded by actual image shape."""
        h, w = image_shape if (image_shape is not None and len(image_shape) >= 2) else self.input_shape
        half_w = min(self.fovea_size[1] // 2, max(0, w // 2))
        half_h = min(self.fovea_size[0] // 2, max(0, h // 2))
        self.fovea_center = (
            int(np.clip(x, half_w, max(half_w, w - half_w))),
            int(np.clip(y, half_h, max(half_h, h - half_h)))
        )

    @staticmethod
    def _extract_valid_gaze(topdown_focus: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        """Trích xuất và xác thực tọa độ gaze chuẩn hóa trong đoạn [0.0, 1.0]."""
        if not isinstance(topdown_focus, dict):
            return 0.5, 0.5
        gaze = topdown_focus.get("gaze")
        if isinstance(gaze, dict):
            raw_x = gaze.get("x", topdown_focus.get("gaze_x", 0.5))
            raw_y = gaze.get("y", topdown_focus.get("gaze_y", 0.5))
        else:
            raw_x = topdown_focus.get("gaze_x", topdown_focus.get("x", 0.5))
            raw_y = topdown_focus.get("gaze_y", topdown_focus.get("y", 0.5))
        try:
            fx = float(raw_x)
            fy = float(raw_y)
        except (TypeError, ValueError):
            return 0.5, 0.5
        if not np.isfinite(fx):
            fx = 0.5
        if not np.isfinite(fy):
            fy = 0.5
        return float(np.clip(fx, 0.0, 1.0)), float(np.clip(fy, 0.0, 1.0))
    
    def process(self, image: np.ndarray, is_bgr: bool = False, topdown_focus: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Process image into peripheral and foveal feature vectors.
        
        Args:
            image: Input image (H, W, 3) RGB/BGR or (H, W) grayscale
            is_bgr: Cờ báo hiệu ảnh đầu vào là BGR (từ cv2.imdecode)
            topdown_focus: Dict tọa độ lấy từ M9 chứa gaze={'x': 0..1, 'y': 0..1}
        """
        if isinstance(image, list):
            image = np.array(image)
            if image.ndim == 3: image = image.astype(np.uint8)
            else: image = image.astype(np.float32)
        elif isinstance(image, np.ndarray) and image.dtype == np.float64:
            image = image.astype(np.float32)

        if image.ndim == 1:
            if topdown_focus is not None:
                fx, fy = self._extract_valid_gaze(topdown_focus)
                h, w = self.input_shape
                self.set_fovea_position(int(round(fx * (w - 1))), int(round(fy * (h - 1))), image_shape=(h, w))
            vec = image[:self.output_dim].astype(np.float32)
            if len(vec) < self.output_dim:
                vec = np.pad(vec, (0, self.output_dim - len(vec)))
            return {'peripheral': vec, 'foveal': vec}

        # [BẢN VÁ MỤC 9]: Nhận, kiểm tra [0.0, 1.0] và ánh xạ gaze theo đúng kích thước ảnh thực tế
        actual_h, actual_w = int(image.shape[0]), int(image.shape[1])
        if topdown_focus is not None:
            fx, fy = self._extract_valid_gaze(topdown_focus)
            self.set_fovea_position(
                int(round(fx * max(0, actual_w - 1))),
                int(round(fy * max(0, actual_h - 1))),
                image_shape=(actual_h, actual_w)
            )
        else:
            self.set_fovea_position(self.fovea_center[0], self.fovea_center[1], image_shape=(actual_h, actual_w))

        # Ensure correct shape
        if image.ndim == 3 and image.shape[2] == 3:
            if is_bgr:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            scale = 255.0 if image.dtype == np.uint8 or np.max(image) > 1.0 else 1.0
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / scale
            color_channels = image.astype(np.float32) / scale
        elif image.ndim == 2:
            scale = 255.0 if image.dtype == np.uint8 or np.max(image) > 1.0 else 1.0
            gray = image.astype(np.float32) / scale
            color_channels = np.stack([gray, gray, gray], axis=-1)
        else:
            raise ValueError(f"Expected image with 2 or 3 channels, got {image.shape}")
        
        # Extract peripheral features (global, low-res)
        peripheral_vec = self._extract_peripheral(gray, color_channels)
        
        # Extract foveal features (local, high-res)
        foveal_vec = self._extract_foveal(gray, color_channels)
        
        return {
            'peripheral': peripheral_vec.astype(np.float32),
            'foveal': foveal_vec.astype(np.float32)
        }
    
    def _extract_peripheral(self, gray: np.ndarray, color: np.ndarray) -> np.ndarray:
        """Extract peripheral (where) features - 20D vector (4x5 spatial grid to match training semantics)."""
        # Training data (mock) assumes a 4x5 grid where each cell represents spatial intensity.
        # Resize the peripheral image to 5x4 (width=5, height=4) to get 20 pixels.
        small_gray = cv2.resize(gray, (5, 4), interpolation=cv2.INTER_AREA)
        
        # Flatten into 20D vector
        features = small_gray.flatten().astype(np.float32)
        
        # Normalize to [0, 1] range to match mock training (random.uniform(0.0, 0.1) and 1.0)
        max_val = np.max(features)
        if max_val > 0:
            features = features / max_val
            
        return features
    
    def _extract_foveal(self, gray: np.ndarray, color: np.ndarray) -> np.ndarray:
        """Extract foveal (what) features - 20D vector (4x5 spatial grid to match training semantics)."""
        fx, fy = self.fovea_center
        fh, fw = self.fovea_size
        
        # Extract foveal region
        y1 = fy - fh // 2
        y2 = y1 + fh
        x1 = fx - fw // 2
        x2 = x1 + fw
        
        # Clamp to image bounds
        y1 = max(0, y1)
        y2 = min(gray.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(gray.shape[1], x2)
        
        fovea_gray = gray[y1:y2, x1:x2]
        
        # Resize to 5x4 grid
        if fovea_gray.size > 0:
            small_fovea = cv2.resize(fovea_gray, (5, 4), interpolation=cv2.INTER_AREA)
        else:
            small_fovea = np.zeros((4, 5), dtype=np.float32)
            
        features = small_fovea.flatten().astype(np.float32)
        
        # Normalize
        max_val = np.max(features)
        if max_val > 0:
            features = features / max_val
            
        return features
    
    def get_peripheral_dim(self) -> int:
        """Get peripheral output dimension."""
        return self.output_dim
    
    def get_foveal_dim(self) -> int:
        """Get foveal output dimension."""
        return self.output_dim


def create_foveated_vision_system(config: Optional[Dict[str, Any]] = None) -> FoveatedVisionSystem:
    """Factory function to create FoveatedVisionSystem from config."""
    if config is None:
        config = {}
    return FoveatedVisionSystem(
        input_shape=config.get('input_shape', (240, 320)),
        fovea_size=config.get('fovea_size', (64, 64)),
        peripheral_downsample=config.get('peripheral_downsample', 8),
        fovea_downsample=config.get('fovea_downsample', 2),
        output_dim=config.get('output_dim', 20)
    )


# Backward compatibility alias
FoveatedVision = FoveatedVisionSystem