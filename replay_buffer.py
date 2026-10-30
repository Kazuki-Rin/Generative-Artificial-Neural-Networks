# replay_buffer.py - BỘ ĐỆM TRẢI NGHIỆM ĐỂ HỌC TĂNG CƯỜNG (EXPERIENCE REPLAY)
import random
import numpy as np
import threading
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(threadName)-15s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        Khởi tạo Replay Buffer dạng Circular Buffer (Vòng lặp con trỏ).
        Sử dụng List để đảm bảo random.sample chạy với tốc độ siêu tốc O(1).
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0 # ⚡ Con trỏ ghi đè (Pointer)
        self.lock = threading.Lock() # Khóa an toàn đa luồng
        logger.info(f"🧠 [Replay Buffer] Đã phân bổ RAM vòng (Circular). Dung lượng: {capacity} Experiences.")

    def push(self, state, action, reward, next_state, done):
        """
        Lưu trữ một trải nghiệm (Experience Tuple) vào bộ đệm.
        Định dạng: (S, A, R, S', Done)
        """
        with self.lock:
            # Nếu mảng chưa đầy, mở rộng kích thước
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            
            # Ghi đè dữ liệu vào vị trí con trỏ hiện tại
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
            # Tịnh tiến con trỏ, quay vòng về 0 nếu chạm đỉnh (Modulo)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=64):
        """
        Lấy mẫu ngẫu nhiên một Batch (32/64/128) với tốc độ O(1).
        Trả về Dictionary chứa các mảng Numpy.
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            
            # ⚡ BẢN VÁ: random.sample trên List là O(1) -> Nhanh gấp hàng nghìn lần trên Deque
            batch = random.sample(self.buffer, batch_size)
            
            # Giải nén (Unzip) batch thành các mảng song song
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Chuyển đổi sang Numpy Array
            return {
                "states": np.array(states, dtype=object),
                "actions": np.array(actions, dtype=object),
                "rewards": np.array(rewards, dtype=np.float32),
                "next_states": np.array(next_states, dtype=object),
                "dones": np.array(dones, dtype=bool)
            }

    def __len__(self):
        """Kiểm tra số lượng trải nghiệm hiện có trong Buffer"""
        with self.lock:
            return len(self.buffer)

    def is_ready(self, batch_size=64):
        """Kiểm tra xem Buffer đã đủ mập để bắt đầu train chưa"""
        with self.lock:
            return len(self.buffer) >= batch_size

    def clear(self):
        """Xóa sạch ký ức"""
        with self.lock:
            self.buffer.clear()
            self.position = 0
            logger.info("🗑️ [Replay Buffer] Đã xóa sạch toàn bộ ký ức trong đệm.")

# Test nhanh tốc độ của Buffer
if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=100000)
    
    logger.info("Ghi thử 100 trải nghiệm giả lập...")
    for i in range(100):
        buffer.push(f"state_{i}", f"click_{i}", 1.0 if i % 2 == 0 else -1.0, f"state_{i+1}", i == 99)
        
    logger.info(f"Đã ghi. Kích thước hiện tại: {len(buffer)}")
    
    batch = buffer.sample(batch_size=32)
    if batch:
        logger.info(f"Đã Sample thành công! Shape của Rewards: {batch['rewards'].shape}")
        logger.info(f"Dones batch sample: {batch['dones'][:5]}...")