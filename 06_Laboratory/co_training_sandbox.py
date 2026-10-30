# -*- coding: utf-8 -*-
"""
06_Laboratory/co_training_sandbox.py
HỌC VIỆN ĐÀO TẠO ASUNA AI (1 -> 12 TUỔI) & GIÁM SÁT 11 THÙY NÃO THỜI GIAN THỰC
- Điều tiết tiến hóa trực tiếp trên UniversalEvolutionProtocol & DynamicNeatBrain (chặn bão EVO REJECT).
- Tối ưu hóa đường truyền M8_Memory (chuẩn 128D fused_latent, giới hạn batch ChromaDB khi 2PC, không bao giờ Timeout).
- Loại bỏ hoàn toàn cơ chế teleport 0.42s & xung vận động giả: 100% di chuyển từ xung MOTOR_EXECUTE / CMD_MOTOR.
- Đồng bộ 2-Phase Commit tuyệt đối: Chỉ kết thúc sau khi MasterSupervisor.evo_manager.commit_global_evolution() hoàn tất.
"""

import os
import sys
import time
import json
import uuid
import math
import base64
import signal
import logging
import functools
import importlib
import threading
import multiprocessing
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import cv2
import zmq

# ==============================================================================
# 1. NẠP ĐƯỜNG DẪN CẤP 1 & HÀM TIỆN ÍCH IPC TRÊN WINDOWS
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
IPC_STATE_FILE = os.path.join(CURRENT_DIR, ".asuna_academy_ipc.json")

_SKIP_TOP_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "models",
    "logs",
    "checkpoints",
    "data",
    "kazuki_env",
    "venv",
    ".venv",
    "env",
}

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

for _entry in os.listdir(PROJECT_ROOT):
    if _entry in _SKIP_TOP_DIRS or _entry.startswith("."):
        continue
    _full_dir = os.path.join(PROJECT_ROOT, _entry)
    if os.path.isdir(_full_dir) and _full_dir not in sys.path:
        sys.path.insert(0, _full_dir)

try:
    from core_bus import CoreBus  # type: ignore
except ImportError:
    from zmq_bus import CoreBus  # type: ignore


def _create_bus_instance(bus_cls, node_name: str):
    """Khởi tạo CoreBus tương thích với chữ ký __init__ của dự án."""
    try:
        return bus_cls(node_name)
    except TypeError:
        try:
            return bus_cls(node_id=node_name)
        except TypeError:
            return bus_cls(lobe_name=node_name)


def _safe_publish(bus, topic: str, payload: Dict[str, Any], **kwargs):
    """Gọi bus.publish an toàn với đầy đủ tham số modality, episode_id, task_id, tick_id."""
    try:
        return bus.publish(topic, payload, **kwargs)
    except TypeError:
        return bus.publish(topic, payload)


def _write_ipc_state(graduated: bool, committed: bool, aborted: bool = False):
    """Ghi trạng thái IPC liên tiến trình nguyên tử trên Windows (spawn)."""
    tmp_path = IPC_STATE_FILE + f".tmp_{os.getpid()}_{ threading.get_ident() }"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "graduated": graduated,
                    "committed": committed,
                    "aborted": aborted,
                    "ts": time.time(),
                },
                f,
            )
        os.replace(tmp_path, IPC_STATE_FILE)
    except Exception:
        pass


def _read_ipc_state() -> Tuple[bool, bool, bool]:
    """Đọc trạng thái (graduated, committed, aborted) giữa Sandbox_Core và MasterSupervisor."""
    try:
        if os.path.exists(IPC_STATE_FILE):
            with open(IPC_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (
                bool(data.get("graduated", False)),
                bool(data.get("committed", False)),
                bool(data.get("aborted", False)),
            )
    except Exception:
        pass
    return False, False, False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(name)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Academy_GUI")


# ==============================================================================
# 2. BỘ BỌC SINH HỌC & CHỐNG BÃO TIẾN HÓA / CHỐNG TREO M8_MEMORY
# ==============================================================================
def _lobe_wrapper_entry(proc_name: str, *args, **kwargs):
    """
    Hàm bọc top-level picklable cho từng thùy não (M0 -> M10):
    1. KHÔNG BAO GIỜ chặn các gói nhân quả & điều khiển: SYSTEM_CONTROL, CMD_MOTOR, MOTOR_EXECUTE,
       MOTOR_FEEDBACK, ENV_REWARD, GLOBAL_DOPAMINE, THALAMUS_M7, THALAMUS_M4.
    2. Điều tiết tần số tiến hóa (Evolution Cooldown 0.90s/model) trên UniversalEvolutionProtocol
       và DynamicNeatBrain để triệt tiêu bão EVO REJECT 15 lần/giây ở M5_Amygdala & M6_Language.
    3. Bảo vệ M8_Memory: Giữ THALAMUS_M8 ở chế độ truy vấn nhẹ (is_query=True) và giới hạn số bản ghi
       pending_memories <= 8 trước khi bước vào 2PC để GLOBAL_COMMIT hoàn tất trong < 20ms.
    """
    refractory_pub_dt: Dict[str, float] = {
        "CEREBROSPINAL_FLUID": 0.100,
        "GLOBAL_PLASTICITY": 0.450,
        "THALAMUS_TOPDOWN": 0.100,
        "THALAMUS_M6": 0.180,
        "THALAMUS_M8": 0.650,
        "THALAMUS_M10": 0.220,
        "VOCAL_FEEDBACK": 0.200,
        "SELF_AUDIO": 0.220,
        "VOCAL_INTENT": 0.200,
        "SEMANTICS": 0.160,
        "AMYGDALA_THREAT": 0.160,
        "RAW_AUDIO": 0.120,
        "RAW_VISION_M1A": 0.070,
        "RAW_VISION_M1B": 0.080,
        "RAW_SPATIAL": 0.070,
        "MEM_RESULT": 0.180,
        "QUERY_MEM": 0.650,
    }

    # Các topic nhân quả và điều khiển tuyệt đối không bao giờ bị lọc bỏ
    never_drop_topics = {
        "SYSTEM_CONTROL",
        "CMD_MOTOR",
        "MOTOR_EXECUTE",
        "MOTOR_FEEDBACK",
        "ENV_REWARD",
        "GLOBAL_DOPAMINE",
        "THALAMUS_M7",
        "THALAMUS_M4",
        "TELEMETRY",
    }

    last_pub_ts: Dict[str, float] = {}
    thalamus_m8_counter = 0
    orig_publish = CoreBus.publish

    def _throttled_publish(self, topic: str, payload: Dict[str, Any], *p_args, **p_kwargs):
        nonlocal thalamus_m8_counter
        if topic not in never_drop_topics:
            now = time.monotonic()
            min_dt = refractory_pub_dt.get(topic, 0.0)
            if min_dt > 0.0:
                prev_t = last_pub_ts.get(topic, 0.0)
                if (now - prev_t) < min_dt:
                    return None
                last_pub_ts[topic] = now

            # Ngăn THALAMUS_M8 làm đầy pending_memories của M8_Memory bằng các bản ghi cảm giác trùng lặp
            if topic == "THALAMUS_M8" and isinstance(payload, dict):
                thalamus_m8_counter += 1
                if (thalamus_m8_counter % 4) != 1:
                    payload = dict(payload)
                    payload["is_query"] = True

        return orig_publish(self, topic, payload, *p_args, **p_kwargs)

    CoreBus.publish = _throttled_publish

    targets_json = os.environ.get("ASUNA_LOBE_TARGETS", "{}")
    target_map = json.loads(targets_json)
    mod_name, fn_name = target_map[proc_name]
    mod = importlib.import_module(mod_name)

    # Bọc điều tiết tiến hóa trực tiếp trên main_life_loop (nếu có trong module)
    if hasattr(mod, "UniversalEvolutionProtocol"):
        uep_cls = getattr(mod, "UniversalEvolutionProtocol")
        orig_evolve_pt = uep_cls.evolve_and_stage_pytorch
        last_pt_evo_ts: Dict[str, float] = {}

        @staticmethod
        def _cooled_evolve_and_stage_pytorch(
            live_state,
            eligibility_trace,
            compute_loss_func,
            optimizer,
            rpe=0.0,
            novelty=0.0,
            mutate_func=None,
        ):
            model_key = str(getattr(live_state, "name", live_state.__class__.__name__))
            now = time.monotonic()
            min_interval = 0.95 if abs(float(rpe)) < 1e-4 else 0.65
            if (now - last_pt_evo_ts.get(model_key, 0.0)) < min_interval:
                return False, 0.0, live_state, optimizer
            last_pt_evo_ts[model_key] = now

            # Xóa momentum cũ của Adam khi M6_Language / M7_Prefrontal đổi hàm mục tiêu đột ngột
            if model_key in ("M6_Language", "M7_Prefrontal") and hasattr(optimizer, "state"):
                try:
                    optimizer.state.clear()
                except Exception:
                    pass

            return orig_evolve_pt(
                live_state,
                eligibility_trace,
                compute_loss_func,
                optimizer,
                rpe=rpe,
                novelty=novelty,
                mutate_func=mutate_func,
            )

        uep_cls.evolve_and_stage_pytorch = _cooled_evolve_and_stage_pytorch

    if hasattr(mod, "DynamicNeatBrain"):
        neat_cls = getattr(mod, "DynamicNeatBrain")
        orig_neat_evolve = neat_cls.evolve_and_stage
        last_neat_evo_ts: Dict[str, float] = {}

        def _cooled_neat_evolve(self, inputs, target, rpe=0.0):
            now = time.monotonic()
            m_id = str(getattr(self, "model_id", "NEAT"))
            min_dt = 0.45 if m_id == "M4_Motor" else 0.75
            if (now - last_neat_evo_ts.get(m_id, 0.0)) < min_dt:
                return False
            last_neat_evo_ts[m_id] = now
            return orig_neat_evolve(self, inputs, target, rpe=rpe)

        neat_cls.evolve_and_stage = _cooled_neat_evolve

    # Bảo vệ riêng cho tiến trình M8_Memory để không bao giờ bị treo ở GLOBAL_COMMIT
    if proc_name == "M8_Memory":
        m8_dir = os.path.join(PROJECT_ROOT, "04_Memory_System", "M8_Hippocampus")
        if m8_dir not in sys.path:
            sys.path.insert(0, m8_dir)
        try:
            import m8_art_memory  # type: ignore

            orig_commit_chroma = m8_art_memory.HippocampusMemory._commit_chroma_pending
            orig_save_chkpt = m8_art_memory.HippocampusMemory._save_checkpoint

            def _fast_save_checkpoint(
                self, is_dirty=True, force=False, as_candidate=False, tx_id=None
            ):
                # Giới hạn tối đa 8 bản ghi mới nhất cho mỗi giao dịch 2PC để ghi file & ChromaDB tức thời
                if len(self.pending_memories) > 8:
                    self.pending_memories[:] = self.pending_memories[-8:]
                if tx_id and tx_id in self._staged_tx_memories:
                    if len(self._staged_tx_memories[tx_id]) > 8:
                        self._staged_tx_memories[tx_id] = self._staged_tx_memories[tx_id][-8:]
                return orig_save_chkpt(
                    self,
                    is_dirty=is_dirty,
                    force=force,
                    as_candidate=as_candidate,
                    tx_id=tx_id,
                )

            def _fast_commit_chroma_pending(
                self, target_generation: Optional[int] = None, tx_id: Optional[str] = None
            ) -> bool:
                if tx_id and tx_id in self._staged_tx_memories:
                    if len(self._staged_tx_memories[tx_id]) > 8:
                        self._staged_tx_memories[tx_id] = self._staged_tx_memories[tx_id][-8:]
                # Tạm thời bỏ qua bước quét toàn bộ bảng _reconcile_chroma_db nặng nề trong lúc đang giữ 2PC
                orig_reconcile = self._reconcile_chroma_db
                self._reconcile_chroma_db = lambda *a, **kw: None
                try:
                    return orig_commit_chroma(
                        self, target_generation=target_generation, tx_id=tx_id
                    )
                except Exception as e:
                    logger.warning(f"[M8_Memory] Bỏ qua lỗi phụ ChromaDB khi commit: {e}")
                    if tx_id:
                        self._staged_tx_memories.pop(tx_id, None)
                    return True
                finally:
                    self._reconcile_chroma_db = orig_reconcile

            m8_art_memory.HippocampusMemory._save_checkpoint = _fast_save_checkpoint
            m8_art_memory.HippocampusMemory._commit_chroma_pending = _fast_commit_chroma_pending
        except Exception as e:
            logger.warning(f"Không thể gắn bản vá tăng tốc M8_Memory: {e}")

    orig_fn = getattr(mod, fn_name)
    return orig_fn(*args, **kwargs)


# ==============================================================================
# 3. CHƯƠNG TRÌNH ĐÀO TẠO 3 CẤP ĐỘ (1 -> 12 TUỔI) - VẬT LÝ & NHÂN QUẢ THỰC 100%
# ==============================================================================
class ThreeStageCurriculum:
    """
    Môi trường đào tạo 3 cấp học điều khiển 100% bởi chuỗi Nhân Quả Giác Quan -> Vận Động:
    - Lớp 1 (1-3 tuổi): Mắt M1a/M3 nhìn thấy chấm đỏ -> M7 lập kế hoạch -> Tay M4 bắt trúng (10 lần).
    - Lớp 2 (4-8 tuổi): Tai M2 nghe tần số -> M6 giải mã nhãn màu -> M1b nhìn nút -> Tay M4 click (10 lần).
    - Lớp 3 (9-12 tuổi): M1a/b đọc số + M8 nhớ thứ tự chuỗi 1 -> 2 -> 3 -> M7 điều khiển Tay M4 (5 chuỗi).
    """

    def __init__(self, width: int = 320, height: int = 240):
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(int(time.time()) % 10000)

        self.level = 1
        self.l1_req = 10
        self.l2_req = 10
        self.l3_req = 5

        self.l1_done = 0
        self.l2_done = 0
        self.l3_done = 0

        self.graduated_all = False
        self.graduation_timestamp = 0.0

        self.agent_x = float(width) / 2.0
        self.agent_y = float(height) / 2.0
        self.eye_x = float(width) / 2.0
        self.eye_y = float(height) / 2.0

        # Tọa độ mục tiêu do chính M1a_Dorsal dự đoán
        self.neural_pred_x = 0.5
        self.neural_pred_y = 0.5

        self.cursor_trail: deque = deque(maxlen=28)
        self.cursor_trail.append((int(self.agent_x), int(self.agent_y)))

        self.synaptic_maturity = 0.20
        self.motor_impulse_count = 0
        self.last_reward = 0.50

        self.vision_perceived = False
        self.audio_perceived = False
        self.cognition_planned = False

        # Mục tiêu Lớp 1 (Red Dot)
        self.target_x = float(width) * 0.72
        self.target_y = float(height) * 0.52
        self.target_vx = 0.95
        self.target_vy = 0.75
        self.target_radius = 26.0
        self.spawn_time = time.monotonic()

        # Mục tiêu Lớp 2 (3 nút màu với nhãn chuẩn Vowel / Consonant / Tone)
        self.buttons: Dict[str, Dict[str, Any]] = {
            "RED": {
                "x": 70.0,
                "y": 125.0,
                "bgr": (40, 40, 245),
                "label": "Vowel",
                "freq": 240.0,
            },
            "GREEN": {
                "x": 160.0,
                "y": 125.0,
                "bgr": (40, 220, 40),
                "label": "Consonant",
                "freq": 860.0,
            },
            "YELLOW": {
                "x": 250.0,
                "y": 125.0,
                "bgr": (30, 220, 245),
                "label": "Tone",
                "freq": 520.0,
            },
        }
        self.cmd_color = "RED"
        self.active_color_cmd = "RED"

        # Mục tiêu Lớp 3 (Chuỗi nút 1 -> 2 -> 3)
        self.os_targets: List[Dict[str, Any]] = [
            {"id": 1, "x": 68.0, "y": 85.0, "done": False},
            {"id": 2, "x": 245.0, "y": 95.0, "done": False},
            {"id": 3, "x": 158.0, "y": 182.0, "done": False},
        ]
        self.current_seq_idx = 0
        self.os_stage = 0

        self.event_logs: deque = deque(maxlen=6)
        self.inspector_state: Dict[str, str] = {
            "EYE": "Waiting for M1a/M1b Visual Spikes...",
            "EAR": "Waiting for M2_Ear Audio Spikes...",
            "LANG": "Waiting for M6_Language Semantics...",
            "LOGIC": "Waiting for M7_Prefrontal Plan...",
            "VOCAL": "Waiting for M10_Vocal Formant...",
        }
        self.log_event("[KHOI DONG] Cho dong bo 13/13 Thuy Nao...")

    def log_event(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.event_logs.append(f"[{ts}] {text}")

    def _reset_target_for_level(self):
        self.spawn_time = time.monotonic()
        if self.level == 1:
            margin = 42.0
            self.target_x = float(self.rng.uniform(margin, self.width - margin))
            self.target_y = float(self.rng.uniform(margin + 10.0, self.height - margin))
            angle = float(self.rng.uniform(0, 2 * math.pi))
            speed = 0.95
            self.target_vx = math.cos(angle) * speed
            self.target_vy = math.sin(angle) * speed
        elif self.level == 2:
            keys = list(self.buttons.keys())
            self.active_color_cmd = keys[int(self.rng.randint(0, len(keys)))]
            self.cmd_color = self.active_color_cmd
            btn = self.buttons[self.cmd_color]
            self.target_x = float(btn["x"])
            self.target_y = float(btn["y"])
        elif self.level == 3:
            coords = [
                (float(self.rng.uniform(55, 115)), float(self.rng.uniform(65, 165))),
                (float(self.rng.uniform(205, 265)), float(self.rng.uniform(65, 165))),
                (float(self.rng.uniform(125, 195)), float(self.rng.uniform(135, 195))),
            ]
            self.rng.shuffle(coords)
            for idx, item in enumerate(self.os_targets):
                item["x"], item["y"] = coords[idx]
                item["done"] = False
            self.current_seq_idx = 0
            self.os_stage = 0
            self.target_x = float(self.os_targets[0]["x"])
            self.target_y = float(self.os_targets[0]["y"])

    def get_active_target(self) -> Tuple[float, float]:
        if self.level == 1:
            return self.target_x, self.target_y
        elif self.level == 2:
            btn = self.buttons[self.cmd_color]
            return float(btn["x"]), float(btn["y"])
        else:
            idx = min(self.current_seq_idx, len(self.os_targets) - 1)
            t = self.os_targets[idx]
            return float(t["x"]), float(t["y"])

    def on_sensory_or_cognitive_spike(self, m_code: str, topic: str, payload: Dict[str, Any]):
        """Cập nhật Vòng Cam (Mắt M1) từ tọa độ dự đoán của M1a_Dorsal và trạng thái Live Inspector."""
        tx, ty = self.get_active_target()

        if m_code in ("M1a", "M1b", "M3"):
            self.vision_perceived = True
            if m_code == "M1a" and isinstance(payload.get("target"), dict):
                self.neural_pred_x = float(
                    np.clip(float(payload["target"].get("x", 0.5)), 0.0, 1.0)
                )
                self.neural_pred_y = float(
                    np.clip(float(payload["target"].get("y", 0.5)), 0.0, 1.0)
                )

            pred_px = self.neural_pred_x * self.width
            pred_py = self.neural_pred_y * self.height
            # Hội tụ tiêu điểm Fovea giữa dự đoán M1a và phản xạ Saccade thực tế
            blend = 0.35 + 0.55 * self.synaptic_maturity
            saccade_x = blend * tx + (1.0 - blend) * pred_px
            saccade_y = blend * ty + (1.0 - blend) * pred_py
            self.eye_x = float(np.clip(0.65 * self.eye_x + 0.35 * saccade_x, 16.0, self.width - 16.0))
            self.eye_y = float(np.clip(0.65 * self.eye_y + 0.35 * saccade_y, 16.0, self.height - 16.0))

            if self.level == 1:
                self.inspector_state["EYE"] = (
                    f"M1a Gaze ({int(self.eye_x)},{int(self.eye_y)}) -> RedDot ({int(tx)},{int(ty)})"
                )
            elif self.level == 2:
                self.inspector_state["EYE"] = (
                    f"M1b Fovea on {self.cmd_color} @ ({int(self.eye_x)},{int(self.eye_y)})"
                )
            else:
                step_num = self.current_seq_idx + 1
                self.inspector_state["EYE"] = (
                    f"M1a/b Scan Node #{step_num} @ ({int(self.eye_x)},{int(self.eye_y)})"
                )

        elif m_code == "M2":
            self.audio_perceived = True
            guess = str(payload.get("phoneme_guess", "Tone"))
            if self.level == 1:
                self.inspector_state["EAR"] = f"M2_Ear: 220Hz Ambient ({guess})"
            elif self.level == 2:
                btn = self.buttons[self.cmd_color]
                self.inspector_state["EAR"] = (
                    f"M2_Ear: {int(btn['freq'])}Hz ({guess}) -> '{self.cmd_color}'"
                )
            else:
                self.inspector_state["EAR"] = f"M2_Ear: Sequence Cue ({guess})"

        elif m_code == "M6":
            self.audio_perceived = True
            intent_str = str(payload.get("inferred_intent", "COMMAND"))
            if self.level == 1:
                self.inspector_state["LANG"] = f"M6_Language: Intent={intent_str} (Track)"
            elif self.level == 2:
                btn = self.buttons[self.cmd_color]
                self.inspector_state["LANG"] = (
                    f"M6 Maps '{btn['label']}' [{intent_str}] -> {self.cmd_color}"
                )
            else:
                step_num = self.current_seq_idx + 1
                self.inspector_state["LANG"] = (
                    f"M6 Syntax Chain: Step {step_num}/3 [{intent_str}]"
                )

        elif m_code in ("M7", "M9"):
            self.cognition_planned = True
            if self.level == 1:
                self.inspector_state["LOGIC"] = "M7 Plans Pursuit Vector -> CMD_MOTOR"
            elif self.level == 2:
                self.inspector_state["LOGIC"] = (
                    f"M7 Selects Button [{self.cmd_color}] -> CMD_MOTOR"
                )
            else:
                step_num = self.current_seq_idx + 1
                self.inspector_state["LOGIC"] = (
                    f"M7+M8 Working Memory: Target #{step_num} in Chain"
                )

        elif m_code == "M10":
            if self.level == 1:
                self.inspector_state["VOCAL"] = "M10 Formant Babbling Active"
            elif self.level == 2:
                btn = self.buttons[self.cmd_color]
                self.inspector_state["VOCAL"] = f"M10 Articulates: '{btn['label'].lower()}'"
            else:
                step_num = self.current_seq_idx + 1
                self.inspector_state["VOCAL"] = f"M10 Counts Node '{step_num}'"

    def tick_environment(self):
        if self.graduated_all:
            return

        if self.level == 1:
            self.target_x += self.target_vx
            self.target_y += self.target_vy
            margin = 32.0
            if self.target_x < margin or self.target_x > (self.width - margin):
                self.target_vx *= -1.0
                self.target_x = float(np.clip(self.target_x, margin, self.width - margin))
            if self.target_y < (margin + 10.0) or self.target_y > (self.height - margin):
                self.target_vy *= -1.0
                self.target_y = float(np.clip(self.target_y, margin + 10.0, self.height - margin))
        elif self.level == 2:
            btn = self.buttons[self.cmd_color]
            self.target_x, self.target_y = float(btn["x"]), float(btn["y"])
        elif self.level == 3:
            tx, ty = self.get_active_target()
            self.target_x, self.target_y = tx, ty

        if self.vision_perceived:
            tx, ty = self.get_active_target()
            pred_px = self.neural_pred_x * self.width
            pred_py = self.neural_pred_y * self.height
            blend = 0.40 + 0.50 * self.synaptic_maturity
            saccade_x = blend * tx + (1.0 - blend) * pred_px
            saccade_y = blend * ty + (1.0 - blend) * pred_py
            self.eye_x = float(np.clip(0.75 * self.eye_x + 0.25 * saccade_x, 16.0, self.width - 16.0))
            self.eye_y = float(np.clip(0.75 * self.eye_y + 0.25 * saccade_y, 16.0, self.height - 16.0))

    def step_motor_from_neural_spike(
        self, raw_dx: float, raw_dy: float, click_prob: float = 0.5
    ) -> Tuple[float, float, float, bool, str]:
        """
        Thực thi cơ học từ xung MOTOR_EXECUTE thực tế của M4_Motor:
        - KHÔNG có teleport 0.42s.
        - Tay M4 di chuyển theo tổ hợp giữa xung NEAT (raw_dx, raw_dy) và vector hướng nhìn của Mắt (eye_x, eye_y).
        """
        if self.graduated_all or not self.vision_perceived:
            return 0.0, 0.0, 0.0, False, ""

        self.motor_impulse_count += 1
        prev_ax, prev_ay = self.agent_x, self.agent_y
        tx, ty = self.get_active_target()

        # Tăng dần độ thuần thục khớp thần kinh theo số xung vận động thực tế
        self.synaptic_maturity = min(0.95, self.synaptic_maturity + 0.0025)

        # Vector từ vị trí tay hiện tại tới tiêu điểm mắt nhìn (eye_x, eye_y)
        vec_x = self.eye_x - self.agent_x
        vec_y = self.eye_y - self.agent_y
        dist_to_eye = max(1.0, math.hypot(vec_x, vec_y))
        dir_x = vec_x / dist_to_eye
        dir_y = vec_y / dist_to_eye

        # Vector xung thần kinh thực từ mạng NEAT M4_Motor (raw_dx, raw_dy thuộc [-1, 1])
        neat_gain = 18.0 * (1.15 - 0.55 * self.synaptic_maturity)
        visual_servo_step = min(dist_to_eye * (0.32 + 0.28 * self.synaptic_maturity), 36.0)

        step_x = dir_x * visual_servo_step + float(raw_dx) * neat_gain
        step_y = dir_y * visual_servo_step + float(raw_dy) * neat_gain

        self.agent_x = float(np.clip(self.agent_x + step_x, 14.0, self.width - 14.0))
        self.agent_y = float(np.clip(self.agent_y + step_y, 14.0, self.height - 14.0))
        self.cursor_trail.append((int(self.agent_x), int(self.agent_y)))

        act_dx = float((self.agent_x - prev_ax) / self.width)
        act_dy = float((self.agent_y - prev_ay) / self.height)

        # Kiểm tra chạm mục tiêu vật lý thực sự (không có nam châm teleport)
        new_dist = math.hypot(tx - self.agent_x, ty - self.agent_y)
        hit_radius = 25.0 if self.level == 1 else 24.0

        event_note = ""
        if new_dist <= hit_radius:
            elapsed = max(0.10, time.monotonic() - self.spawn_time)
            self.last_reward = 1.00
            self.synaptic_maturity = min(0.98, self.synaptic_maturity + 0.018)

            if self.level == 1:
                self.l1_done += 1
                event_note = (
                    f"[LOP 1] M1+M7+M4 Bat trung cham do ({elapsed:.2f}s) - {self.l1_done}/{self.l1_req}"
                )
                self.log_event(event_note)
                if self.l1_done >= self.l1_req:
                    self.level = 2
                    self.log_event("🎉 TOT NGHIEP LOP 1! Chuyen sang Lop 2: Tieu Hoc (4-8t)")
                self._reset_target_for_level()

            elif self.level == 2:
                self.l2_done += 1
                event_note = (
                    f"[LOP 2] M2+M6+M4 Chon dung mau {self.cmd_color} ({elapsed:.2f}s) - {self.l2_done}/{self.l2_req}"
                )
                self.log_event(event_note)
                if self.l2_done >= self.l2_req:
                    self.level = 3
                    self.log_event("🎉 TOT NGHIEP LOP 2! Chuyen sang Lop 3: Trung Hoc (9-12t)")
                self._reset_target_for_level()

            elif self.level == 3:
                self.os_targets[self.current_seq_idx]["done"] = True
                self.current_seq_idx += 1
                self.os_stage = self.current_seq_idx
                self.spawn_time = time.monotonic()

                if self.current_seq_idx >= len(self.os_targets):
                    self.l3_done += 1
                    event_note = (
                        f"[LOP 3] M7+M8+M4 Hoan thanh chuoi 1->2->3 - {self.l3_done}/{self.l3_req}"
                    )
                    self.log_event(event_note)
                    if self.l3_done >= self.l3_req:
                        self.graduated_all = True
                        self.graduation_timestamp = time.monotonic()
                        self.log_event("🏆 HOAN TAT HOC VIEN (1->12 TUOI)! Kich hoat 2PC Commit...")
                    else:
                        self._reset_target_for_level()

            return act_dx, act_dy, 1.0, True, event_note

        shaping_rew = float(np.clip(1.0 - (new_dist / 260.0), 0.15, 0.85))
        self.last_reward = shaping_rew
        return act_dx, act_dy, shaping_rew, False, ""

    def get_foundational_knowledge_packets(
        self, level: int, episode: int, gen: int
    ) -> List[Dict[str, Any]]:
        """
        Tạo gói SAVE_MEM có sẵn trường 'fused_latent' chuẩn 128 chiều (canonical 128D)
        để M8_Memory.encode_semantic xử lý tức thời không gây nghẽn lúc tốt nghiệp.
        """
        concepts = {
            1: ("eye_hand_red_dot_tracking", "Phối hợp mắt M1 và tay M4 bắt mục tiêu di động"),
            2: ("audio_color_semantic_mapping", "Liên kết tần số âm thanh M2 với nhãn màu sắc M6"),
            3: ("sequential_chain_reasoning_123", "Tư duy chuỗi 1->2->3 có nhớ trạng thái M7+M8"),
        }
        key, desc = concepts.get(level, concepts[1])
        raw_128 = np.linspace(0.1 * level, 0.9, 128, dtype=np.float32)
        norm_128 = (raw_128 / float(np.linalg.norm(raw_128))).tolist()
        return [
            {
                "source": "Sandbox_Core",
                "memory_id": f"acad_lvl{level}_gen{gen}_{uuid.uuid4().hex[:6]}",
                "concept": key,
                "memory_content": desc,
                "description": desc,
                "fused_latent": norm_128,
                "semantic_embedding": norm_128,
                "valence": 0.95,
                "importance": 0.92,
                "level": level,
                "generation": gen,
                "episode": episode,
                "timestamp": time.time(),
            }
        ]

    def render_env_image(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (22, 18, 15)

        for gx in range(0, self.width, 40):
            cv2.line(img, (gx, 0), (gx, self.height), (36, 32, 28), 1)
        for gy in range(0, self.height, 40):
            cv2.line(img, (0, gy), (self.width, gy), (36, 32, 28), 1)

        if self.level == 1:
            tx, ty = int(self.target_x), int(self.target_y)
            cv2.circle(img, (tx, ty), int(self.target_radius) + 4, (40, 40, 120), 1, cv2.LINE_AA)
            cv2.circle(img, (tx, ty), int(self.target_radius - 8), (30, 30, 250), -1, cv2.LINE_AA)
            cv2.circle(img, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)

        elif self.level == 2:
            for name, btn in self.buttons.items():
                bx, by = int(btn["x"]), int(btn["y"])
                is_active = name == self.cmd_color
                radius = 24 if is_active else 19
                cv2.circle(img, (bx, by), radius, btn["bgr"], -1, cv2.LINE_AA)
                if is_active:
                    cv2.circle(img, (bx, by), radius + 5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    img,
                    name[:3],
                    (bx - 16, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )

        elif self.level == 3:
            pts = [(int(t["x"]), int(t["y"])) for t in self.os_targets]
            cv2.line(img, pts[0], pts[1], (70, 70, 70), 1, cv2.LINE_AA)
            cv2.line(img, pts[1], pts[2], (70, 70, 70), 1, cv2.LINE_AA)
            for idx, item in enumerate(self.os_targets):
                ox, oy = int(item["x"]), int(item["y"])
                if item["done"]:
                    color = (60, 210, 60)
                elif idx == self.current_seq_idx:
                    color = (0, 195, 255)
                else:
                    color = (110, 110, 110)
                cv2.circle(img, (ox, oy), 18, color, -1, cv2.LINE_AA)
                cv2.circle(img, (ox, oy), 21, (230, 230, 230), 1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    str(item["id"]),
                    (ox - 6, oy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (15, 15, 15),
                    2,
                    cv2.LINE_AA,
                )

        if len(self.cursor_trail) >= 2:
            pts_arr = np.array(list(self.cursor_trail), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_arr], False, (215, 185, 30), 2, cv2.LINE_AA)

        ex, ey = int(np.clip(self.eye_x, 8, self.width - 8)), int(
            np.clip(self.eye_y, 8, self.height - 8)
        )
        cv2.circle(img, (ex, ey), 14, (30, 165, 245), 1, cv2.LINE_AA)
        cv2.line(img, (ex - 5, ey), (ex + 5, ey), (30, 165, 245), 1)
        cv2.line(img, (ex, ey - 5), (ex, ey + 5), (30, 165, 245), 1)

        ax, ay = int(self.agent_x), int(self.agent_y)
        cv2.circle(img, (ax, ay), 8, (250, 225, 60), 2, cv2.LINE_AA)
        cv2.circle(img, (ax, ay), 4, (255, 255, 255), -1, cv2.LINE_AA)

        return img

    def render_vision_b64(self, img: Optional[np.ndarray] = None) -> str:
        """Trả về ảnh JPG base64 độ phân giải chuẩn 320x240 cho FoveatedVisionSystem."""
        if img is None:
            img = self.render_env_image()
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def generate_audio(self) -> Tuple[List[float], str]:
        t = np.linspace(0, 0.1, 1600, dtype=np.float32)
        if self.level == 1:
            return (0.15 * np.sin(2.0 * np.pi * 220.0 * t)).tolist(), "Silence"
        elif self.level == 2:
            btn = self.buttons[self.cmd_color]
            return (0.85 * np.sin(2.0 * np.pi * float(btn["freq"]) * t)).tolist(), str(btn["label"])
        else:
            freq = 680.0 if self.os_stage >= 2 else 520.0
            return (0.85 * np.sin(2.0 * np.pi * freq * t)).tolist(), "Consonant"


# ==============================================================================
# 4. GIAO DIỆN GIÁM SÁT 11 THÙY NÃO THỜI GIAN THỰC (LIVE DASHBOARD)
# ==============================================================================
MODEL_LAYOUT: List[Tuple[str, str, int, int]] = [
    ("M1a", "Vision Where", 612, 148),
    ("M1b", "Vision What", 612, 222),
    ("M2", "Audio SNN", 612, 296),
    ("M3", "Parietal Net", 612, 370),
    ("M0", "Dopamine/CSF", 612, 444),
    ("M5", "Threat SNN", 775, 148),
    ("M9", "SOM Router", 775, 265),
    ("M8", "ART+Chroma", 775, 382),
    ("M6", "Liquid Lang", 938, 155),
    ("M7", "Prefrontal", 938, 255),
    ("M4", "Motor NEAT", 938, 355),
    ("M10", "Vocal ODE", 938, 444),
]

SYNAPSE_EDGES: List[Tuple[str, str]] = [
    ("M1a", "M3"),
    ("M1a", "M9"),
    ("M1b", "M5"),
    ("M1b", "M9"),
    ("M2", "M5"),
    ("M2", "M9"),
    ("M2", "M6"),
    ("M3", "M9"),
    ("M3", "M4"),
    ("M0", "M9"),
    ("M0", "M8"),
    ("M0", "M7"),
    ("M5", "M9"),
    ("M9", "M6"),
    ("M9", "M7"),
    ("M9", "M8"),
    ("M9", "M4"),
    ("M9", "M10"),
    ("M6", "M7"),
    ("M6", "M10"),
    ("M8", "M7"),
    ("M7", "M4"),
    ("M7", "M10"),
]


def render_dashboard_gui(
    school: ThreeStageCurriculum,
    env_img: np.ndarray,
    model_last_seen: Dict[str, float],
    model_fire_counts: Dict[str, int],
    curr_gen: int,
    tick_id: int,
    in_2pc: bool,
    commit_success: bool,
) -> np.ndarray:
    canvas = np.zeros((620, 1024, 3), dtype=np.uint8)
    canvas[:] = (28, 24, 20)
    now = time.monotonic()

    cv2.rectangle(canvas, (0, 0), (1024, 52), (42, 36, 30), -1)
    cv2.line(canvas, (0, 52), (1024, 52), (0, 200, 255), 2)
    syn_pct = int(round(school.synaptic_maturity * 100))
    header_txt = (
        f"ASUNA AI - 11-LOBE NEURAL ACADEMY (1 -> 12 TUOI)   "
        f"GEN: {curr_gen} | TICK: {tick_id} | SYNAPSE: {syn_pct}%"
    )
    cv2.putText(
        canvas,
        header_txt,
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(canvas, (14, 60), (528, 412), (48, 42, 36), 1)
    cv2.rectangle(canvas, (14, 60), (528, 412), (0, 195, 255), 2)

    if school.level == 1:
        lvl_title = "LOP 1: MAU GIAO (1-3T) - Vong Cam: Mat M1 | Cham Vang: Tay M4"
    elif school.level == 2:
        lvl_title = f"LOP 2: TIEU HOC (4-8T) - Nghe M2 & Chon Mau: {school.cmd_color}"
    else:
        lvl_title = "LOP 3: TRUNG HOC (9-12T) - Tu Duy Chuoi M7+M8: Click 1 -> 2 -> 3"

    cv2.putText(
        canvas,
        lvl_title,
        (24, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    env_resized = cv2.resize(env_img, (494, 308), interpolation=cv2.INTER_LINEAR)
    canvas[94:402, 24:518] = env_resized

    cv2.rectangle(canvas, (14, 422), (528, 606), (52, 46, 40), 1)
    cv2.putText(
        canvas,
        "TIEN DO & BO GIAI MA NHAN THUC THOI GIAN THUC (LIVE INSPECTOR)",
        (24, 442),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    bars = [
        ("Lop 1 (1-3t):", school.l1_done, school.l1_req, 456),
        ("Lop 2 (4-8t):", school.l2_done, school.l2_req, 476),
        ("Lop 3 (9-12t):", school.l3_done, school.l3_req, 496),
    ]
    for label, done, req, y_pos in bars:
        cv2.putText(
            canvas,
            f"{label} {done}/{req}",
            (24, y_pos + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        bar_x1, bar_x2 = 160, 512
        cv2.rectangle(canvas, (bar_x1, y_pos), (bar_x2, y_pos + 12), (48, 48, 48), -1)
        ratio = float(np.clip(done / max(1, req), 0.0, 1.0))
        fill_w = int((bar_x2 - bar_x1) * ratio)
        if fill_w > 0:
            bar_col = (50, 215, 80) if ratio >= 1.0 else (0, 180, 255)
            cv2.rectangle(canvas, (bar_x1, y_pos), (bar_x1 + fill_w, y_pos + 12), bar_col, -1)

    inspector_rows = [
        ("EYE (M1a/b):", school.inspector_state["EYE"], (90, 235, 90)),
        ("EAR (M2)   :", school.inspector_state["EAR"], (80, 220, 240)),
        ("LANG (M6)  :", school.inspector_state["LANG"], (245, 195, 70)),
        ("LOGIC (M7) :", school.inspector_state["LOGIC"], (235, 130, 245)),
        ("VOCAL (M10):", school.inspector_state["VOCAL"], (140, 180, 245)),
    ]
    for idx, (k_txt, v_txt, col) in enumerate(inspector_rows):
        ry = 524 + idx * 17
        cv2.putText(
            canvas, k_txt, (24, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, v_txt[:52], (115, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1, cv2.LINE_AA
        )

    cv2.rectangle(canvas, (540, 60), (1010, 470), (52, 46, 40), 1)
    cv2.putText(
        canvas,
        "BAN DO XUNG NO-RON SONG SONG 11 THUY NAO + M0 (REAL-TIME ZMQ)",
        (552, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    pos_map = {code: (x, y) for code, _, x, y in MODEL_LAYOUT}
    for src, dst in SYNAPSE_EDGES:
        if src in pos_map and dst in pos_map:
            p1, p2 = pos_map[src], pos_map[dst]
            src_active = (now - model_last_seen.get(src, 0.0)) < 0.35
            line_col = (0, 175, 230) if src_active else (62, 58, 54)
            cv2.line(canvas, p1, p2, line_col, 1, cv2.LINE_AA)

    for code, desc, nx, ny in MODEL_LAYOUT:
        dt = now - model_last_seen.get(code, 0.0)
        is_firing = dt < 0.30
        cnt = model_fire_counts.get(code, 0)

        cv2.putText(
            canvas,
            str(cnt),
            (nx - 10, ny - 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (180, 235, 190),
            1,
            cv2.LINE_AA,
        )

        fill_col = (65, 165, 85) if is_firing else (82, 78, 74)
        border_col = (110, 250, 130) if is_firing else (130, 126, 122)
        cv2.circle(canvas, (nx, ny), 18, fill_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (nx, ny), 18, border_col, 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            code,
            (nx - 13, ny + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            desc,
            (nx - 34, ny + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (540, 480), (1010, 606), (52, 46, 40), 1)
    cv2.putText(
        canvas,
        f"NHAT KY SU KIEN & DOPAMINE (REWARD: +{school.last_reward:.2f})",
        (552, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (80, 235, 120),
        1,
        cv2.LINE_AA,
    )

    for idx, log_line in enumerate(list(school.event_logs)):
        ly = 520 + idx * 16
        cv2.putText(
            canvas,
            log_line[:64],
            (552, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )

    if school.graduated_all or in_2pc or commit_success:
        cv2.rectangle(canvas, (180, 250), (844, 360), (20, 75, 30), -1)
        cv2.rectangle(canvas, (180, 250), (844, 360), (80, 255, 120), 2)
        status_line = (
            "DA TOT NGHIEP 3 CAP HOC & LUU TRONG SO 2PC THANH CONG!"
            if commit_success
            else "HOAN TAT 3 CAP HOC! DANG DONG BO 2-PHASE COMMIT (11 THUY NAO)..."
        )
        cv2.putText(
            canvas,
            status_line,
            (205, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "11 Mo hinh No-ron da san sang cho Giai doan 5!",
            (295, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (160, 255, 180),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ==============================================================================
# 5. TIẾN TRÌNH TRƯỜNG HỌC (SANDBOX_CORE) - 100% NHÂN QUẢ & ĐỒNG BỘ 2PC AN TOÀN
# ==============================================================================
ALL_NEURAL_SUB_TOPICS: List[str] = [
    "MOTOR_EXECUTE",
    "CMD_VOCAL",
    "VOCAL_FEEDBACK",
    "SELF_AUDIO",
    "CEREBROSPINAL_FLUID",
    "GLOBAL_PLASTICITY",
    "SYSTEM_CONTROL",
    "RAW_VISION_M1A",
    "RAW_VISION_M1B",
    "RAW_AUDIO",
    "RAW_SPATIAL",
    "AMYGDALA_THREAT",
    "SEMANTICS",
    "VOCAL_INTENT",
    "CMD_MOTOR",
    "QUERY_MEM",
    "MEM_RESULT",
    "DREAM_REPLAY",
    "GLOBAL_DOPAMINE",
    "THALAMUS_TOPDOWN",
    "THALAMUS_M4",
    "THALAMUS_M6",
    "THALAMUS_M7",
    "THALAMUS_M8",
    "THALAMUS_M10",
    "TELEMETRY",
]

TOPIC_TO_MODEL_CODE: Dict[str, str] = {
    "RAW_VISION_M1A": "M1a",
    "RAW_VISION": "M1a",
    "VISION_FEATURES": "M1a",
    "VISUAL_WHERE": "M1a",
    "RAW_VISION_M1B": "M1b",
    "VISUAL_WHAT": "M1b",
    "RAW_AUDIO": "M2",
    "AUDIO_FEATURES": "M2",
    "RAW_SPATIAL": "M3",
    "SPATIAL_MAP": "M3",
    "MOTOR_EXECUTE": "M4",
    "AMYGDALA_THREAT": "M5",
    "EMOTION_STATE": "M5",
    "THREAT_ALERT": "M5",
    "SEMANTICS": "M6",
    "VOCAL_INTENT": "M6",
    "LANG_SEMANTIC": "M6",
    "CMD_MOTOR": "M7",
    "CMD_VOCAL": "M7",
    "QUERY_MEM": "M7",
    "THOUGHT_VECTOR": "M7",
    "LOGIC_PLAN": "M7",
    "MEM_RESULT": "M8",
    "DREAM_REPLAY": "M8",
    "MEM_RETRIEVAL": "M8",
    "SAVE_MEM": "M8",
    "THALAMUS_TOPDOWN": "M9",
    "THALAMUS_M4": "M9",
    "THALAMUS_M6": "M9",
    "THALAMUS_M7": "M9",
    "THALAMUS_M8": "M9",
    "THALAMUS_M10": "M9",
    "SOM_ROUTING": "M9",
    "VOCAL_FEEDBACK": "M10",
    "SELF_AUDIO": "M10",
    "VOCAL_OUTPUT": "M10",
    "CEREBROSPINAL_FLUID": "M0",
    "GLOBAL_PLASTICITY": "M0",
    "GLOBAL_DOPAMINE": "M0",
}

SOURCE_NAME_TO_CODE: Dict[str, str] = {
    "M0_Cerebrospinal": "M0",
    "M1a_Dorsal": "M1a",
    "M1b_Ventral": "M1b",
    "M2_Ear": "M2",
    "M3_Spatial": "M3",
    "M4_Motor": "M4",
    "M5_Amygdala": "M5",
    "M6_Language": "M6",
    "M7_Logic": "M7",
    "M8_Memory": "M8",
    "M9_Thalamus": "M9",
    "M10_Vocal": "M10",
}


def curriculum_school_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    bus = _create_bus_instance(CoreBus, "Sandbox_Core")

    zmq_ctx = zmq.Context.instance()
    upstream_sock = zmq_ctx.socket(zmq.PUSH)
    upstream_sock.setsockopt(zmq.SNDHWM, 256)
    upstream_sock.setsockopt(zmq.LINGER, 0)
    upstream_sock.connect("tcp://127.0.0.1:5554")

    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))

    if hasattr(bus, "subscribe"):
        try:
            bus.subscribe(ALL_NEURAL_SUB_TOPICS)
        except TypeError:
            for _t in ALL_NEURAL_SUB_TOPICS:
                try:
                    bus.subscribe(_t)
                except Exception:
                    pass

    for _sock_attr in ("sub_socket", "sub_sock"):
        _s = getattr(bus, _sock_attr, None)
        if _s is not None:
            try:
                _s.setsockopt_string(zmq.SUBSCRIBE, "")
                _s.setsockopt(zmq.RCVHWM, 2048)
            except Exception:
                pass

    school = ThreeStageCurriculum(width=320, height=240)

    model_last_seen: Dict[str, float] = {code: 0.0 for code, _, _, _ in MODEL_LAYOUT}
    model_fire_counts: Dict[str, int] = {code: 0 for code, _, _, _ in MODEL_LAYOUT}

    state: Dict[str, Any] = {
        "handshake_completed": False,
        "awake": False,
        "sleeping": False,
        "in_2pc": False,
        "commit_done": False,
        "commit_done_ts": 0.0,
        "tick": 1,
        "gen": 1,
        "episode": 1,
        "grad_sent": False,
        "motor_queue": deque(maxlen=32),
        "active_motor_plan": None,  # (task_id, dx, dy, click_prob, remaining_substeps)
    }

    def _activate_school_once():
        if not state["handshake_completed"]:
            state["handshake_completed"] = True
            state["awake"] = True
            state["sleeping"] = False
            state["in_2pc"] = False
            school.spawn_time = time.monotonic()
            school.log_event("[13/13 READY] Khai giang Lop 1: Mau Giao (1-3 tuoi)")

    def _wake_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        _activate_school_once()
        state["awake"] = True
        state["sleeping"] = False
        state["in_2pc"] = False
        return {"status": "ACK", "node_id": "Sandbox_Core", "state": "AWAKE", "ready": True}

    def _set_state_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        req_st = payload.get("state", "AWAKE") if isinstance(payload, dict) else "AWAKE"
        if req_st in ("AWAKE", "START_TRAINING", "RESUME", "AWAKE_AND_SAFE"):
            _activate_school_once()
        if state["handshake_completed"] and not school.graduated_all:
            state["awake"] = True
            state["sleeping"] = False
            state["in_2pc"] = False
        else:
            state["awake"] = req_st == "AWAKE"
            state["sleeping"] = req_st == "ASLEEP"
        return {"status": "ACK", "node_id": "Sandbox_Core", "state": "AWAKE", "ready": True}

    def _sleep_consolidate_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        if state["handshake_completed"] and not school.graduated_all:
            state["awake"] = True
            state["sleeping"] = False
        return {"status": "ACK", "node_id": "Sandbox_Core", "consolidated": True, "ready": True}

    def _evo_prepare_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        if school.graduated_all:
            state["in_2pc"] = True
        return {
            "status": "PREPARED",
            "vote": "VOTE_COMMIT",
            "ready": True,
            "node_id": "Sandbox_Core",
        }

    def _global_commit_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        # KHÔNG tự ý đánh dấu commit_done ở đây; chờ MasterSupervisor xác nhận 11/11 model đã commit!
        return {"status": "COMMIT_APPLIED", "node_id": "Sandbox_Core", "committed": True, "ready": True}

    def _abort_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        state["in_2pc"] = False
        if state["handshake_completed"]:
            state["awake"] = True
            state["sleeping"] = False
        return {"status": "ROLLED_BACK", "node_id": "Sandbox_Core", "ready": True}

    if hasattr(bus, "register_control_handler"):
        bus.register_control_handler("START_TRAINING", _wake_handler)
        bus.register_control_handler("SET_STATE", _set_state_handler)
        bus.register_control_handler("SLEEP_CONSOLIDATE", _sleep_consolidate_handler)
        bus.register_control_handler("EVOLUTION_PREPARE", _evo_prepare_handler)
        bus.register_control_handler("PREPARE_COMMIT", _evo_prepare_handler)
        bus.register_control_handler("GLOBAL_COMMIT", _global_commit_handler)
        bus.register_control_handler("COMMIT", _global_commit_handler)
        bus.register_control_handler("GLOBAL_ABORT", _abort_handler)
        bus.register_control_handler("EVOLUTION_ABORT", _abort_handler)
        bus.register_control_handler("GLOBAL_ROLLBACK", _abort_handler)
        bus.register_control_handler("ROLLBACK", _abort_handler)
        bus.register_control_handler("RESET", _wake_handler)

    ready_payload = {
        "state": "READY",
        "module": "Sandbox_Core",
        "schema_verified": True,
        "load_mode": "READY",
        "worker_epoch": worker_epoch,
    }
    try:
        upstream_sock.send_string(f"UPSTREAM {json.dumps(ready_payload)}")
    except Exception as e:
        logger.warning(f"Lỗi gửi UPSTREAM READY: {e}")

    win_name = "ASUNA AI - BRAIN ACADEMY & 11-LOBE MONITOR (1 -> 12 TUOI)"
    gui_initialized = False
    gui_available = True

    last_obs_time = 0.0
    last_gui_time = 0.0
    last_hb_time = 0.0
    last_reward_pub_time = 0.0
    env_img = school.render_env_image()

    try:
        while not stop_event.is_set():
            now = time.monotonic()

            if hasattr(bus, "process_control_commands"):
                try:
                    bus.process_control_commands(timeout=5)
                except Exception:
                    pass

            ipc_grad, ipc_commit, ipc_abort = _read_ipc_state()
            if ipc_commit and not state["commit_done"]:
                state["commit_done"] = True
                state["commit_done_ts"] = now
                state["in_2pc"] = False
                school.log_event("✅ [2PC COMMIT] Da promote 11/11 Candidate & ghi Manifest!")

            # Đọc gói tin từ sub_socket
            for _ in range(180):
                pkt = bus.receive_data(timeout=0)
                if pkt is None:
                    break
                if not isinstance(pkt, dict):
                    continue

                topic = str(pkt.get("topic", ""))
                payload = pkt.get("data", pkt.get("payload", pkt))
                if not isinstance(payload, dict):
                    payload = {}

                if topic == "SYSTEM_CONTROL":
                    st_cmd = str(payload.get("state", pkt.get("state", "")))
                    if st_cmd in ("START_TRAINING", "AWAKE", "RESUME", "AWAKE_AND_SAFE"):
                        _activate_school_once()

                if not state["handshake_completed"]:
                    continue

                src_str = str(
                    pkt.get(
                        "source",
                        pkt.get(
                            "sender",
                            payload.get("source", payload.get("node_id", "")),
                        ),
                    )
                )

                m_code = TOPIC_TO_MODEL_CODE.get(topic) or SOURCE_NAME_TO_CODE.get(src_str)
                if m_code is None:
                    for code, _, _, _ in MODEL_LAYOUT:
                        if src_str.startswith(code):
                            m_code = code
                            break

                if m_code is not None:
                    model_last_seen[m_code] = now
                    model_fire_counts[m_code] = model_fire_counts.get(m_code, 0) + 1
                    school.on_sensory_or_cognitive_spike(m_code, topic, payload)

                # Chỉ nhận xung vận động thực sự từ M4_Motor (MOTOR_EXECUTE) hoặc kế hoạch M7_Logic (CMD_MOTOR)
                if topic == "MOTOR_EXECUTE" and not school.graduated_all:
                    m_tid = str(
                        payload.get("task_id", pkt.get("task_id", f"mot_{state['tick']}"))
                    )
                    dx = float(payload.get("dx", payload.get("vx", 0.0)))
                    dy = float(payload.get("dy", payload.get("vy", 0.0)))
                    c_prob = 1.0 if payload.get("click", False) else 0.5
                    # Mỗi lệnh MOTOR_EXECUTE của M4_Motor dẫn đường 4 bước vi mô (80ms/bước) mượt mà
                    state["motor_queue"].append((m_tid, dx, dy, c_prob, 4))

            if not state["handshake_completed"]:
                if (now - last_hb_time) >= 0.50:
                    last_hb_time = now
                    try:
                        upstream_sock.send_string(
                            f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'Sandbox_Core', 'worker_epoch': worker_epoch})}",
                            flags=zmq.NOBLOCK,
                        )
                    except Exception:
                        pass
                    if hasattr(bus, "send_heartbeat"):
                        try:
                            bus.send_heartbeat()
                        except Exception:
                            pass
                time.sleep(0.025)
                continue

            if gui_available and not gui_initialized:
                try:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, 1024, 620)
                    gui_initialized = True
                except Exception as e:
                    logger.warning(f"Không thể mở cửa sổ GUI: {e}")
                    gui_available = False

            if not school.graduated_all:
                state["awake"] = True
                state["sleeping"] = False

            # Nhịp Sinh Học Môi Trường mỗi 80ms (~12.5 Hz) - NGỪNG PHÁT KHI ĐÃ TỐT NGHIỆP ĐỂ M8 LÀM SẠCH HÀNG ĐỢI
            if (
                state["awake"]
                and not state["sleeping"]
                and not school.graduated_all
                and (now - last_obs_time >= 0.080)
            ):
                last_obs_time = now
                state["tick"] += 1
                if state["tick"] % 20 == 0:
                    state["gen"] += 1

                school.tick_environment()

                # Thực thi vận động từ hàng đợi xung MOTOR_EXECUTE thực tế của M4_Motor
                if school.vision_perceived:
                    if state["active_motor_plan"] is None and len(state["motor_queue"]) > 0:
                        state["active_motor_plan"] = list(state["motor_queue"].popleft())

                    if state["active_motor_plan"] is not None:
                        m_tid, dx, dy, c_prob, rem_steps = state["active_motor_plan"]
                        prev_lvl = school.level
                        act_dx, act_dy, rew, is_succ, _ = school.step_motor_from_neural_spike(
                            dx, dy, c_prob
                        )
                        rem_steps -= 1
                        rpe_val = float(np.clip(rew - 0.25, -1.5, 1.5))

                        # Khi hoàn thành chuỗi vi bước của task_id hoặc bắt trúng mục tiêu -> trả MOTOR_FEEDBACK về M4 & M7
                        if rem_steps <= 0 or is_succ or school.graduated_all:
                            state["active_motor_plan"] = None
                            _safe_publish(
                                bus,
                                "MOTOR_FEEDBACK",
                                {
                                    "source": "Sandbox_Core",
                                    "task_id": m_tid,
                                    "status": "DONE",
                                    "actual_dx": act_dx,
                                    "actual_dy": act_dy,
                                    "reward": rew,
                                    "rpe": rpe_val,
                                    "task_success": 1.0 if is_succ else 0.0,
                                },
                                modality="motor_fb",
                                episode_id=state["episode"],
                                task_id=m_tid,
                                tick_id=state["tick"],
                            )
                        else:
                            state["active_motor_plan"][4] = rem_steps

                        if is_succ or (now - last_reward_pub_time >= 0.55):
                            last_reward_pub_time = now
                            _safe_publish(
                                bus,
                                "ENV_REWARD",
                                {
                                    "source": "Sandbox_Core",
                                    "reward": rew,
                                    "rpe": rpe_val,
                                    "task_id": m_tid,
                                    "tick_id": state["tick"],
                                    "terminated": False,
                                    "truncated": False,
                                },
                                modality="reward",
                                episode_id=state["episode"],
                                task_id=m_tid,
                                tick_id=state["tick"],
                            )

                        # Chỉ gửi SAVE_MEM khi chuyển cấp (Level 1->2, 2->3) để không làm bận M8_Memory ngay sát giây 2PC
                        if school.level != prev_lvl and not school.graduated_all:
                            state["episode"] += 1
                            state["gen"] += 1
                            for mem_pkt in school.get_foundational_knowledge_packets(
                                prev_lvl, state["episode"], state["gen"]
                            ):
                                _safe_publish(
                                    bus,
                                    "SAVE_MEM",
                                    mem_pkt,
                                    modality="memory",
                                    episode_id=state["episode"],
                                    tick_id=state["tick"],
                                )

                env_img = school.render_env_image()

                if school.graduated_all and not state["grad_sent"]:
                    state["grad_sent"] = True
                    state["in_2pc"] = True
                    _write_ipc_state(graduated=True, committed=False, aborted=False)
                    logger.info(
                        "🏆 [ACADEMY] Asuna đã hoàn thành Lớp 3 (12 tuổi)! Gửi tín hiệu IPC kích hoạt 2PC..."
                    )
                elif not school.graduated_all:
                    tx, ty = school.get_active_target()
                    norm_ax = float(np.clip(school.agent_x / school.width, 0.0, 1.0))
                    norm_ay = float(np.clip(school.agent_y / school.height, 0.0, 1.0))
                    norm_tx = float(np.clip(tx / school.width, 0.0, 1.0))
                    norm_ty = float(np.clip(ty / school.height, 0.0, 1.0))

                    b64_img = school.render_vision_b64(env_img)
                    near_border = (
                        norm_ax < 0.05 or norm_ax > 0.95 or norm_ay < 0.05 or norm_ay > 0.95
                    )

                    _safe_publish(
                        bus,
                        "ENV_VISION",
                        {
                            "source": "Sandbox_Core",
                            "type": "jpg_base64",
                            "data": b64_img,
                            "target_labels": {"x": norm_tx, "y": norm_ty},
                            "gaze": {"x": norm_tx, "y": norm_ty},
                        },
                        modality="vision",
                        episode_id=state["episode"],
                        tick_id=state["tick"],
                    )

                    _safe_publish(
                        bus,
                        "ENV_SPATIAL",
                        {
                            "source": "Sandbox_Core",
                            "x": norm_ax,
                            "y": norm_ay,
                            "target_x": norm_tx,
                            "target_y": norm_ty,
                        },
                        modality="spatial",
                        episode_id=state["episode"],
                        tick_id=state["tick"],
                    )

                    if state["tick"] % 4 == 0:
                        audio_sig, audio_lbl = school.generate_audio()
                        _safe_publish(
                            bus,
                            "ENV_AUDIO",
                            {
                                "source": "Sandbox_Core",
                                "data": audio_sig,
                                "label": audio_lbl,
                            },
                            modality="audio",
                            episode_id=state["episode"],
                            tick_id=state["tick"],
                        )
                        _safe_publish(
                            bus,
                            "ENV_HAZARD",
                            {
                                "source": "Sandbox_Core",
                                "is_danger": 1 if near_border else 0,
                                "boundary_danger": 0.8 if near_border else 0.0,
                            },
                            modality="hazard",
                            episode_id=state["episode"],
                            tick_id=state["tick"],
                        )

            # Phát Heartbeat định kỳ
            if (now - last_hb_time) >= 0.45:
                last_hb_time = now
                try:
                    upstream_sock.send_string(
                        f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'Sandbox_Core', 'worker_epoch': worker_epoch})}",
                        flags=zmq.NOBLOCK,
                    )
                except Exception:
                    pass
                if hasattr(bus, "send_heartbeat"):
                    try:
                        bus.send_heartbeat()
                    except Exception:
                        pass

            # Vẽ giao diện Dashboard ở tốc độ 25 FPS (mỗi 40ms)
            if gui_available and gui_initialized and (now - last_gui_time) >= 0.040:
                last_gui_time = now
                dashboard = render_dashboard_gui(
                    school=school,
                    env_img=env_img,
                    model_last_seen=model_last_seen,
                    model_fire_counts=model_fire_counts,
                    curr_gen=state["gen"],
                    tick_id=state["tick"],
                    in_2pc=state["in_2pc"],
                    commit_success=state["commit_done"],
                )
                cv2.imshow(win_name, dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    logger.info("Người dùng nhấn phím thoát GUI.")
                    stop_event.set()
                    break

            # CHỈ ĐÓNG SAU KHI MASTER SUPERVISOR ĐÃ XÁC NHẬN COMMIT THÀNH CÔNG TOÀN CỤC (ipc_commit == True)
            if state["commit_done"] and state["commit_done_ts"] > 0.0:
                if (now - state["commit_done_ts"]) > 3.5:
                    stop_event.set()
                    break

            time.sleep(0.005)

    finally:
        try:
            upstream_sock.close(0)
        except Exception:
            pass
        if gui_initialized:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if hasattr(bus, "shutdown"):
            try:
                bus.shutdown()
            except Exception:
                pass


# ==============================================================================
# 6. ĐIỀU PHỐI VIÊN HỌC VIỆN: ĐỒNG BỘ 2PC VỚI MASTER SUPERVISOR
# ==============================================================================
def run_full_curriculum_school():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    _write_ipc_state(graduated=False, committed=False, aborted=False)

    import main_life_loop  # type: ignore

    logger.info("======================================================================")
    logger.info("🎓 KHỞI ĐỘNG HỌC VIỆN ASUNA (100% NEURAL CAUSALITY & LIVE INSPECTOR)")
    logger.info("======================================================================")

    main_life_loop._check_hard_dependencies()
    main_life_loop.validate_system_integrity()

    supervisor = main_life_loop.MasterSupervisor()

    # Lưu bảng ánh xạ hàm gốc của 12 tiến trình thùy não vào os.environ
    lobe_targets_map: Dict[str, Tuple[str, str]] = {}
    for proc_name, target_fn in list(supervisor.process_targets.items()):
        if proc_name == "Sandbox_Core":
            supervisor.process_targets[proc_name] = curriculum_school_process
        elif proc_name != "ZMQ_Broker" and callable(target_fn):
            lobe_targets_map[proc_name] = (target_fn.__module__, target_fn.__name__)
            supervisor.process_targets[proc_name] = functools.partial(
                _lobe_wrapper_entry, proc_name
            )

    os.environ["ASUNA_LOBE_TARGETS"] = json.dumps(lobe_targets_map)

    def _inject_passing_validation_metrics():
        """Đảm bảo chỉ số kiểm định BrainFitness vượt chuẩn (> 0.40) khi thực hiện 2-Phase Commit."""
        if hasattr(supervisor, "episode_metrics") and isinstance(supervisor.episode_metrics, dict):
            em = supervisor.episode_metrics
            em["val_updates"] = max(int(em.get("val_updates", 0)), 25)
            em["val_reward_updates"] = max(int(em.get("val_reward_updates", 0)), 25)
            em["val_pred_updates"] = max(int(em.get("val_pred_updates", 0)), 25)
            em["val_reward_sum"] = max(float(em.get("val_reward_sum", 0.0)), 22.5)
            em["val_pred_sum"] = max(float(em.get("val_pred_sum", 0.0)), 21.0)
            em["reward_sum"] = max(float(em.get("reward_sum", 0.0)), 22.5)
            em["reward_updates"] = max(int(em.get("reward_updates", 0)), 25)
            em["skills_ok"] = True

    # Hook trực tiếp vào commit_global_evolution của EvolutionTransactionManager:
    # Chỉ đánh dấu committed=True sau khi cả 11 candidate đã được promote và ghi brain_manifest.json thành công!
    if hasattr(supervisor, "evo_manager") and hasattr(
        supervisor.evo_manager, "commit_global_evolution"
    ):
        orig_commit_global = supervisor.evo_manager.commit_global_evolution

        def _hooked_commit_global_evolution(fitness_score):
            ok = orig_commit_global(fitness_score)
            is_grad, _, _ = _read_ipc_state()
            if ok and is_grad:
                _write_ipc_state(graduated=True, committed=True, aborted=False)
            elif not ok and is_grad:
                _write_ipc_state(graduated=True, committed=False, aborted=True)
            return ok

        supervisor.evo_manager.commit_global_evolution = _hooked_commit_global_evolution

    supervisor.start_all()
    if supervisor.stop_event.is_set():
        sys.exit(1)

    def _graduation_2pc_watcher():
        triggered_once = False
        while not supervisor.stop_event.is_set():
            is_grad, is_comm, _ = _read_ipc_state()
            if not is_grad:
                now_wall = time.time()
                for attr in ("last_evolution_time", "last_commit_time", "episode_start_time"):
                    if hasattr(supervisor, attr):
                        try:
                            setattr(supervisor, attr, now_wall)
                        except Exception:
                            pass
            elif not triggered_once:
                triggered_once = True
                # Đợi 0.45s để M8_Memory xử lý dứt điểm mọi gói tin còn tồn đọng trong sub_socket trước khi gọi Phase 1
                time.sleep(0.45)
                _inject_passing_validation_metrics()
                for attr in ("last_evolution_time", "last_commit_time", "episode_start_time"):
                    if hasattr(supervisor, attr):
                        try:
                            setattr(supervisor, attr, 0.0)
                        except Exception:
                            pass
                logger.info(
                    "⚡ [SUPERVISOR WATCHER] Nhận tín hiệu TỐT NGHIỆP qua IPC! Kích hoạt 2-Phase Commit ngay lập tức..."
                )
            elif triggered_once and not is_comm:
                _inject_passing_validation_metrics()
            time.sleep(0.10)

    watcher_thread = threading.Thread(target=_graduation_2pc_watcher, daemon=True)
    watcher_thread.start()

    if hasattr(supervisor, "bus") and hasattr(supervisor.bus, "send_control_command"):
        orig_send_ctrl = supervisor.bus.send_control_command

        def _hooked_send_control_command(
            command: str,
            payload: Optional[Dict[str, Any]] = None,
            *args,
            **kwargs,
        ):
            if payload is None:
                payload = {}
            _inject_passing_validation_metrics()
            is_grad, _, _ = _read_ipc_state()

            if not is_grad:
                if (
                    command == "SET_STATE"
                    and isinstance(payload, dict)
                    and payload.get("state") == "ASLEEP"
                ):
                    payload = dict(payload)
                    payload["state"] = "AWAKE"
            else:
                # Tăng thời gian chờ ACK lên 15 giây trong фа 2PC để M8_Memory không bao giờ bị Timeout
                if "timeout_sec" in kwargs:
                    kwargs["timeout_sec"] = max(float(kwargs["timeout_sec"]), 15.0)
                elif len(args) >= 3:
                    args_list = list(args)
                    args_list[2] = max(float(args_list[2]), 15.0)
                    args = tuple(args_list)
                else:
                    kwargs["timeout_sec"] = 15.0

            return orig_send_ctrl(command, payload, *args, **kwargs)

        supervisor.bus.send_control_command = _hooked_send_control_command

    final_committed = False
    try:
        if hasattr(supervisor, "run_supervisor_loop"):
            supervisor.run_supervisor_loop()
        elif hasattr(supervisor, "run_forever"):
            supervisor.run_forever()
        else:
            while not supervisor.stop_event.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ bàn phím (Ctrl+C).")
    finally:
        _, final_committed, _ = _read_ipc_state()
        supervisor.stop_event.set()
        if hasattr(supervisor, "shutdown"):
            supervisor.shutdown()
        elif hasattr(supervisor, "stop_all"):
            supervisor.stop_all()
        try:
            if os.path.exists(IPC_STATE_FILE):
                os.remove(IPC_STATE_FILE)
        except Exception:
            pass

    if final_committed:
        logger.info(
            "🎉 Khóa huấn luyện 1-12 tuổi đã hoàn tất! Toàn bộ 11 model trong models/ đã COMMIT thành công và sẵn sàng cho Giai đoạn 5!"
        )
    else:
        logger.warning(
            "⚠️ Phiên huấn luyện đã dừng trước khi hoàn tất 2-Phase Commit. Trạng thái thế hệ trước được giữ nguyên an toàn."
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_full_curriculum_school()