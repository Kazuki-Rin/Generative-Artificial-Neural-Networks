#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import subprocess
import time
import zmq
import sys
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import signal
import json
import base64
import copy
import hashlib
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from core_bus import CoreBus, run_central_broker

from replay_buffer import ReplayBuffer
import logging
import random
import uuid
import threading
from shared_utils import CheckpointManager, CheckpointError, MODEL_SPECS, tensor_state_hash

try:
    import numpy as np
    import cv2
except ImportError as e:
    raise RuntimeError(f"🚨 HỆ THỐNG TỪ CHỐI KHỞI ĐỘNG: Thiếu thư viện Cốt lõi. Chi tiết: {e}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
_original_dumps = json.dumps
def _patched_dumps(*args, **kwargs):
    kwargs['cls'] = NumpyEncoder
    return _original_dumps(*args, **kwargs)
json.dumps = _patched_dumps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, "03_Cognitive_Core"))
sys.path.append(os.path.join(BASE_DIR, "00_Central_Router", "M5_Amygdala"))
try:
    import importlib
    ProgressiveLiquidBrain = importlib.import_module("liquid_progressive_brain").ProgressiveLiquidBrain
    SproutingSNN = importlib.import_module("train_m5_snn").SproutingSNN
except Exception:
    pass

NEAT_AVAILABLE = False
try:
    import neat
    NEAT_AVAILABLE = True
except ImportError:
    neat = None

CHROMA_AVAILABLE = False
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None

TORCHDIFFEQ_AVAILABLE = False
try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    odeint = None

HAS_AUDIO = False
try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    sd = None

DEVICE_CPU = torch.device('cpu')
torch.set_num_threads(1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(processName)-15s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# =========================================================================
# [PHASE 4] ENCODER POLICY (VÁ LỖI P0-6 TỪ AUDIT: BẢO VỆ DỮ LIỆU TỪ DICT)
# =========================================================================
def categorical_encoder(label, dim, num_classes=10):
    vec = [0.0] * dim
    if isinstance(label, (int, np.integer)): idx = label % dim; vec[idx] = 1.0
    elif isinstance(label, str):
        h = int(hashlib.md5(label.encode('utf-8')).hexdigest()[:4], 16)
        vec[h % dim] = 1.0
    return vec

def flatten_numeric(data):
    if isinstance(data, dict):
        out = []
        for key in sorted(data.keys()):
            out.extend(flatten_numeric(data[key]))
        return out
    if isinstance(data, (list, tuple, np.ndarray)):
        out = []
        for item in data:
            out.extend(flatten_numeric(item))
        return out
    if torch.is_tensor(data):
        out = []
        for item in data.detach().cpu().flatten().tolist():
            out.extend(flatten_numeric(item))
        return out
    if isinstance(data, (int, float, np.number, bool)):
        return [float(data)]
    if isinstance(data, str):
        return categorical_encoder(data, 8)
    return []

def numeric_encoder(data, dim):
    if data is None:
        return [0.0] * dim
    flat = np.array(flatten_numeric(data), dtype=np.float32)
    if flat.size == 0: return [0.0] * dim
    if flat.size == dim: return flat.tolist()
    
    # [BẢN VÁ P1]: Dùng nội suy tuyến tính để resize tensor 1D thay vì lấy mẫu rời rạc gây mất thông tin
    x_old = np.linspace(0, 1, flat.size)
    x_new = np.linspace(0, 1, dim)
    sample = np.interp(x_new, x_old, flat)
    
    m = np.max(np.abs(sample))
    return (sample / m).tolist() if m > 0 else sample.tolist()

def embedding_encoder(data, dim):
    if isinstance(data, (str, dict)):
        return encode_semantic_hash(data, dim)
    if isinstance(data, (list, np.ndarray)):
        flat = np.array(data, dtype=np.float32).flatten()
        if flat.size == dim: return flat.tolist()
        if flat.size > dim: return flat[:dim].tolist()
        return flat.tolist() + [0.0] * (dim - flat.size)
    return [0.0] * dim

def fingerprint_hash(data, dim=39):
    text_str = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
    h = hashlib.sha256(text_str.encode('utf-8')).digest()
    vec = [(float(b) / 127.5) - 1.0 for b in h]
    return (vec * (dim // len(vec) + 1))[:dim]

def encode_semantic_hash(data, dim=39): return fingerprint_hash(data, dim)

# =========================================================================
# UTILITIES & ERROR HANDLING
# =========================================================================
class BrainErrorCategory:
    RECOVERABLE = "RECOVERABLE"; DEGRADED = "DEGRADED"; FATAL = "FATAL"; SCHEMA_MISMATCH = "SCHEMA_MISMATCH"

def diagnose_error(e, module_name):
    e_str = str(e).lower()
    if "out of memory" in e_str or "alloc" in e_str or "cuda error" in e_str or "segmentation" in e_str:
        logger.critical(f"💀 [FATAL] {module_name} SẬP NGUỒN DO TRÀN BỘ NHỚ: {e}"); return BrainErrorCategory.FATAL
    if "state_dict" in e_str or "pickle" in e_str or "schema" in e_str or "magic number" in e_str or "eof" in e_str:
        logger.error(f"🧩 [SCHEMA_MISMATCH] {module_name} DỮ LIỆU BỊ CORRUPT: {e}"); return BrainErrorCategory.SCHEMA_MISMATCH
    if "size mismatch" in e_str or "dimension" in e_str or "shape" in e_str or "mat1" in e_str or "zero" in e_str or "nan" in e_str:
        logger.warning(f"📉 [DEGRADED] {module_name} Đột biến dị tật hoặc lỗi đạo hàm. Hủy bỏ candidate: {e}"); return BrainErrorCategory.DEGRADED
    logger.error(f"⚠️ [SILENT_KILLER] {module_name} Lỗi tính toán ngầm: {e}"); return BrainErrorCategory.RECOVERABLE

def reseed_brain():
    seed = os.getpid() + int(time.time() * 1000) % 100000
    random.seed(seed); torch.manual_seed(seed)
    if np is not None: np.random.seed(seed)

def create_standard_packet(source, modality, payload, episode_id="UNKNOWN", tick_id=0, task_id="NO_TASK", parent_event_id=None):
    return {"schema_version": 1, "episode_id": episode_id, "tick_id": tick_id, "task_id": task_id,
            "event_id": f"evt_{uuid.uuid4().hex[:12]}", "parent_event_id": parent_event_id,
            "source": source, "modality": modality, "timestamp": time.time(), "payload": payload}

def extract_data(msg):
    if not msg: return "NO_PAYLOAD"
    parts = msg.split(" ", 1)
    if len(parts) > 1:
        try:
            parsed = json.loads(parts[1])
            if isinstance(parsed, dict):
                if "payload" in parsed and isinstance(parsed["payload"], dict):
                    for k, v in parsed["payload"].items():
                        if k not in parsed or (k == "task_id" and parsed.get("task_id") in (None, "NO_TASK")):
                            parsed[k] = v
                return parsed
            return {"payload": parsed}
        except json.JSONDecodeError: return parts[1]
    return "NO_PAYLOAD"

def get_all_msgs(sock, bus=None):
    msgs = []

    if bus and hasattr(bus, 'ctrl_socket') and sock == bus.ctrl_socket:
        while True:
            try:
                frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) >= 1:
                    msg_bytes = frames[-1]
                    try:
                        packet = json.loads(msg_bytes.decode('utf-8'))
                        cmd_type = packet.get("type")
                        if getattr(bus, "is_master", False):
                            identity = frames[0].decode('utf-8', errors='replace') if len(frames) >= 2 else ""
                            if cmd_type == "CONTROL" and packet.get("command") == "REPLAY_REQUEST" and hasattr(bus, "_handle_master_replay_request"):
                                bus._handle_master_replay_request(identity, packet)
                                continue
                            if cmd_type == "CONTROL_ACK" and packet.get("status") == "REPLAY_CACHE_RESULT":
                                det = packet.get("details", {}) if isinstance(packet.get("details"), dict) else {}
                                req_node = det.get("requester")
                                pkts = det.get("packets", [])
                                r_cmd_id = det.get("req_cmd_id", packet.get("command_id", ""))
                                r_tx_id = det.get("req_tx_id", packet.get("transaction_id", ""))
                                if req_node:
                                    if pkts:
                                        for p_item in pkts:
                                            if isinstance(p_item, dict) and p_item.get("source") and isinstance(p_item.get("topic_seq"), int):
                                                bus._store_in_replay_cache(p_item["source"], p_item.get("topic", "UNKNOWN"), p_item["topic_seq"], p_item)
                                        from core_bus import create_control_packet, create_ack_packet
                                        rep_pkt = create_control_packet("REPLAY_DATA", {"source": packet.get("source"), "packets": pkts}, f"rep_{r_cmd_id}", r_tx_id, bus.node_id, requires_ack=False, tick_id=bus.current_tick)
                                        bus._send_ctrl_msg([req_node.encode('utf-8'), b'', json.dumps(rep_pkt).encode('utf-8')])
                                    from core_bus import create_ack_packet
                                    ack_pkt = create_ack_packet(r_cmd_id, r_tx_id, bus.node_id, status="REPLAY_ACK", details={"replayed_count": len(pkts)})
                                    bus._send_ctrl_msg([req_node.encode('utf-8'), b'', json.dumps(ack_pkt).encode('utf-8')])
                                continue
                        else:
                            if cmd_type == "CONTROL":
                                cmd = packet.get("command", "")
                                if hasattr(bus, "_control_handlers") and cmd in bus._control_handlers:
                                    res = bus._control_handlers[cmd](packet.get("payload", {}))
                                    if packet.get("requires_ack", True):
                                        det = dict(res) if isinstance(res, dict) else {}
                                        st = det.pop("status", "OK") if isinstance(res, dict) else (res if isinstance(res, str) else "OK")
                                        bus.send_ack(packet.get("command_id", ""), packet.get("transaction_id", ""), st, det)
                                    continue

                        if cmd_type == "CONTROL":
                            cmd = packet.get("command", "")
                            packet["state"] = cmd
                        elif cmd_type == "CONTROL_ACK":
                            packet["state"] = packet.get("status", packet.get("details", {}).get("status", ""))
                            if "candidate_hash" in packet.get("details", {}):
                                packet["candidate_hash"] = packet["details"]["candidate_hash"]
                        else:
                            packet["state"] = packet.get("command", "")
                        
                        msgs.append(f"SYSTEM_CONTROL {json.dumps(packet)}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Lỗi JSON: {e}")
            except zmq.Again: 
                break
            except Exception as e:
                logger.warning(f"Lỗi ZMQ: {e}")
                break
    else:
        if bus and hasattr(bus, '_replayed_queue') and sock == getattr(bus, 'sub_socket', None):
            while len(bus._replayed_queue) > 0:
                pkt = bus._replayed_queue.popleft()
                msgs.append(f"{pkt.get('topic', 'UNKNOWN')} {json.dumps(pkt, ensure_ascii=False)}")
        while True:
            try:
                frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                if not frames: break
                if len(frames) == 1:
                    raw_str = frames[0].decode('utf-8', errors='replace')
                else:
                    topic = frames[0].decode('utf-8', errors='replace')
                    payload = frames[-1].decode('utf-8', errors='replace')
                    raw_str = f"{topic} {payload}"
                if bus and hasattr(bus, '_parse_data_msg') and sock == getattr(bus, 'sub_socket', None):
                    bus._parse_data_msg(raw_str)
                msgs.append(raw_str)
            except zmq.Again: 
                break
            except Exception as e:
                logger.warning(f"Bỏ qua gói tin lỗi: {e}")
                break
    return msgs

def encode_spatial_fourier(x, y, dim=10):
    feats = []
    for i in range(dim // 2):
        feats.append(float(np.sin(x * np.pi * (2**i))))
        feats.append(float(np.cos(y * np.pi * (2**i))))
    while len(feats) < dim: feats.append(0.0)
    return feats[:dim]

def encode_audio_stft(audio_array, dim=20):
    if audio_array is None or len(audio_array) == 0: return [0.0] * dim
    
    if len(audio_array) < dim * 2:
        pad_size = dim * 2 - len(audio_array)
        audio_array = list(audio_array) + [0.0] * pad_size
        
    tensor_audio = torch.tensor(audio_array, dtype=torch.float32)
    fft_result = torch.abs(torch.fft.rfft(tensor_audio))
    bin_size = max(1, len(fft_result) // dim)
    
    pooled = []
    for i in range(dim):
        slice_tensor = fft_result[i*bin_size:(i+1)*bin_size]
        pooled.append(float(torch.mean(slice_tensor)) if len(slice_tensor) > 0 else 0.0)
        
    m = max(pooled) if max(pooled) > 0 else 1.0
    return [p / m for p in pooled][:dim]

def encode_vision_onehot(label, dim=10): return categorical_encoder(label, dim)

def compute_local_plasticity(module_name, payload):
    rpe, arousal, threat, novelty, salience = [float(payload.get(k, 0.0)) for k in ["rpe", "arousal", "threat", "novelty", "salience"]]
    credit_factor = float(payload.get("credit", {}).get(module_name, 1.0))
    receptors = {
        "M1a_Dorsal":  {"rpe": 0.5, "salience": 1.0}, "M1b_Ventral": {"rpe": 0.5, "salience": 1.0, "novelty": 0.3},
        "M2_Ear":      {"rpe": 0.5, "salience": 0.8}, "M3_Spatial":  {"rpe": 0.6, "threat": 0.6, "salience": 0.4},
        "M4_Motor":    {"rpe": 1.0, "arousal": 0.8, "threat": 0.7},   "M5_Amygdala": {"threat": 1.5, "arousal": 0.9, "rpe": 0.1},  
        "M6_Language": {"rpe": 1.0, "novelty": 0.5},                  "M7_Logic":    {"rpe": 1.0, "novelty": 0.5, "threat": 0.4},
        "M8_Memory":   {"novelty": 1.0, "rpe": 0.8, "salience": 0.6}, "M9_Thalamus": {"salience": 1.5, "arousal": 0.5},            
        "M10_Vocal":   {"rpe": 1.0, "arousal": 0.3}                  
    }
    w = receptors.get(module_name, {"rpe": 1.0})
    magnitude = (abs(rpe)*w.get("rpe",0.0) + threat*w.get("threat",0.0) + novelty*w.get("novelty",0.0) + salience*w.get("salience",0.0) + arousal*w.get("arousal",0.0))
    final_plasticity = float(magnitude * credit_factor * (1.0 if rpe >= 0 else -1.0))
    if module_name == "M5_Amygdala": return float(magnitude * credit_factor * (1.0 if threat > 0.5 else -1.0))
    return final_plasticity

def _load_resource_governor():
    arch_path = os.path.join(BASE_DIR, "architecture_manifest.json")
    default_gov = {
        "global_max_parameters": 500000,
        "max_params_per_model": 150000,
        "max_memory_mb": 4096,
        "max_process_rss_mb": 2048,
        "max_nodes_per_module": {},
        "max_parameters_per_module": {}
    }
    if os.path.exists(arch_path):
        try:
            with open(arch_path, "r", encoding="utf-8") as f:
                arch = json.load(f)
            gov = arch.get("system_topology", {}).get("resource_governor", arch.get("governor", {}))
            if isinstance(gov, dict):
                default_gov.update(gov)
        except Exception:
            pass
    return default_gov

RESOURCE_GOVERNOR = _load_resource_governor()
MAX_PARAMS_BUDGET = int(RESOURCE_GOVERNOR.get("global_max_parameters", 500000))

def topology_budget_exceeded(model, module_name: str = None):
    gov = _load_resource_governor()
    raw_name = str(module_name or getattr(model, "name", model.__class__.__name__))
    class_map = {
        "EvolvingSNN": "M2_Ear",
        "SproutingSNN": "M5_Amygdala",
        "DynamicSOM": "M9_Thalamus",
        "VocalTractODE": "M10_Vocal",
    }
    if "M6" in raw_name: mod_key = "M6_Language"
    elif "M7" in raw_name: mod_key = "M7_Logic"
    else: mod_key = class_map.get(raw_name, raw_name)

    per_mod_budgets = gov.get("max_parameters_per_module", {})
    model_budget = int(per_mod_budgets.get(mod_key, gov.get("max_params_per_model", MAX_PARAMS_BUDGET)))
    global_budget = int(gov.get("global_max_parameters", MAX_PARAMS_BUDGET))
    total_params = sum(p.numel() for p in model.parameters() if getattr(p, 'requires_grad', False))
    if total_params > model_budget or total_params > global_budget:
        logger.warning(f"⚠️ [BUDGET LIMIT] {mod_key} vượt ngân sách tham số Governor ({total_params} > {min(model_budget, global_budget)}). TỪ CHỐI TIẾN HÓA CẤU TRÚC!")
        return True

    node_caps = gov.get("max_nodes_per_module", {})
    if mod_key in node_caps:
        max_n = int(node_caps[mod_key])
        curr_n = 0
        if hasattr(model, "columns"): curr_n = len(model.columns)
        elif hasattr(model, "num_nodes"): curr_n = int(model.num_nodes)
        elif hasattr(model, "fc1") and hasattr(model.fc1, "out_features"): curr_n = int(model.fc1.out_features)
        elif hasattr(model, "num_hidden"): curr_n = int(model.num_hidden)
        if curr_n > max_n:
            logger.warning(f"⚠️ [BUDGET LIMIT] {mod_key} vượt giới hạn node/cột Governor ({curr_n} > {max_n}). TỪ CHỐI TIẾN HÓA CẤU TRÚC!")
            return True

    try:
        max_rss = float(gov.get("max_process_rss_mb", 2048))
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        if rss_mb >= max_rss:
            logger.warning(f"⚠️ [BUDGET LIMIT] {mod_key} vượt giới hạn RAM tiến trình ({rss_mb:.1f}MB >= {max_rss}MB)!")
            return True
    except Exception:
        pass
    return False

import inspect
import psutil

class WorkerLoadMode:
    READY = "READY"
    DEGRADED = "DEGRADED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

def load_worker_checkpoint_strict(model_id, is_torch=True, allow_bootstrap=False):
    """
    [BẢN VÁ C]: Thống nhất chế độ tải checkpoint (READY, DEGRADED, RECOVERY_REQUIRED).
    Mặc định cấm tự ý khởi tạo lại im lặng khi lỗi hoặc thiếu file.
    """
    epoch_file = os.path.join(BASE_DIR, "models", model_id, ".worker_epoch")
    if os.path.exists(epoch_file):
        try:
            with open(epoch_file, "r", encoding="utf-8") as ef:
                boot_epoch = int(ef.read().strip())
            os.environ["ASUNA_WORKER_EPOCH"] = str(boot_epoch)
            os.environ["ASUNA_WORKER_INSTANCE_ID"] = f"{model_id}_ep{boot_epoch}_{os.getpid()}"
        except Exception:
            pass

    try:
        chkpt = CheckpointManager.load(model_id, is_torch=is_torch, validate_hash=True, enforce_manifest=True)
    except CheckpointError as e:
        logger.critical(f"🚨 [RECOVERY_REQUIRED] {model_id}: Checkpoint lỗi toàn vẹn ({e}). Dừng worker!")
        raise RuntimeError(f"RECOVERY_REQUIRED:{model_id}:{e}")

    if chkpt is not None:
        return chkpt, WorkerLoadMode.READY

    bootstrap_allowed = allow_bootstrap or (os.environ.get("ASUNA_ALLOW_BOOTSTRAP", "0") == "1")
    if not bootstrap_allowed:
        logger.critical(f"🚨 [RECOVERY_REQUIRED] {model_id}: Thiếu checkpoint chuẩn. Cấm tự ý khởi tạo im lặng!")
        raise RuntimeError(f"RECOVERY_REQUIRED:{model_id}:Missing canonical checkpoint")

    logger.warning(f"🆕 [BOOTSTRAP_NEW_MODEL] {model_id}: Khởi tạo MODEL MỚI (Chế độ DEGRADED).")
    return None, WorkerLoadMode.DEGRADED

def can_worker_write_canonical(model_id, staged_tx_id=None):
    """Chặn worker thế hệ cũ (zombie) hoặc worker đang giữ giao dịch treo ghi đè checkpoint chuẩn."""
    if staged_tx_id is not None:
        return False
    my_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    epoch_file = os.path.join(BASE_DIR, "models", model_id, ".worker_epoch")
    if os.path.exists(epoch_file):
        try:
            with open(epoch_file, "r", encoding="utf-8") as f:
                active_epoch = int(f.read().strip())
            if my_epoch < active_epoch:
                logger.error(f"🛑 [FENCE BLOCKED] {model_id}: Worker epoch {my_epoch} < active epoch {active_epoch}. Từ chối ghi checkpoint!")
                return False
        except Exception:
            pass
    return True

def stage_pytorch_candidate_file(model_id, live_state, staged_state, optimizer, staged_opt, tx_id, current_gen, target_gen, fitness, topology_version, topology_dict):
    """
    [BẢN VÁ A]: Lưu ứng viên thành bản bất biến riêng (.candidate_<tx_id>), gắn transaction_id, parent_state_hash và target_generation.
    Tuyệt đối không ghi đè checkpoint chuẩn trong giai đoạn PREPARE.
    """
    raw_live_sd = live_state.state_dict() if hasattr(live_state, "state_dict") else live_state
    if isinstance(raw_live_sd, dict):
        raw_live_sd = {k: v for k, v in raw_live_sd.items() if not k.endswith('.mem') and not k.endswith('.syn') and not k.endswith('.spk')}
    parent_hash = tensor_state_hash(raw_live_sd)
    has_candidate = staged_state is not None and staged_state is not live_state
    cand_model = staged_state if has_candidate else live_state
    cand_opt = staged_opt if (has_candidate and staged_opt is not None) else optimizer
    raw_cand_sd = cand_model.state_dict() if hasattr(cand_model, "state_dict") else cand_model
    if isinstance(raw_cand_sd, dict):
        raw_cand_sd = {k: v for k, v in raw_cand_sd.items() if not k.endswith('.mem') and not k.endswith('.syn') and not k.endswith('.spk')}
    raw_hash = tensor_state_hash(raw_cand_sd)
    cand_hash_str = raw_hash if has_candidate else f"UNCHANGED_{raw_hash}"
    extra = {
        "is_candidate": True,
        "phase": "PREPARED",
        "transaction_id": tx_id,
        "candidate_transaction_id": tx_id,
        "parent_state_hash": parent_hash,
        "candidate_hash": cand_hash_str,
        "generation": target_gen,
        "parent_generation": current_gen,
        "fitness": fitness,
        "topology_version": topology_version,
        "topology": topology_dict,
        "worker_epoch": int(os.environ.get("ASUNA_WORKER_EPOCH", "1")),
        "worker_instance_id": os.environ.get("ASUNA_WORKER_INSTANCE_ID", f"{model_id}_local")
    }
    ok = CheckpointManager.save(model_id, cand_model, cand_opt, is_torch=True, extra_states=extra)
    if not ok:
        raise CheckpointError(f"Không thể lưu file candidate cho {model_id} (tx={tx_id})")
    return cand_hash_str

class UniversalEvolutionProtocol:
    _validation_cache = {}

    @staticmethod
    def evolve_and_stage_pytorch(live_state, eligibility_trace, compute_loss_func, optimizer, rpe=0.0, novelty=0.0, mutate_func=None):
        try:
            proc = psutil.Process(os.getpid())
            rss_before_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            trace_for_loss = eligibility_trace.detach() if torch.is_tensor(eligibility_trace) else eligibility_trace

            model_key = getattr(live_state, "name", live_state.__class__.__name__)
            train_trace = trace_for_loss
            cached_val = UniversalEvolutionProtocol._validation_cache.get(model_key)
            if torch.is_tensor(trace_for_loss):
                if torch.is_tensor(cached_val) and cached_val.shape == trace_for_loss.shape:
                    val_trace = cached_val
                else:
                    val_trace = trace_for_loss.detach().clone()
                UniversalEvolutionProtocol._validation_cache[model_key] = (
                    0.7 * val_trace + 0.3 * trace_for_loss.detach().clone()
                )
            else:
                val_trace = trace_for_loss

            live_state.eval()
            t0_live = time.perf_counter()
            with torch.no_grad():
                live_train_loss = float(compute_loss_func(live_state, train_trace).item())
                live_val_loss = float(compute_loss_func(live_state, val_trace).item())
            live_infer_ms = (time.perf_counter() - t0_live) * 1000.0
            live_state.train()
            
            candidate_state = copy.deepcopy(live_state)
            old_params = sum(p.numel() for p in candidate_state.parameters())
            cand_opt = type(optimizer)(filter(lambda p: p.requires_grad, candidate_state.parameters()), **optimizer.defaults)
            try:
                cand_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            except Exception as e:
                logger.error(f"Lỗi sao chép optimizer ban đầu: {e}")
            
            should_mutate = mutate_func is not None and (rpe < -0.3 or novelty > 0.7)
            if should_mutate:
                if not topology_budget_exceeded(candidate_state):
                    try:
                        sig = inspect.signature(mutate_func)
                        if len(sig.parameters) >= 2:
                            mutate_func(candidate_state, cand_opt)
                        else:
                            mutate_func(candidate_state)
                    except Exception:
                        mutate_func(candidate_state)
                    if topology_budget_exceeded(candidate_state):
                        candidate_state = copy.deepcopy(live_state)
                        cand_opt = type(optimizer)(filter(lambda p: p.requires_grad, candidate_state.parameters()), **optimizer.defaults)
                        try: cand_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                        except Exception: pass
                
            new_params = sum(p.numel() for p in candidate_state.parameters())
            topology_mutated = (old_params != new_params)
            if topology_mutated:
                opt_param_ids = {id(p) for g in cand_opt.param_groups for p in g.get("params", [])}
                cand_param_ids = {id(p) for p in candidate_state.parameters() if p.requires_grad}
                if opt_param_ids != cand_param_ids:
                    cand_opt = type(optimizer)(filter(lambda p: p.requires_grad, candidate_state.parameters()), **optimizer.defaults)
                logger.info(f"🧬 [PHASE 5] Đột biến mở rộng ({old_params} -> {new_params} params). Đã đồng bộ Optimizer!")
                
            cand_loss_tensor = compute_loss_func(candidate_state, train_trace)
            hebbian_penalty = sum(torch.sum(param ** 2) for param in candidate_state.parameters() if param.requires_grad)
            total_loss = cand_loss_tensor * max(0.1, abs(rpe)) + 0.001 * hebbian_penalty
            
            cand_opt.zero_grad(); total_loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate_state.parameters(), 1.0); cand_opt.step()
            candidate_state.eval()
            
            t0_cand = time.perf_counter()
            with torch.no_grad():
                val_loss = float(compute_loss_func(candidate_state, val_trace).item())
            cand_infer_ms = (time.perf_counter() - t0_cand) * 1000.0
            rss_after_mb = proc.memory_info().rss / (1024.0 * 1024.0)

            if np.isnan(val_loss) or np.isinf(val_loss):
                logger.info(f"❌ [EVO REJECT] {model_key}: val_loss NaN/Inf.")
                return False, live_val_loss, live_state, optimizer
            
            latency_delta_ms = max(0.0, cand_infer_ms - live_infer_ms)
            ram_delta_mb = max(0.0, rss_after_mb - rss_before_mb)
            real_latency_penalty = min(0.2, latency_delta_ms / 100.0)
            real_memory_penalty = min(0.2, ram_delta_mb / 256.0)
            val_fitness_score = val_loss + real_latency_penalty + real_memory_penalty
            
            if val_fitness_score < live_val_loss:
                candidate_state.train()
                logger.info(
                    f"✅ [EVO ACCEPT] {model_key}: val_loss={val_loss:.4f} < live_val={live_val_loss:.4f} "
                    f"(train_loss={live_train_loss:.4f}, lat={cand_infer_ms:.2f}ms, ram_delta={ram_delta_mb:.2f}MB)"
                )
                return True, val_fitness_score, candidate_state, cand_opt
            
            logger.info(
                f"❌ [EVO REJECT] {model_key}: val_score={val_fitness_score:.4f} >= live_val={live_val_loss:.4f} "
                f"(val_loss={val_loss:.4f}, lat={cand_infer_ms:.2f}ms, ram_delta={ram_delta_mb:.2f}MB)"
            )
            return False, live_val_loss, live_state, optimizer
        except Exception as e:
            diagnose_error(e, "UniversalEvolution")
            return False, float('inf'), live_state, optimizer

# ================= NEAT BRAIN WITH ONLINE WEIGHT PERTURBATION =================
class DynamicNeatBrain:
    """
    [BẢN VÁ 15]: Runtime NEAT thực hiện tinh chỉnh trọng số trực tuyến (Online Weight Perturbation)
    trên cấu trúc mạng cố định từ checkpoint. Không tự tuyên bố tiến hóa cấu trúc (add/del node/conn) ở Runtime.
    """
    def __init__(self, model_id, config_path=None):
        self.model_id = model_id; self.config_path = config_path
        self.live_state = None; self.active_net = None; self.candidate_state = None
        self.eligibility_trace = {"inputs": None, "outputs": None, "targets": None, "decay_factor": 0.9}
        self.generation = 0; self.fitness = 0.0; self.topology_version = 1
        chkpt, self.load_mode = load_worker_checkpoint_strict(model_id, is_torch=False)
        if chkpt:
            if isinstance(chkpt, dict) and "model_state" in chkpt:
                self.live_state = chkpt.get("model_state")
                self.generation = chkpt.get("generation", 0)
                self.fitness = chkpt.get("fitness", 0.0)
                self.topology_version = chkpt.get("topology_version", chkpt.get("architecture_version", 1))
            else: 
                self.live_state = chkpt
        else:
            if self.config_path and os.path.exists(self.config_path) and NEAT_AVAILABLE:
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.config_path)
                pop = neat.Population(config); self.live_state = list(pop.population.values())[0]
        if self.live_state and NEAT_AVAILABLE:
            if isinstance(self.live_state, str):
                self.live_state = pickle.loads(bytes.fromhex(self.live_state))
                
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.config_path)
            self.active_net = neat.nn.RecurrentNetwork.create(self.live_state, config)
        self.parent_state = pickle.loads(pickle.dumps(self.live_state)) if self.live_state else None
        self.parent_net = pickle.loads(pickle.dumps(self.active_net)) if self.active_net else None
        self._evo_state = "LIVE"
        self._current_transaction_id = None
        self._candidate_topology_changed = False

    def prepare_candidate(self, transaction_id: str, target_generation: int = None) -> dict:
        if not self.live_state or not self.active_net:
            return {"status": "ERROR", "reason": "No live state"}
        
        if self._evo_state in ("CANDIDATE_STAGED", "STAGED_NOOP") and self._current_transaction_id == transaction_id and self.candidate_state is not None:
            raw_h = tensor_state_hash(self.candidate_state)
            cand_hash_str = f"UNCHANGED_{raw_h}" if self._evo_state == "STAGED_NOOP" else raw_h
            return {"status": "PREPARED", "candidate_hash": cand_hash_str, "transaction_id": transaction_id}

        if self._evo_state in ("CANDIDATE_STAGED", "STAGED_NOOP") and self._current_transaction_id is not None and self._current_transaction_id != transaction_id:
            logger.warning(f"[{self.model_id}] Giao dịch {self._current_transaction_id} đang chờ. Loại bỏ candidate cũ vì không khớp transaction.")
            self.candidate_state = None
            self.active_net_candidate = None
            self._evo_state = "LIVE"
        
        self._current_transaction_id = transaction_id
        self.parent_generation = self.generation
        self.parent_state = pickle.loads(pickle.dumps(self.live_state))
        self.parent_net = pickle.loads(pickle.dumps(self.active_net))
        parent_hash = tensor_state_hash(self.live_state)
        target_gen = target_generation if target_generation is not None else (self.generation + 1)
        
        if getattr(self, 'candidate_state', None) is None:
            self.candidate_state = pickle.loads(pickle.dumps(self.live_state))
            self.active_net_candidate = pickle.loads(pickle.dumps(self.active_net))
            self._evo_state = "STAGED_NOOP"
            cand_hash_str = f"UNCHANGED_{parent_hash}"
        else:
            self._evo_state = "CANDIDATE_STAGED"
            cand_hash_str = tensor_state_hash(self.candidate_state)

        mut_rec = getattr(self, 'last_mutation_record', "weight_perturbation")
        CheckpointManager.save(self.model_id, self.candidate_state, is_torch=False, extra_states={
            "is_candidate": True, "phase": "PREPARED", "transaction_id": transaction_id,
            "candidate_transaction_id": transaction_id, "parent_state_hash": parent_hash,
            "candidate_hash": cand_hash_str, "generation": target_gen, "parent_generation": self.generation,
            "fitness": self.fitness, "topology_version": self.topology_version, "mutation_record": mut_rec,
            "worker_epoch": int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
        })
        return {"status": "PREPARED", "candidate_hash": cand_hash_str, "transaction_id": transaction_id}

    def commit_candidate(self, transaction_id: str, target_generation: int = None) -> bool:
        if getattr(self, 'last_committed_tx_id', None) == transaction_id:
            return True
        if self._current_transaction_id != transaction_id:
            logger.error(f"[{self.model_id}] Commit Failed: Tx ID mismatch (Expected {self._current_transaction_id}, Got {transaction_id})")
            return False
        if self._evo_state not in ["CANDIDATE_STAGED", "STAGED_NOOP"] or self.candidate_state is None:
            return False
            
        self.last_committed_tx_id = transaction_id
        self.live_state = self.candidate_state
        if hasattr(self, 'active_net_candidate') and self.active_net_candidate is not None:
            self.active_net = self.active_net_candidate
        
        if target_generation is not None:
            self.generation = target_generation
            
        self.candidate_state = None
        self.active_net_candidate = None
        self._current_transaction_id = None
        self._evo_state = "LIVE"
        return True

    def rollback_candidate(self, transaction_id: str) -> bool:
        CheckpointManager.discard_candidate(self.model_id, transaction_id)
        if self._current_transaction_id != transaction_id and getattr(self, 'last_committed_tx_id', None) != transaction_id:
            return self._current_transaction_id is None
        if getattr(self, 'parent_state', None) is not None:
            self.live_state = pickle.loads(pickle.dumps(self.parent_state))
        if getattr(self, 'parent_net', None) is not None:
            self.active_net = pickle.loads(pickle.dumps(self.parent_net))
        if hasattr(self, 'parent_generation'):
            self.generation = self.parent_generation
        if getattr(self, 'last_committed_tx_id', None) == transaction_id:
            self.last_committed_tx_id = None
        self.candidate_state = None
        self.active_net_candidate = None
        self._current_transaction_id = None
        self._evo_state = "LIVE"
        return True
        
    def update_trace(self, inputs, outputs, targets):
        if not hasattr(self, '_trace_batch'): self._trace_batch = []
        self._trace_batch.append({"inputs": inputs, "outputs": outputs, "targets": targets})
        if len(self._trace_batch) > 32: self._trace_batch.pop(0)
        
        if getattr(self, 'eligibility_trace', None) is None or not isinstance(self.eligibility_trace, dict) or self.eligibility_trace.get("inputs") is None:
            self.eligibility_trace = {"inputs": np.array(inputs), "outputs": np.array(outputs), "targets": np.array(targets), "decay_factor": 0.9}
        else:
            df = self.eligibility_trace["decay_factor"]
            self.eligibility_trace["inputs"] = df * self.eligibility_trace["inputs"] + (1 - df) * np.array(inputs)
            self.eligibility_trace["outputs"] = df * self.eligibility_trace["outputs"] + (1 - df) * np.array(outputs)
            self.eligibility_trace["targets"] = df * self.eligibility_trace["targets"] + (1 - df) * np.array(targets)

    def activate(self, inputs):
        if not self.active_net:
            raise RuntimeError(f"🚨 [FATAL] Mạng NEAT tại {self.model_id} bị thiếu hoặc chưa khởi tạo. Yêu cầu phục hồi!")
        try:
            return self.active_net.activate(inputs)
        except Exception as e:
            logger.error(f"🚨 [FATAL] Lỗi kích hoạt NEAT tại {self.model_id}: {e}")
            raise RuntimeError(str(e))

    def evolve_and_stage(self, inputs, target, rpe=0.0):
        if not NEAT_AVAILABLE:
            raise RuntimeError(f"🚨 [FATAL] Thư viện NEAT bị thiếu, không thể tinh chỉnh trọng số tại {self.model_id}.")
        if not self.live_state or self._evo_state != "LIVE":
            return False
        try:
            batch = getattr(self, '_trace_batch', [])
            if len(batch) >= 4:
                mid = len(batch) // 2
                train_batch = batch[:mid]
                val_batch = batch[mid:]
            else:
                train_batch = [{"inputs": inputs, "targets": target}]
                val_batch = list(batch) if len(batch) > 0 else train_batch

            proc = psutil.Process(os.getpid())
            rss_before_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.config_path)
            base_genome = self.candidate_state if getattr(self, 'candidate_state', None) is not None else self.live_state
            live_eval_net = neat.nn.RecurrentNetwork.create(base_genome, config)
            
            t0_live = time.perf_counter()
            live_train_loss = sum(
                sum((float(n) - float(t))**2 for n, t in zip(live_eval_net.activate(s["inputs"]), s["targets"]))
                for s in train_batch
            ) / max(1, len(train_batch))
            live_val_loss = sum(
                sum((float(n) - float(t))**2 for n, t in zip(live_eval_net.activate(s["inputs"]), s["targets"]))
                for s in val_batch
            ) / max(1, len(val_batch))
            live_infer_ms = (time.perf_counter() - t0_live) * 1000.0
            
            best_candidate = None
            best_train_loss = float('inf')
            best_net = None
            
            for _ in range(5):
                cand_state = pickle.loads(pickle.dumps(base_genome))
                mutated = False
                for cg in cand_state.connections.values():
                    if random.random() < 0.8:
                        cg.weight += random.gauss(0, 0.5 * max(1.0, abs(rpe)))
                        mutated = True
                if not mutated:
                    continue
                    
                mutant_net = neat.nn.RecurrentNetwork.create(cand_state, config)
                cand_train_loss = sum(
                    sum((float(n) - float(t))**2 for n, t in zip(mutant_net.activate(s["inputs"]), s["targets"]))
                    for s in train_batch
                ) / max(1, len(train_batch))
                
                if cand_train_loss < best_train_loss:
                    best_train_loss = cand_train_loss
                    best_candidate = cand_state
                    best_net = mutant_net

            if best_candidate is None:
                return False

            t0_cand = time.perf_counter()
            cand_val_loss = sum(
                sum((float(n) - float(t))**2 for n, t in zip(best_net.activate(s["inputs"]), s["targets"]))
                for s in val_batch
            ) / max(1, len(val_batch))
            cand_infer_ms = (time.perf_counter() - t0_cand) * 1000.0
            rss_after_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            ram_delta_mb = max(0.0, rss_after_mb - rss_before_mb)
            real_penalty = min(0.2, max(0.0, cand_infer_ms - live_infer_ms) / 100.0) + min(0.2, ram_delta_mb / 256.0)
            total_val_score = cand_val_loss + real_penalty
                
            self.last_mutation_record = "weight_perturbation_only"
            self._candidate_topology_changed = False
            
            if total_val_score < live_val_loss:
                logger.info(
                    f"✅ [NEAT ACCEPT] {self.model_id}: val={cand_val_loss:.4f} < base_val={live_val_loss:.4f} "
                    f"(train={best_train_loss:.4f}/{live_train_loss:.4f}, lat={cand_infer_ms:.2f}ms, ram_delta={ram_delta_mb:.2f}MB)"
                )
                self.candidate_state = best_candidate
                self.active_net_candidate = best_net
                return True
            
            logger.info(
                f"❌ [NEAT REJECT] {self.model_id}: val_score={total_val_score:.4f} >= base_val={live_val_loss:.4f} "
                f"(lat={cand_infer_ms:.2f}ms, ram_delta={ram_delta_mb:.2f}MB)"
            )
            return False
        except Exception as e:
            logger.error(f"NEAT Weight Perturbation Error {self.model_id}: {e}")
            return False

    def atomic_save(self):
        if self.live_state and self._evo_state == "LIVE" and can_worker_write_canonical(self.model_id, self._current_transaction_id):
            mut_rec = getattr(self, 'last_mutation_record', "Base")
            return CheckpointManager.save(self.model_id, self.live_state, is_torch=False, extra_states={
                "generation": self.generation, "fitness": self.fitness, "topology_version": self.topology_version,
                "mutation_record": mut_rec, "transaction_id": getattr(self, 'last_committed_tx_id', None),
                "worker_epoch": int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
            })
        return False

# =========================================================================
# HELPER: Publish Health/Telemetry to CoreBus
# =========================================================================
def _publish_heartbeat(bus, module_name, episode_id):
    payload = {
        "state": "HEARTBEAT",
        "module": module_name,
        "worker_epoch": int(os.environ.get("ASUNA_WORKER_EPOCH", "1")),
        "worker_instance_id": os.environ.get("ASUNA_WORKER_INSTANCE_ID", f"{module_name}_local"),
        "modality": "control"
    }
    bus.publish("SYSTEM_CONTROL", payload, episode_id=episode_id)

def _publish_telemetry(bus, module_name, episode_id):
    curr = time.perf_counter()
    if not hasattr(bus, '_last_telemetry_time'):
        bus._last_telemetry_time = curr
        bus._msg_count_window = 0
    bus._msg_count_window = getattr(bus, '_msg_count_window', 0) + 1
    elapsed = curr - bus._last_telemetry_time
    if elapsed >= 1.0:
        step_latency_ms = max(0.1, min(100.0, (elapsed * 1000.0) / max(1, bus._msg_count_window)))
        try:
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            rss_mb = 0.0
        payload = {
            "module": module_name,
            "timestamp": curr,
            "generation": getattr(bus, '_sequence_id', 0),
            "loss": 0.0,
            "latency_ms": step_latency_ms,
            "rss_mb": rss_mb,
            "updates": getattr(bus, '_sequence_id', 0)
        }
        bus.publish("TELEMETRY", payload, modality="telemetry", episode_id=episode_id)
        bus._last_telemetry_time = curr
        bus._msg_count_window = 0

def _publish_candidate_ready(bus, module_name, candidate_hash, episode_id, transaction_id=None, command_id=""):
    payload = {"state": "CANDIDATE_READY", "module": module_name, "candidate_hash": candidate_hash, "transaction_id": transaction_id, "modality": "evolution"}
    bus.publish("SYSTEM_CONTROL", payload, episode_id=episode_id)
    if hasattr(bus, 'ctrl_socket'):
        ack = {"schema_version": 2, "type": "CONTROL_ACK", "command_id": command_id, "transaction_id": transaction_id, "source": module_name, "timestamp_ms": int(time.time() * 1000), "status": "PREPARED", "details": {"candidate_hash": candidate_hash}}
        bus._send_ctrl_msg([b'', json.dumps(ack).encode('utf-8')])
        
def _send_manual_ack(bus, status, tx_id, command_id):
    if hasattr(bus, 'ctrl_socket'):
        ack = {
            "schema_version": 2, "type": "CONTROL_ACK", 
            "command_id": command_id, "transaction_id": tx_id, 
            "source": bus.node_id, "timestamp_ms": int(time.time() * 1000), "status": status
        }
        bus._send_ctrl_msg([b'', json.dumps(ack).encode('utf-8')])

# =========================================================================
# 0. M0_CEREBROSPINAL_FLUID
# =========================================================================
def m0_cerebrospinal_fluid_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M0_Cerebrospinal")
    poller = zmq.Poller()
    # [BẢN VÁ FINAL]: M0 BẮT BUỘC phải nghe GLOBAL_DOPAMINE từ M7 để tổng hợp thành GLOBAL_PLASTICITY cho toàn não bộ!
    bus.subscribe(MODEL_SPECS["M0_Cerebrospinal"].get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_DOPAMINE", "ENV_VISION", "ENV_AUDIO", "ENV_SPATIAL", "ENV_HAZARD", "SELF_AUDIO", "MOTOR_FEEDBACK", "VOCAL_FEEDBACK", "THALAMUS_TOPDOWN"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")

    global_latent_state = np.zeros(128, dtype=np.float32)
    hormonal_state = {"dopamine": 0.5, "adrenaline": 0.1, "serotonin": 0.5, "cortisol": 0.1,
                      "threat": 0.0, "novelty": 0.0, "salience": 0.0, "arousal": 0.1}
    modality_states = {"threat": 0.0, "memory": 0.0, "salience": 0.0, "dopamine": 0.0, "novelty": 0.0, "arousal": 0.0}
    current_global_credit = {}
    
    chkpt, load_mode = load_worker_checkpoint_strict("M0_Cerebrospinal", is_torch=False, allow_bootstrap=True)
    if chkpt:
        m0_state = chkpt.get("model_state", chkpt) if isinstance(chkpt, dict) else {}
        if isinstance(m0_state, dict):
            hormonal_state.update(m0_state.get("hormonal_state", chkpt.get("hormonal_state", {})))
            modality_states.update(m0_state.get("modality_states", chkpt.get("modality_states", {})))
            if "global_latent_state" in m0_state:
                global_latent_state = np.array(m0_state["global_latent_state"], dtype=np.float32)

    global_tick, current_episode_id = 0, f"EP_{int(time.time())}"
    last_broadcast, last_heartbeat = time.monotonic(), time.monotonic()
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M0_Cerebrospinal', 'schema_verified': True, 'load_mode': load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M0_Cerebrospinal", current_episode_id)
    local_tick = 0
    active_brain_generation = chkpt.get("generation", 0) if isinstance(chkpt, dict) else 0
    
    try:
        while not stop_event.is_set():
            socks = dict(poller.poll(10)); global_tick += 1
            hormonal_state["dopamine"] += (0.5 - hormonal_state["dopamine"]) * 0.02
            hormonal_state["adrenaline"] += (0.1 - hormonal_state["adrenaline"]) * 0.05
            hormonal_state["cortisol"] += (0.1 - hormonal_state["cortisol"]) * 0.02
            hormonal_state["threat"] *= 0.95; hormonal_state["novelty"] *= 0.95; hormonal_state["salience"] *= 0.95
            hormonal_state["arousal"] *= 0.95; modality_states["dopamine"] *= 0.90

            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]
                    packet = extract_data(msg)
                    if isinstance(packet, dict):
                        payload = packet.get("payload", packet)
                        modality = packet.get("modality", "")

                        if topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                m0_snap = {"global_latent_state": global_latent_state.tolist(), "hormonal_state": hormonal_state, "modality_states": modality_states}
                                h = f"UNCHANGED_{tensor_state_hash(m0_snap)}"
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M0_Cerebrospinal', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M0_Cerebrospinal", h, current_episode_id, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                if isinstance(packet, dict) and "payload" in packet:
                                    active_brain_generation = packet["payload"].get("brain_generation", active_brain_generation)
                                _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                hormonal_state = {"dopamine": 0.5, "adrenaline": 0.1, "serotonin": 0.5, "cortisol": 0.1,
                                                    "threat": 0.0, "novelty": 0.0, "salience": 0.0, "arousal": 0.1}
                                modality_states = {"threat": 0.0, "memory": 0.0, "salience": 0.0, "dopamine": 0.0, "novelty": 0.0, "arousal": 0.0}
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))

                        if modality == "threat" and isinstance(payload, dict):
                            val = payload.get("threat_probability", 0.0)
                            hormonal_state["threat"] = val; modality_states["threat"] = val
                        elif modality == "memory" and isinstance(payload, dict):
                            val = 1.0 if "Không có" not in str(payload.get("retrieved_memory", "")) else 0.0
                            hormonal_state["novelty"] = val; modality_states["memory"] = val
                        elif modality == "salience" and isinstance(payload, dict):
                            val = payload.get("salience", 0.0)
                            hormonal_state["salience"] = val; modality_states["salience"] = val
                        elif modality == "dopamine" and isinstance(payload, dict):
                            if packet.get("source") != "M7_Logic":
                                logger.warning(f"⚠️ [M0_Cerebrospinal] Từ chối GLOBAL_DOPAMINE giả mạo từ nguồn: {packet.get('source')}. Đã chặn đứng cuộc tấn công Plasticity!")
                                continue
                            
                            val = payload.get("rpe", 0.0)
                            modality_states["dopamine"] = val
                            
                            if "credit" in payload:
                                current_global_credit = payload["credit"]
                                
                            if val > 0: hormonal_state["dopamine"] = min(1.0, hormonal_state["dopamine"] + val)
                            else: hormonal_state["cortisol"] = min(1.0, hormonal_state["cortisol"] + abs(val))
                        elif modality == "novelty" and isinstance(payload, dict):
                            val = payload.get("novelty", 0.0)
                            hormonal_state["novelty"] = val; modality_states["novelty"] = val
                        elif modality == "arousal" and isinstance(payload, dict):
                            val = payload.get("arousal", payload.get("adrenaline", 0.0))
                            hormonal_state["arousal"] = val; modality_states["arousal"] = val

                        if isinstance(payload, dict) and payload.get("latent_vector"):
                            vec_np = np.array(numeric_encoder(payload["latent_vector"], 128), dtype=np.float32)
                            global_latent_state = 0.8 * global_latent_state + 0.2 * vec_np

                        if isinstance(payload, dict) and "threat_probability" in payload:
                            hormonal_state["adrenaline"] = min(1.0, hormonal_state["adrenaline"] + payload["threat_probability"])
                            hormonal_state["cortisol"] = min(1.0, hormonal_state["cortisol"] + payload["threat_probability"] * 0.5)

                        if "episode_id" in packet and packet["episode_id"] != "UNKNOWN":
                            current_episode_id = packet["episode_id"]

            current_time = time.monotonic()
            if current_time - last_broadcast > 0.05:
                bus.current_tick = global_tick
                bus.current_episode = current_episode_id
                sync_payload = {
                    "global_latent_state": global_latent_state.tolist(),
                    "hormonal_state": hormonal_state,
                    "modality_states": modality_states,
                    "timestamp": time.time(), "tick": global_tick, "tick_id": global_tick
                }
                bus.publish("CEREBROSPINAL_FLUID", sync_payload, modality="neuromodulation", episode_id=current_episode_id, tick_id=global_tick)
                
                bus.publish("GLOBAL_PLASTICITY", {
                    "dopamine": hormonal_state["dopamine"],
                    "rpe": modality_states["dopamine"],
                    "novelty": hormonal_state["novelty"],
                    "salience": hormonal_state["salience"],
                    "threat": hormonal_state["threat"],
                    "arousal": hormonal_state["arousal"],
                    "global_latent": global_latent_state.tolist(),
                    "tick_id": global_tick,
                    "generation": active_brain_generation,
                    "credit": current_global_credit
                }, modality="neuromodulation", episode_id=current_episode_id, tick_id=global_tick)
                last_broadcast = current_time

            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M0_Cerebrospinal'})}")
                _publish_heartbeat(bus, "M0_Cerebrospinal", current_episode_id)
                last_heartbeat = current_time
            if current_time - getattr(bus, '_last_save_time', 0) > 60 and can_worker_write_canonical("M0_Cerebrospinal"):
                os.makedirs(os.path.join(BASE_DIR, "models", "M0_Cerebrospinal"), exist_ok=True)
                CheckpointManager.save("M0_Cerebrospinal", {
                    "global_latent_state": global_latent_state.tolist(),
                    "hormonal_state": hormonal_state,
                    "modality_states": modality_states
                }, is_torch=False, extra_states={"generation": active_brain_generation})
                bus._last_save_time = current_time
    finally: bus.shutdown(); upstream_sock.close()

# ================= 1a. M1A_DORSAL_STREAM =================
def m1a_dorsal_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M1a_Dorsal")
    spec = MODEL_SPECS["M1a_Dorsal"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "01_Sensory_Inputs", "M1_Eye_Vision")
    if code_dir in sys.path: sys.path.remove(code_dir)
    sys.path.insert(0, code_dir)
    try:
        from foveated_vision_system import FoveatedVisionSystem
        vision_system = FoveatedVisionSystem()
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo FoveatedVisionSystem M1a: {e}. Fallback resize cũ."); vision_system = None
    m1a_nn = DynamicNeatBrain("M1a_Dorsal", config_path=os.path.join(code_dir, "config-m1a-cppn"))
    is_dirty, is_sleeping = False, True
    last_heartbeat, last_save = time.monotonic(), time.monotonic()
    local_episode = "UNKNOWN"
    local_tick = 0
    current_m1a_focus, current_m1b_focus = 0.5, 0.5
    current_gaze_x, current_gaze_y = 0.5, 0.5
    current_topdown_payload = {"m1a_focus": 0.5, "m1b_focus": 0.5, "gaze": {"x": 0.5, "y": 0.5}}
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M1a_Dorsal', 'schema_verified': True, 'load_mode': m1a_nn.load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M1a_Dorsal", local_episode)
    
    try:
        while not stop_event.is_set():
            
            socks = dict(poller.poll(100))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]
                    packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M1a_Dorsal", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", m1a_nn.generation + 1)
                                cand_info = m1a_nn.prepare_candidate(tx_id, target_generation=target_gen)
                                h = cand_info.get("candidate_hash", "UNCHANGED")
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M1a_Dorsal', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M1a_Dorsal", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                target_gen = payload.get("brain_generation", m1a_nn.generation + 1)
                                if m1a_nn.commit_candidate(tx_id, target_generation=target_gen):
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    is_dirty = False; last_save = time.monotonic()
                                else:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                m1a_nn.rollback_candidate(tx_id)
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif not is_sleeping and topic == "ENV_VISION":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            try:
                                raw_gaze = payload.get("gaze") if isinstance(payload, dict) and isinstance(payload.get("gaze"), dict) else current_topdown_payload.get("gaze", {})
                                gaze_x = float(np.clip(float(raw_gaze.get("x", current_gaze_x)), 0.0, 1.0))
                                gaze_y = float(np.clip(float(raw_gaze.get("y", current_gaze_y)), 0.0, 1.0))
                                current_gaze_x, current_gaze_y = gaze_x, gaze_y
                                topdown_focus_arg = copy.deepcopy(current_topdown_payload)
                                topdown_focus_arg.update({
                                    "m1a_focus": float(np.clip(current_m1a_focus, 0.0, 1.0)),
                                    "m1b_focus": float(np.clip(current_m1b_focus, 0.0, 1.0)),
                                    "gaze": {"x": gaze_x, "y": gaze_y},
                                    "gaze_x": gaze_x,
                                    "gaze_y": gaze_y
                                })
                                if vision_system and isinstance(payload, dict) and payload.get("type") == "jpg_base64":
                                    b64_string = payload["data"]; b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
                                    img_data = base64.b64decode(b64_string); np_arr = np.frombuffer(img_data, np.uint8)
                                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                    if img is not None:
                                        features = vision_system.process(img, is_bgr=True, topdown_focus=topdown_focus_arg)
                                        flat_features = features["peripheral"]
                                        target_labels = payload.get("target_labels", [0.5, 0.5]) if isinstance(payload, dict) else [0.5, 0.5]
                                        target_x = float(np.clip(target_labels.get("x", 0.5) if isinstance(target_labels, dict) else target_labels[0], 0.0, 1.0))
                                        target_y = float(np.clip(target_labels.get("y", 0.5) if isinstance(target_labels, dict) else target_labels[1], 0.0, 1.0))
                                    else: raise ValueError("Ảnh không hợp lệ")
                                else:
                                    flat_features = numeric_encoder(payload, 20)
                                    target_labels = payload.get("target_labels", [0.5, 0.5]) if isinstance(payload, dict) else [0.5, 0.5]
                                    target_x = float(np.clip(target_labels.get("x", 0.5) if isinstance(target_labels, dict) else target_labels[0], 0.0, 1.0))
                                    target_y = float(np.clip(target_labels.get("y", 0.5) if isinstance(target_labels, dict) else target_labels[1], 0.0, 1.0))
                                flat_in = np.array(flat_features, dtype=float).flatten().tolist()
                                out = m1a_nn.activate(flat_in)
                                m1a_nn.update_trace(flat_in, out, [target_x, target_y])
                                pred_x, pred_y = float(np.clip(out[0], 0.0, 1.0)) if len(out)>0 else 0.5, float(np.clip(out[1], 0.0, 1.0)) if len(out)>1 else 0.5
                                m1a_payload = {"target": {"x": pred_x, "y": pred_y}, "gaze": {"x": gaze_x, "y": gaze_y}, "latent_vector": encode_spatial_fourier(pred_x, pred_y, dim=10), "timestamp": time.time()}
                                bus.publish("RAW_VISION_M1A", m1a_payload, modality="vision_where", parent_event_id=parent_id, episode_id=local_episode)
                            except Exception as e:
                                logger.warning(f"M1a Vision Error: {e}. Fallback hash.")
                                flat_features = numeric_encoder(payload, 20)
                                out = m1a_nn.activate(flat_features)
                                m1a_nn.update_trace(flat_features, out, [0.5, 0.5])
                                bus.publish("RAW_VISION_M1A", {"target": {"x": 0.5, "y": 0.5}, "gaze": {"x": current_gaze_x, "y": current_gaze_y}, "latent_vector": [0.0]*10}, modality="vision_where", parent_event_id=parent_id, episode_id=local_episode)
                        elif topic == "THALAMUS_TOPDOWN":
                            td_container = packet.get("payload", packet) if isinstance(packet, dict) else {}
                            topdown = td_container.get("topdown_attention", packet.get("topdown_attention")) if isinstance(td_container, dict) else None
                            if isinstance(topdown, dict):
                                current_topdown_payload = copy.deepcopy(topdown)
                                current_m1a_focus = float(np.clip(float(topdown.get("m1a_focus", current_m1a_focus)), 0.0, 1.0))
                                current_m1b_focus = float(np.clip(float(topdown.get("m1b_focus", current_m1b_focus)), 0.0, 1.0))
                                if isinstance(topdown.get("gaze"), dict):
                                    current_gaze_x = float(np.clip(float(topdown["gaze"].get("x", current_gaze_x)), 0.0, 1.0))
                                    current_gaze_y = float(np.clip(float(topdown["gaze"].get("y", current_gaze_y)), 0.0, 1.0))
                                    current_topdown_payload["gaze"] = {"x": current_gaze_x, "y": current_gaze_y}
                        
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                local_plasticity = compute_local_plasticity("M1a_Dorsal", dopa_payload)
                                if abs(local_plasticity) > 0.05 and m1a_nn.eligibility_trace["inputs"] is not None:
                                    try:
                                        m1a_nn.evolve_and_stage(m1a_nn.eligibility_trace["inputs"].tolist(), m1a_nn.eligibility_trace["targets"].tolist(), local_plasticity)
                                    except Exception as e:
                                        if diagnose_error(e, "M1a_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
            
            current_time = time.monotonic()
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M1a_Dorsal'})}")
                _publish_heartbeat(bus, "M1a_Dorsal", local_episode)
                last_heartbeat = current_time
            if is_dirty and current_time - last_save > 60:
                if m1a_nn.atomic_save(): is_dirty = False; last_save = current_time
    finally:
        if is_dirty: m1a_nn.atomic_save()
        bus.shutdown(); upstream_sock.close()

# ================= 1b. M1B_VENTRAL_STREAM =================
def m1b_ventral_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M1b_Ventral")
    spec = MODEL_SPECS["M1b_Ventral"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "01_Sensory_Inputs", "M1_Eye_Vision")
    if code_dir in sys.path: sys.path.remove(code_dir)
    sys.path.insert(0, code_dir)
    try:
        from foveated_vision_system import FoveatedVisionSystem
        vision_system = FoveatedVisionSystem()
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo FoveatedVisionSystem M1b: {e}. Fallback."); vision_system = None
    m1b_nn = DynamicNeatBrain("M1b_Ventral", config_path=os.path.join(code_dir, "config-m1b-cppn"))
    is_dirty, is_sleeping = False, True
    last_heartbeat, last_save = time.monotonic(), time.monotonic()
    local_episode = "UNKNOWN"
    local_tick = 0
    current_m1a_focus, current_m1b_focus = 0.5, 0.5
    current_gaze_x, current_gaze_y = 0.5, 0.5
    current_topdown_payload = {"m1a_focus": 0.5, "m1b_focus": 0.5, "gaze": {"x": 0.5, "y": 0.5}}
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M1b_Ventral', 'schema_verified': True, 'load_mode': m1b_nn.load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M1b_Ventral", local_episode)
    
    try:
        while not stop_event.is_set():
            
            socks = dict(poller.poll(timeout=100))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]
                    packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M1b_Ventral", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", m1b_nn.generation + 1)
                                cand_info = m1b_nn.prepare_candidate(tx_id, target_generation=target_gen)
                                h = cand_info.get("candidate_hash", "UNCHANGED")
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M1b_Ventral', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M1b_Ventral", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                target_gen = payload.get("brain_generation", m1b_nn.generation + 1)
                                if m1b_nn.commit_candidate(tx_id, target_generation=target_gen):
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    is_dirty = False; last_save = time.monotonic()
                                else:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                m1b_nn.rollback_candidate(tx_id)
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif not is_sleeping and topic == "ENV_VISION":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            try:
                                raw_gaze = payload.get("gaze") if isinstance(payload, dict) and isinstance(payload.get("gaze"), dict) else current_topdown_payload.get("gaze", {})
                                gaze_x = float(np.clip(float(raw_gaze.get("x", current_gaze_x)), 0.0, 1.0))
                                gaze_y = float(np.clip(float(raw_gaze.get("y", current_gaze_y)), 0.0, 1.0))
                                current_gaze_x, current_gaze_y = gaze_x, gaze_y
                                topdown_focus_arg = copy.deepcopy(current_topdown_payload)
                                topdown_focus_arg.update({
                                    "m1a_focus": float(np.clip(current_m1a_focus, 0.0, 1.0)),
                                    "m1b_focus": float(np.clip(current_m1b_focus, 0.0, 1.0)),
                                    "gaze": {"x": gaze_x, "y": gaze_y},
                                    "gaze_x": gaze_x,
                                    "gaze_y": gaze_y
                                })
                                if vision_system and isinstance(payload, dict) and payload.get("type") == "jpg_base64":
                                    b64_string = payload["data"]; b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
                                    img_data = base64.b64decode(b64_string); np_arr = np.frombuffer(img_data, np.uint8)
                                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                    if img is not None:
                                        features = vision_system.process(img, is_bgr=True, topdown_focus=topdown_focus_arg)
                                        flat_features = features["foveal"]
                                        target_labels = payload.get("target_labels", [0.5, 0.5]) if isinstance(payload, dict) else [0.5, 0.5]
                                        tx = float(np.clip(target_labels.get("x", 0.5) if isinstance(target_labels, dict) else target_labels[0], 0.0, 1.0))
                                        ty = float(np.clip(target_labels.get("y", 0.5) if isinstance(target_labels, dict) else target_labels[1], 0.0, 1.0))
                                        dist_to_fovea = np.sqrt((tx - gaze_x)**2 + (ty - gaze_y)**2)
                                        target_conf = max(0.0, 1.0 - dist_to_fovea * 2.0)
                                        target_obj = 1.0 if target_conf > 0.5 else 0.0
                                    else: raise ValueError("Ảnh không hợp lệ")
                                else:
                                    flat_features = numeric_encoder(payload, 20)
                                    target_labels = payload.get("target_labels", [0.5, 0.5]) if isinstance(payload, dict) else [0.5, 0.5]
                                    tx = float(np.clip(target_labels.get("x", 0.5) if isinstance(target_labels, dict) else target_labels[0], 0.0, 1.0))
                                    ty = float(np.clip(target_labels.get("y", 0.5) if isinstance(target_labels, dict) else target_labels[1], 0.0, 1.0))
                                    dist_to_fovea = np.sqrt((tx - gaze_x)**2 + (ty - gaze_y)**2)
                                    target_conf = max(0.0, 1.0 - dist_to_fovea * 2.0)
                                    target_obj = 1.0 if target_conf > 0.5 else 0.0
                                
                                flat_in = (np.array(flat_features, dtype=float).flatten() * (1.0 + current_m1b_focus)).tolist()
                                out = m1b_nn.activate(flat_in)
                                m1b_nn.update_trace(flat_in, out, [target_obj, target_conf])
                                confidence = float(np.clip(out[0], 0.0, 1.0)) if len(out) > 0 else 0.0
                                label = "UNKNOWN" if confidence < 0.5 else "OBJECT_DETECTED"
                                m1b_payload = {"objects": [label] if label != "UNKNOWN" else [], "confidence": confidence, "gaze": {"x": gaze_x, "y": gaze_y}, "latent_vector": encode_vision_onehot(label, dim=10)}
                                bus.publish("RAW_VISION_M1B", m1b_payload, modality="vision_what", parent_event_id=parent_id, episode_id=local_episode)
                            except Exception as e:
                                logger.warning(f"M1b Vision Error: {e}. Fallback.")
                                bus.publish("RAW_VISION_M1B", {"objects": [], "confidence": 0.0, "gaze": {"x": current_gaze_x, "y": current_gaze_y}, "latent_vector": [0.0]*10}, modality="vision_what", parent_event_id=parent_id, episode_id=local_episode)
                        elif topic == "THALAMUS_TOPDOWN":
                            td_container = packet.get("payload", packet) if isinstance(packet, dict) else {}
                            topdown = td_container.get("topdown_attention", packet.get("topdown_attention")) if isinstance(td_container, dict) else None
                            if isinstance(topdown, dict):
                                current_topdown_payload = copy.deepcopy(topdown)
                                current_m1a_focus = float(np.clip(float(topdown.get("m1a_focus", current_m1a_focus)), 0.0, 1.0))
                                current_m1b_focus = float(np.clip(float(topdown.get("m1b_focus", current_m1b_focus)), 0.0, 1.0))
                                if isinstance(topdown.get("gaze"), dict):
                                    current_gaze_x = float(np.clip(float(topdown["gaze"].get("x", current_gaze_x)), 0.0, 1.0))
                                    current_gaze_y = float(np.clip(float(topdown["gaze"].get("y", current_gaze_y)), 0.0, 1.0))
                                    current_topdown_payload["gaze"] = {"x": current_gaze_x, "y": current_gaze_y}
                        
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                local_plasticity = compute_local_plasticity("M1b_Ventral", dopa_payload)
                                if abs(local_plasticity) > 0.05 and m1b_nn.eligibility_trace["inputs"] is not None:
                                    try:
                                        m1b_nn.evolve_and_stage(m1b_nn.eligibility_trace["inputs"].tolist(), m1b_nn.eligibility_trace["targets"].tolist(), local_plasticity)
                                    except Exception as e:
                                        if diagnose_error(e, "M1b_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
            
            current_time = time.monotonic()
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M1b_Ventral'})}")
                _publish_heartbeat(bus, "M1b_Ventral", local_episode)
                last_heartbeat = current_time
            if is_dirty and current_time - last_save > 60:
                if m1b_nn.atomic_save(): is_dirty = False; last_save = current_time
    finally:
        if is_dirty: m1b_nn.atomic_save()
        bus.shutdown(); upstream_sock.close()

# ================= 2. M2_EAR_HEARING =================
def m2_ear_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M2_Ear")
    spec = MODEL_SPECS["M2_Ear"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "01_Sensory_Inputs/M2_Ear_Hearing")
    if code_dir not in sys.path: sys.path.append(code_dir)
    
    # [BẢN VÁ P0]: Trì hoãn import để tránh Deadlock RAM
    try:
        from train_m2_esnn import EvolvingSNN
        import torch
    except ImportError as e:
        logger.critical(f"🛑 Lỗi nạp thư viện M2_Ear: {e}"); stop_event.set(); return

    chkpt, load_mode = load_worker_checkpoint_strict("M2_Ear", is_torch=True)
    if chkpt:
        topology = chkpt.get("topology", {}); num_classes = topology.get("num_classes", 3)
        live_state = EvolvingSNN(num_inputs=20, initial_classes=num_classes)
        with torch.no_grad(): live_state(torch.randn(10, 1, 20))
        try:
            state_dict = chkpt["model_state"]
            live_state.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.critical(f"💀 [RECOVERY_REQUIRED] M2_Ear Mismatch! {e}")
            raise RuntimeError(f"RECOVERY_REQUIRED:M2_Ear:{e}")
        optimizer = torch.optim.Adam(live_state.parameters(), lr=1e-3)
        if chkpt.get("optimizer_state"):
            try: optimizer.load_state_dict(chkpt["optimizer_state"])
            except Exception: pass
        generation, fitness, topology_version = chkpt.get("generation", 0), chkpt.get("fitness", 0.0), chkpt.get("topology_version", chkpt.get("architecture_version", 1))
    else:
        live_state = EvolvingSNN(num_inputs=20, initial_classes=3)
        optimizer = torch.optim.Adam(live_state.parameters(), lr=1e-3)
        topology_version, generation, fitness = 1, 0, 0.0
    parent_state_backup = None; parent_opt_backup = None
    staged_state = None; staged_opt = None; staged_tx_id = None; staged_fitness_inc = 0.0
    is_dirty, last_save, last_heartbeat = False, time.monotonic(), time.monotonic()
    is_sleeping, last_m2_target_class, eligibility_trace = True, -1, None
    local_episode = "UNKNOWN"; lateral_vocal = None; m2_focus = 0.0
    audio_history = deque(maxlen=10) 
    for _ in range(10): audio_history.append([0.0]*20)
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M2_Ear', 'schema_verified': True, 'load_mode': load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M2_Ear", local_episode)
    local_tick = 0
    
    try:
        while not stop_event.is_set():
            socks = dict(poller.poll(100))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M2_Ear", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", generation + 1)
                                if staged_tx_id != tx_id:
                                    if staged_tx_id is not None:
                                        staged_state, staged_opt, staged_fitness_inc = None, None, 0.0
                                    staged_tx_id = tx_id
                                    parent_gen_backup = generation
                                    parent_topo_ver_backup = topology_version
                                    parent_fit_backup = fitness
                                    parent_model_backup = copy.deepcopy(live_state)
                                    parent_state_backup = copy.deepcopy(live_state.state_dict())
                                    parent_opt_backup = copy.deepcopy(optimizer.state_dict())
                                cand_ref = staged_state if (staged_state is not None and staged_state is not live_state) else live_state
                                h_dim = cand_ref.fc1.out_features if hasattr(cand_ref, 'fc1') else getattr(cand_ref, 'hidden_dim', 64)
                                topo_d = {"hidden_dim": h_dim, "num_inputs": 20, "num_classes": getattr(cand_ref, 'num_classes', 3)}
                                cand_topo_ver = topology_version + (1 if (staged_state is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in live_state.parameters())) else 0)
                                h = stage_pytorch_candidate_file("M2_Ear", live_state, staged_state, optimizer, staged_opt, tx_id, generation, target_gen, fitness + staged_fitness_inc, cand_topo_ver, topo_d)
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M2_Ear', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M2_Ear", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                if 'last_committed_tx_id' in locals() and tx_id == last_committed_tx_id:
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    continue
                                if staged_tx_id != tx_id:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                                    continue
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                generation = payload.get("brain_generation", generation + 1)
                                if staged_state is not None and staged_state is not live_state:
                                    if 'parent_model_backup' in locals() and parent_model_backup is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in parent_model_backup.parameters()):
                                        topology_version += 1 
                                    live_state = staged_state
                                    if staged_opt is not None:
                                        optimizer = staged_opt
                                    fitness += staged_fitness_inc
                                staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                last_committed_tx_id = tx_id
                                is_dirty = False; last_save = time.monotonic()
                                _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                CheckpointManager.discard_candidate("M2_Ear", tx_id)
                                if staged_tx_id == tx_id or locals().get('last_committed_tx_id') == tx_id:
                                    staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                    if locals().get('last_committed_tx_id') == tx_id:
                                        last_committed_tx_id = None
                                    if 'parent_gen_backup' in locals(): generation = parent_gen_backup
                                    if 'parent_topo_ver_backup' in locals(): topology_version = parent_topo_ver_backup
                                    if 'parent_fit_backup' in locals(): fitness = parent_fit_backup
                                    if 'parent_model_backup' in locals() and parent_model_backup is not None:
                                        live_state = copy.deepcopy(parent_model_backup)
                                        if 'optimizer' in locals() and optimizer is not None:
                                            optimizer = type(optimizer)(filter(lambda p: p.requires_grad, live_state.parameters()), **optimizer.defaults)
                                            if sum(p.numel() for p in live_state.parameters()) == sum(p.numel() for p in parent_model_backup.parameters()):
                                                try: optimizer.load_state_dict(parent_opt_backup)
                                                except Exception: pass
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif topic == "VOCAL_FEEDBACK": lateral_vocal = packet.get("payload", packet) if isinstance(packet, dict) else packet
                        elif not is_sleeping and topic in ["ENV_AUDIO", "SELF_AUDIO"]:
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            try:
                                label_str = "Unknown"
                                if isinstance(payload, dict) and "data" in payload:
                                    raw_audio_list = payload["data"]; label_str = payload.get("label", "Unknown")
                                elif isinstance(payload, list): raw_audio_list = payload
                                else: raw_audio_list = [0.0] * 1600
                                audio_np = np.array(raw_audio_list, dtype=np.float32)
                                last_m2_target_class = 0 if label_str == "Silence" else (1 if label_str == "Vowel" else (2 if label_str == "Consonant" else -1))
                                mel_features = encode_audio_stft(audio_np, dim=20)
                                fused_audio = numeric_encoder({"audio": mel_features, "vocal": lateral_vocal}, 20)
                                modulated_audio = (np.array(fused_audio, dtype=np.float32) * (1.0 + m2_focus)).tolist()
                                audio_history.append(modulated_audio)
                                current_trace = torch.tensor(list(audio_history), dtype=torch.float32).unsqueeze(1) 
                                if eligibility_trace is None: eligibility_trace = current_trace
                                else: eligibility_trace = 0.9 * eligibility_trace.detach() + 0.1 * current_trace
                                live_state.eval()
                                with torch.no_grad():
                                    spikes = live_state(eligibility_trace)
                                    last_m2_tensor = spikes.sum(dim=0)
                                    pred_idx = int(torch.argmax(last_m2_tensor[0]).item())
                                phoneme_guess = ["Silence", "Vowel", "Consonant"][pred_idx] if pred_idx < 3 else f"Unknown_{pred_idx}"
                                m2_payload = { "acoustic_features": last_m2_tensor.tolist()[0], "phoneme_guess": phoneme_guess, "intensity": float(np.mean(np.abs(audio_np))) }
                                bus.publish("RAW_AUDIO", m2_payload, modality="audio", parent_event_id=parent_id, episode_id=local_episode)
                            except Exception as e:
                                if diagnose_error(e, "M2_Ear") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif topic == "THALAMUS_TOPDOWN":
                            td_payload = packet.get("payload", packet) if isinstance(packet, dict) else {}
                            topdown = td_payload.get("topdown_attention", packet.get("topdown_attention")) if isinstance(td_payload, dict) else None
                            if isinstance(topdown, dict):
                                m2_focus = float(np.clip(float(topdown.get("m2_focus", m2_focus)), 0.0, 1.0))
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                novelty_val = float(dopa_payload.get("novelty", 0.0))
                                local_plasticity = compute_local_plasticity("M2_Ear", dopa_payload)
                                if abs(local_plasticity) > 0.05 and eligibility_trace is not None:
                                    if staged_tx_id is None:
                                        try:
                                            def m2_val_loss(model, x):
                                                out_t = model(x).sum(dim=0)
                                                if out_t.dim() == 1: out_t = out_t.unsqueeze(0)
                                                if last_m2_target_class != -1:
                                                    target = torch.tensor([last_m2_target_class], dtype=torch.long, device=out_t.device)
                                                    return torch.nn.functional.cross_entropy(out_t, target)
                                                else: return -torch.nn.functional.log_softmax(out_t, dim=-1)[0, torch.argmax(out_t[0]).item()]
                                            m2_mut = lambda net, opt=None: net.sprout_new_neuron(optimizer=opt) if hasattr(net, "sprout_new_neuron") else None
                                            base_m2 = staged_state if staged_state is not None else live_state
                                            base_m2_opt = staged_opt if staged_opt is not None else optimizer
                                            is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m2, eligibility_trace, m2_val_loss, base_m2_opt, rpe=local_plasticity, novelty=novelty_val, mutate_func=m2_mut)
                                            if is_valid:
                                                staged_state, staged_opt = new_staged, new_opt
                                                staged_fitness_inc = local_plasticity
                                        except Exception as e:
                                            if diagnose_error(e, "M2_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
            current_time = time.monotonic()
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M2_Ear'})}")
                _publish_heartbeat(bus, "M2_Ear", local_episode)
                last_heartbeat = current_time
            if is_dirty and current_time - last_save > 60 and can_worker_write_canonical("M2_Ear", staged_tx_id):
                hidden_dim = getattr(live_state, 'fc1', None); h_dim = hidden_dim.out_features if hidden_dim else 64
                extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_inputs": 20, "hidden_dim": h_dim, "num_classes": getattr(live_state, 'num_classes', 3)}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
                CheckpointManager.save("M2_Ear", live_state, optimizer, is_torch=True, extra_states=extra_st); is_dirty = False; last_save = current_time
    finally:
        if is_dirty and can_worker_write_canonical("M2_Ear", staged_tx_id):
            hidden_dim = getattr(live_state, 'fc1', None); h_dim = hidden_dim.out_features if hidden_dim else 64
            extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_inputs": 20, "hidden_dim": h_dim, "num_classes": getattr(live_state, 'num_classes', 3)}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
            CheckpointManager.save("M2_Ear", live_state, optimizer, is_torch=True, extra_states=extra_st)
        bus.shutdown(); upstream_sock.close()
        
# ================= 3. M3_SPATIAL_PARIETAL =================
def m3_spatial_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M3_Spatial")
    spec = MODEL_SPECS["M3_Spatial"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "01_Sensory_Inputs/M3_Spatial_Parietal")
    if code_dir not in sys.path: sys.path.append(code_dir)
    m3_nn = DynamicNeatBrain("M3_Spatial", config_path=os.path.join(code_dir, "config-m3"))
    if not NEAT_AVAILABLE: return
    is_dirty, is_sleeping = False, True
    last_x, last_y, last_time = 0.5, 0.5, time.monotonic()
    last_heartbeat, last_save = time.monotonic(), time.monotonic()
    spatial_history = deque(maxlen=5)
    for _ in range(5): spatial_history.append([0.5, 0.5])
    local_episode = "UNKNOWN"; lateral_motor = None
    local_tick = 0; m3_focus = 0.0 # [BẢN VÁ AUDIT]: Khởi tạo biến focus cho M3
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M3_Spatial', 'schema_verified': True, 'load_mode': m3_nn.load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M3_Spatial", local_episode)
    
    try:
        while not stop_event.is_set():
            
            socks = dict(poller.poll(100))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M3_Spatial", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", m3_nn.generation + 1)
                                cand_info = m3_nn.prepare_candidate(tx_id, target_generation=target_gen)
                                h = cand_info.get("candidate_hash", "UNCHANGED")
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M3_Spatial', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M3_Spatial", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                target_gen = payload.get("brain_generation", m3_nn.generation + 1)
                                if m3_nn.commit_candidate(tx_id, target_generation=target_gen):
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    is_dirty = False; last_save = time.monotonic()
                                else:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                m3_nn.rollback_candidate(tx_id)
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif topic == "MOTOR_FEEDBACK": lateral_motor = packet.get("payload", packet) if isinstance(packet, dict) else packet
                        elif topic == "THALAMUS_TOPDOWN":
                            if isinstance(packet, dict) and isinstance(packet.get("topdown_attention"), dict):
                                topdown = packet["topdown_attention"]
                                m3_focus = float(topdown.get("m3_focus", 0.0))
                        
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                local_plast = compute_local_plasticity("M3_Spatial", dopa_payload)
                                if abs(local_plast) > 0.05 and m3_nn.eligibility_trace["inputs"] is not None:
                                    try:
                                        m3_nn.evolve_and_stage(m3_nn.eligibility_trace["inputs"].tolist(), m3_nn.eligibility_trace["targets"].tolist(), local_plast)
                                    except Exception as e:
                                        if diagnose_error(e, "M3_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif not is_sleeping and topic == "ENV_SPATIAL":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            try:
                                if isinstance(payload, dict) and "x" in payload and "y" in payload:
                                    curr_x, curr_y = float(payload["x"]), float(payload["y"])
                                else:
                                    embed_arr = numeric_encoder(payload, 2)
                                    if len(embed_arr) < 2:
                                        embed_arr = embed_arr + [0.5] * (2 - len(embed_arr))
                                    curr_x, curr_y = embed_arr[0], embed_arr[1]
                                current_time = time.monotonic(); dt = max(current_time - last_time, 0.001)
                                vx, vy = (curr_x - last_x)/dt, (curr_y - last_y)/dt
                                spatial_history.append([curr_x, curr_y])
                                
                                motor_vec = numeric_encoder(lateral_motor, 4) if lateral_motor else [0.0]*4
                                
                                fused_spatial = [curr_x, curr_y, vx, vy] + motor_vec + [m3_focus, 0.0]
                                
                                previous_fused = getattr(m3_nn, 'last_fused_input', None)
                                previous_out = getattr(m3_nn, 'last_out_vector', None)
                                if previous_fused is not None and previous_out is not None:
                                    m3_nn.update_trace(previous_fused, previous_out, [curr_x, curr_y])
                                
                                flat_fused = np.array(fused_spatial, dtype=float).flatten().tolist()
                                out = m3_nn.activate(flat_fused)
                                
                                m3_nn.last_fused_input = fused_spatial
                                m3_nn.last_out_vector = out
                                
                                pred_next_x = float(np.clip(out[0], 0.0, 1.0)) if out and len(out) > 0 else 0.5
                                pred_next_y = float(np.clip(out[1], 0.0, 1.0)) if out and len(out) > 1 else 0.5
                                last_x, last_y, last_time = curr_x, curr_y, current_time
                                m3_payload = {"current_pos": {"x": curr_x, "y": curr_y}, "predicted_next_pos": {"x": pred_next_x, "y": pred_next_y}, "velocity": float(np.sqrt(vx**2 + vy**2))}
                                bus.publish("RAW_SPATIAL", m3_payload, modality="spatial", parent_event_id=parent_id, episode_id=local_episode)
                            except Exception as e:
                                if diagnose_error(e, "M3_Spatial") == BrainErrorCategory.FATAL: stop_event.set(); break
            
            current_time = time.monotonic()
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M3_Spatial'})}")
                _publish_heartbeat(bus, "M3_Spatial", local_episode)
                last_heartbeat = current_time
            if is_dirty and current_time - last_save > 60:
                if m3_nn.atomic_save(): is_dirty = False; last_save = current_time
    finally:
        if is_dirty: m3_nn.atomic_save()
        bus.shutdown(); upstream_sock.close()

# ================= 4. M4_MOTOR_CORTEX =================
def m4_motor_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M4_Motor")
    spec = MODEL_SPECS["M4_Motor"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID", "CMD_MOTOR", "THALAMUS_M4", "AMYGDALA_THREAT", "MOTOR_FEEDBACK", "DREAM_REPLAY", "THALAMUS_TOPDOWN"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "02_Motor_Actuators/M4_Motor_Cortex")
    if code_dir not in sys.path: sys.path.append(code_dir)
    m4_nn = DynamicNeatBrain("M4_Motor", config_path=os.path.join(code_dir, "config-m4"))
    if not NEAT_AVAILABLE: return
    task_queue = deque(maxlen=50); motor_history = {}
    is_dirty, is_sleeping, motor_inhibited = False, True, False
    last_heartbeat, last_save = time.monotonic(), time.monotonic()
    local_episode = "UNKNOWN"; lateral_threat = 0.0
    local_tick = 0; latest_thalamus_m4 = None
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M4_Motor', 'schema_verified': True, 'load_mode': m4_nn.load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M4_Motor", local_episode)
    
    try:
        while not stop_event.is_set():
            
            socks = dict(poller.poll(50))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M4_Motor", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping, motor_inhibited = False, False
                            elif st == "EMERGENCY_STOP":
                                motor_inhibited = True
                                task_queue.clear()
                                task_queue.append(({"action": "EMERGENCY_REFLEX", "state_snapshot": {}}, f"reflex_{local_tick}", packet.get("event_id")))
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", m4_nn.generation + 1)
                                cand_info = m4_nn.prepare_candidate(tx_id, target_generation=target_gen)
                                h = cand_info.get("candidate_hash", "UNCHANGED")
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M4_Motor', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M4_Motor", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                target_gen = payload.get("brain_generation", m4_nn.generation + 1)
                                if m4_nn.commit_candidate(tx_id, target_generation=target_gen):
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    is_dirty = False; last_save = time.monotonic()
                                else:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                m4_nn.rollback_candidate(tx_id)
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif topic == "AMYGDALA_THREAT":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict):
                                lateral_threat = max(float(payload.get("threat_probability", 0.0)), float(payload.get("emergency_priority", 0.0)))
                        elif not is_sleeping and topic == "THALAMUS_M4":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict): 
                                latest_thalamus_m4 = payload
                                if (isinstance(lateral_threat, float) and lateral_threat > 0.5) or motor_inhibited:
                                    task_id = f"reflex_{local_tick}"
                                    parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                                    task_queue.append(({"action": "EMERGENCY_REFLEX", "state_snapshot": {}}, task_id, parent_id))
                        elif not is_sleeping and not motor_inhibited and topic == "CMD_MOTOR":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            task_id = packet.get("task_id", payload.get("task_id", "NO_TASK")) if isinstance(packet, dict) else "NO_TASK"
                            if isinstance(payload, dict): task_queue.append((payload, task_id, parent_id))
                        elif topic == "THALAMUS_TOPDOWN":
                            if isinstance(packet, dict) and "topdown_attention" in packet:
                                topdown = packet["topdown_attention"]
                        
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                local_plasticity = compute_local_plasticity("M4_Motor", dopa_payload)
                                if abs(local_plasticity) > 0.05 and m4_nn.eligibility_trace["inputs"] is not None:
                                    try:
                                        m4_nn.evolve_and_stage(m4_nn.eligibility_trace["inputs"].tolist(), m4_nn.eligibility_trace["targets"].tolist(), local_plasticity)
                                    except Exception as e:
                                        if diagnose_error(e, "M4_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif not is_sleeping and not motor_inhibited and topic == "MOTOR_FEEDBACK":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and payload.get("task_id") in motor_history:
                                hist = motor_history[payload["task_id"]]
                                target_vec = hist.get("ideal_target", [hist["out_dx"], hist["out_dy"], 0.0])
                                fb_rpe = float(payload.get("rpe", payload.get("reward", 0.0)))
                                local_motor_error = (hist["out_dx"] - target_vec[0])**2 + (hist["out_dy"] - target_vec[1])**2
                                if local_motor_error > 0.02 or fb_rpe < -0.05:
                                    m4_nn.evolve_and_stage(hist["inputs"], target_vec, fb_rpe if fb_rpe != 0.0 else -local_motor_error)
                                del motor_history[payload["task_id"]]
                        elif is_sleeping and topic == "DREAM_REPLAY":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "vector" in payload:
                                try: m4_nn.activate(numeric_encoder(payload["vector"], 10))
                                except Exception as e:
                                    if diagnose_error(e, "M4_Dream") == BrainErrorCategory.FATAL: stop_event.set(); break
            current_time = time.monotonic()
            
            expired_tasks = [tid for tid, hist in motor_history.items() if current_time - hist.get("timestamp", current_time) > 10.0]
            for tid in expired_tasks: del motor_history[tid]
            
            if len(task_queue) > 0 and not is_sleeping and (not motor_inhibited or task_queue[0][0].get("action") == "EMERGENCY_REFLEX"):
                current_task, current_task_id, current_parent_id = task_queue.popleft()
                try:
                    state_snap = current_task.get("state_snapshot", {})
                    default_x = float(latest_thalamus_m4.get("x", 0.5)) if isinstance(latest_thalamus_m4, dict) else 0.5
                    default_y = float(latest_thalamus_m4.get("y", 0.5)) if isinstance(latest_thalamus_m4, dict) else 0.5
                    curr_x = float(state_snap.get("x", default_x)) if isinstance(state_snap, dict) else default_x
                    curr_y = float(state_snap.get("y", default_y)) if isinstance(state_snap, dict) else default_y
                    targ_x = float(state_snap.get("target_x", curr_x)) if isinstance(state_snap, dict) else curr_x
                    targ_y = float(state_snap.get("target_y", curr_y)) if isinstance(state_snap, dict) else curr_y
                    
                    if current_task.get("action") == "EMERGENCY_REFLEX":
                        targ_x = 0.5
                        targ_y = 0.5
                    elif isinstance(latest_thalamus_m4, dict) and ("target_x" in latest_thalamus_m4 or "target_y" in latest_thalamus_m4):
                        targ_x = float(latest_thalamus_m4.get("target_x", targ_x))
                        targ_y = float(latest_thalamus_m4.get("target_y", targ_y))
                    elif isinstance(locals().get("topdown"), dict) and isinstance(topdown.get("gaze"), dict):
                        targ_x = float(topdown["gaze"].get("x", targ_x))
                        targ_y = float(topdown["gaze"].get("y", targ_y))
                    thal_vec = numeric_encoder(latest_thalamus_m4.get("fused_latent", []), 5) if isinstance(latest_thalamus_m4, dict) else [0.0] * 5
                    fused_motor = [curr_x, curr_y, targ_x, targ_y, float(lateral_threat)] + thal_vec[:5]
                    flat_fused_motor = fused_motor
                    
                    out_vector = m4_nn.activate(flat_fused_motor)
                    
                    if len(out_vector) < 3:
                        out_vector = out_vector + [0.0] * (3 - len(out_vector))
                        
                    out_dx, out_dy = float(np.clip(out_vector[0], -1.0, 1.0)), float(np.clip(out_vector[1], -1.0, 1.0))
                    ideal_dx = float(np.clip((targ_x - curr_x) * 5.0, -1.0, 1.0))
                    ideal_dy = float(np.clip((targ_y - curr_y) * 5.0, -1.0, 1.0))
                    if current_task.get("action") == "EMERGENCY_REFLEX":
                        out_dx, out_dy = ideal_dx, ideal_dy
                    ideal_click = 1.0 if float(np.hypot(targ_x - curr_x, targ_y - curr_y)) < 0.05 else 0.0
                    ideal_target = [ideal_dx, ideal_dy, ideal_click]
                    motor_history[current_task_id] = {
                        "inputs": fused_motor, "out_dx": out_dx, "out_dy": out_dy,
                        "ideal_target": ideal_target, "timestamp": current_time
                    }
                    m4_nn.update_trace(fused_motor, out_vector, ideal_target)
                    m4_payload = { "status": "EXECUTING", "task_id": current_task.get("task_id", current_task_id), "action_name": current_task.get("action", "REFLEX"), "dx": out_dx, "dy": out_dy, "click": bool(out_vector[2] > 0.5) }
                    bus.publish("MOTOR_EXECUTE", m4_payload, modality="motor", parent_event_id=current_parent_id, episode_id=local_episode)
                except Exception as e:
                    if diagnose_error(e, "M4_Motor") == BrainErrorCategory.FATAL: stop_event.set(); break
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M4_Motor'})}")
                _publish_heartbeat(bus, "M4_Motor", local_episode)
                last_heartbeat = current_time
            if is_dirty and current_time - last_save > 60:
                if m4_nn.atomic_save(): is_dirty = False; last_save = current_time
    finally:
        if is_dirty: m4_nn.atomic_save()
        bus.shutdown(); upstream_sock.close()

# ================= 5. M5_AMYGDALA =================
def m5_amygdala_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M5_Amygdala")
    spec = MODEL_SPECS["M5_Amygdala"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "00_Central_Router/M5_Amygdala")
    if code_dir not in sys.path: sys.path.append(code_dir)
    
    # [BẢN VÁ P0]: Trì hoãn import để tránh Deadlock RAM
    try:
        from train_m5_snn import SproutingSNN
        import torch
    except ImportError as e:
        logger.critical(f"🛑 Lỗi nạp thư viện M5_Amygdala: {e}"); stop_event.set(); return

    chkpt, load_mode = load_worker_checkpoint_strict("M5_Amygdala", is_torch=True)
    if chkpt:
        topology = chkpt.get("topology", {}); num_hidden = topology.get("num_hidden", topology.get("hidden_dim", 64))
        state_dict = chkpt["model_state"]
        if "fc1.weight" in state_dict:
            num_hidden = int(state_dict["fc1.weight"].shape[0])
        live_state = SproutingSNN(num_inputs=20, num_hidden=num_hidden, num_outputs=2)
        with torch.no_grad(): live_state(torch.randn(10, 1, 20))
        try: 
            live_state.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.critical(f"💀 [RECOVERY_REQUIRED] M5_Amygdala Mismatch! {e}")
            raise RuntimeError(f"RECOVERY_REQUIRED:M5_Amygdala:{e}")
        optimizer = torch.optim.Adam(live_state.parameters(), lr=1e-3)
        if chkpt.get("optimizer_state"):
            try: optimizer.load_state_dict(chkpt["optimizer_state"])
            except Exception: pass
        generation, fitness, topology_version = chkpt.get("generation", 0), chkpt.get("fitness", 0.0), chkpt.get("topology_version", chkpt.get("architecture_version", 1))
    else:
        live_state = SproutingSNN(num_inputs=20, num_hidden=64, num_outputs=2)
        optimizer = torch.optim.Adam(live_state.parameters(), lr=1e-3)
        topology_version, generation, fitness = 1, 0, 0.0
    parent_state_backup = None; parent_opt_backup = None
    staged_state = None; staged_opt = None; staged_tx_id = None; staged_fitness_inc = 0.0
    live_state.train(); is_dirty, is_sleeping, last_heartbeat = False, True, time.monotonic()
    eligibility_trace, last_save = None, time.monotonic()
    local_episode = "UNKNOWN"
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M5_Amygdala', 'schema_verified': True, 'load_mode': load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M5_Amygdala", local_episode)
    local_tick = 0
    
    try:
        current_danger_label = 0
        m5_history = []
        while not stop_event.is_set():
            socks = dict(poller.poll(100))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M5_Amygdala", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", generation + 1)
                                if staged_tx_id != tx_id:
                                    if staged_tx_id is not None:
                                        staged_state, staged_opt, staged_fitness_inc = None, None, 0.0
                                    staged_tx_id = tx_id
                                    parent_gen_backup = generation
                                    parent_topo_ver_backup = topology_version
                                    parent_fit_backup = fitness
                                    parent_model_backup = copy.deepcopy(live_state)
                                    parent_state_backup = copy.deepcopy(live_state.state_dict())
                                    parent_opt_backup = copy.deepcopy(optimizer.state_dict())
                                cand_ref = staged_state if staged_state is not None else live_state
                                h_dim = cand_ref.fc1.out_features if hasattr(cand_ref, 'fc1') else getattr(cand_ref, 'num_hidden', 64)
                                topo_d = {"hidden_dim": h_dim, "num_inputs": 20, "num_hidden": getattr(cand_ref, 'num_hidden', h_dim), "num_outputs": 2}
                                cand_topo_ver = topology_version + (1 if (staged_state is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in live_state.parameters())) else 0)
                                h = stage_pytorch_candidate_file("M5_Amygdala", live_state, staged_state, optimizer, staged_opt, tx_id, generation, target_gen, fitness + staged_fitness_inc, cand_topo_ver, topo_d)
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M5_Amygdala', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M5_Amygdala", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                if 'last_committed_tx_id' in locals() and tx_id == last_committed_tx_id:
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    continue
                                if staged_tx_id != tx_id:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                                    continue
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                generation = payload.get("brain_generation", generation + 1)
                                if staged_state is not None:
                                    if 'parent_model_backup' in locals() and parent_model_backup is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in parent_model_backup.parameters()):
                                        topology_version += 1 
                                    live_state = staged_state
                                    if staged_opt is not None: optimizer = staged_opt
                                    fitness += staged_fitness_inc
                                staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                last_committed_tx_id = tx_id
                                is_dirty = False; last_save = time.monotonic()
                                _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                CheckpointManager.discard_candidate("M5_Amygdala", tx_id)
                                if staged_tx_id == tx_id or locals().get('last_committed_tx_id') == tx_id:
                                    if locals().get('last_committed_tx_id') == tx_id:
                                        last_committed_tx_id = None
                                    if 'parent_gen_backup' in locals(): generation = parent_gen_backup
                                    if 'parent_topo_ver_backup' in locals(): topology_version = parent_topo_ver_backup
                                    if 'parent_fit_backup' in locals(): fitness = parent_fit_backup
                                    if 'parent_model_backup' in locals() and parent_model_backup is not None:
                                        live_state = copy.deepcopy(parent_model_backup)
                                        if parent_opt_backup is not None:
                                            optimizer = type(optimizer)(filter(lambda p: p.requires_grad, live_state.parameters()), **optimizer.defaults)
                                            try: optimizer.load_state_dict(parent_opt_backup)
                                            except Exception: pass
                                    staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif topic == "ENV_HAZARD":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict): current_danger_label = int(payload.get("is_danger", 0))
                        elif not is_sleeping and topic in ["RAW_VISION_M1B", "RAW_AUDIO"]:
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            if isinstance(payload, dict):
                                try:
                                    td_gain = 1.0 + float(lateral_topdown.get("m1b_focus", 0.0)) if isinstance(locals().get("lateral_topdown"), dict) else 1.0
                                    new_frame = (np.array(numeric_encoder(payload, 20), dtype=np.float32) * td_gain).tolist()
                                    m5_history.append(new_frame)
                                    if len(m5_history) > 10: m5_history.pop(0)
                                    pad_len = 10 - len(m5_history)
                                    history_list = [m5_history[0]] * pad_len + m5_history if pad_len > 0 else m5_history
                                    
                                    current_trace = torch.tensor(history_list, dtype=torch.float32).unsqueeze(1) # (10, 1, 20)
                                    
                                    if eligibility_trace is None: eligibility_trace = current_trace
                                    else: eligibility_trace = 0.9 * eligibility_trace.detach() + 0.1 * current_trace
                                    live_state.eval()
                                    with torch.no_grad():
                                        spikes = live_state(eligibility_trace)
                                        last_m5_tensor = spikes.sum(dim=0)
                                        raw_prob = float(torch.softmax(last_m5_tensor, dim=-1)[0, 1].item())
                                        safe_spk = float(last_m5_tensor[0, 0].item())
                                        danger_spk = float(last_m5_tensor[0, 1].item())
                                        if current_danger_label == 0 and danger_spk <= safe_spk:
                                            spike_gate = min(1.0, max(0.0, danger_spk - safe_spk) / 2.0)
                                            threat_prob = raw_prob * spike_gate
                                        else:
                                            threat_prob = raw_prob
                                    live_state.train()
                                    def m5_live_loss(model, x):
                                        out_t = model(x).sum(dim=0)
                                        if out_t.dim() == 1: out_t = out_t.unsqueeze(0)
                                        target = torch.tensor([current_danger_label], dtype=torch.long, device=out_t.device)
                                        return torch.nn.functional.cross_entropy(out_t, target)
                                    m5_mut = lambda net, opt=None: net.sprout_new_neuron(optimizer=opt) if hasattr(net, 'sprout_new_neuron') else None
                                    if staged_tx_id is None:
                                        base_m5 = staged_state if staged_state is not None else live_state
                                        base_m5_opt = staged_opt if staged_opt is not None else optimizer
                                        is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m5, eligibility_trace, m5_live_loss, base_m5_opt, rpe=0.0, mutate_func=m5_mut)
                                        if is_valid: 
                                            staged_state, staged_opt = new_staged, new_opt
                                    inhibition_str = float(np.clip((threat_prob - 0.5) * 2.0, 0.0, 1.0))
                                    
                                    if not hasattr(live_state, 'consecutive_threats'):
                                        live_state.consecutive_threats = 0
                                        
                                    if threat_prob > 0.8 or current_danger_label == 1:
                                        live_state.consecutive_threats += 1
                                    else:
                                        live_state.consecutive_threats = 0
                                        
                                    is_emergency = 1 if live_state.consecutive_threats >= 3 else 0
                                    if is_emergency == 1:
                                        threat_prob = max(threat_prob, 0.85)
                                        inhibition_str = max(inhibition_str, 0.85)
                                    was_emergency = getattr(live_state, 'in_emergency_state', False)
                                    
                                    m5_payload = { "threat_vector": last_m5_tensor.detach().tolist()[0], "threat_probability": threat_prob, "inhibition_strength": inhibition_str, "emergency_priority": is_emergency }
                                    bus.publish("AMYGDALA_THREAT", m5_payload, modality="threat", parent_event_id=parent_id, episode_id=local_episode)
                                    if is_emergency == 1 and not was_emergency:
                                        live_state.in_emergency_state = True
                                        bus.publish("SYSTEM_CONTROL", {"state": "EMERGENCY_STOP", "source": "M5_Amygdala"}, modality="control", episode_id=local_episode)
                                    elif is_emergency == 0 and was_emergency:
                                        live_state.in_emergency_state = False
                                        bus.publish("SYSTEM_CONTROL", {"state": "AWAKE_AND_SAFE", "source": "M5_Amygdala"}, modality="control", episode_id=local_episode)
                                except Exception as e:
                                    if diagnose_error(e, "M5_Amygdala") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif is_sleeping and topic == "DREAM_REPLAY":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "vector" in payload:
                                try:
                                    dream_frame = numeric_encoder(payload["vector"], 20)
                                    dream_seq = torch.tensor([dream_frame] * 10, dtype=torch.float32).unsqueeze(1)
                                    with torch.no_grad(): _ = live_state(dream_seq)
                                except Exception as e:
                                    if diagnose_error(e, "M5_Dream") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif topic == "THALAMUS_TOPDOWN":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "topdown_attention" in payload:
                                lateral_topdown = payload["topdown_attention"]
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                novelty_val = float(dopa_payload.get("novelty", 0.0))
                                local_plasticity = compute_local_plasticity("M5_Amygdala", dopa_payload)
                                if abs(local_plasticity) > 0.05 and eligibility_trace is not None:
                                    try:
                                        implicit_target = 1 if local_plasticity > 0 else 0
                                        def m5_dopa_loss(model, x):
                                            out_t = model(x).sum(dim=0)
                                            if out_t.dim() == 1: out_t = out_t.unsqueeze(0)
                                            target = torch.tensor([implicit_target], dtype=torch.long, device=out_t.device)
                                            return torch.nn.functional.cross_entropy(out_t, target)
                                        m5_mut = lambda net, opt=None: net.sprout_new_neuron(optimizer=opt) if hasattr(net, "sprout_new_neuron") else None
                                        if staged_tx_id is None:
                                            base_m5 = staged_state if staged_state is not None else live_state
                                            base_m5_opt = staged_opt if staged_opt is not None else optimizer
                                            is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m5, eligibility_trace, m5_dopa_loss, base_m5_opt, rpe=local_plasticity, novelty=novelty_val, mutate_func=m5_mut)
                                            if is_valid:
                                                staged_state, staged_opt = new_staged, new_opt
                                                staged_fitness_inc = local_plasticity
                                    except Exception as e:
                                        if diagnose_error(e, "M5_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
            current_time = time.monotonic()
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M5_Amygdala'})}")
                _publish_heartbeat(bus, "M5_Amygdala", local_episode)
                last_heartbeat = current_time
            if is_dirty and current_time - last_save > 60 and can_worker_write_canonical("M5_Amygdala", staged_tx_id):
                extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_hidden": getattr(live_state, 'num_hidden', 64)}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
                CheckpointManager.save("M5_Amygdala", live_state, optimizer, is_torch=True, extra_states=extra_st); is_dirty = False; last_save = current_time
    finally:
        if is_dirty and can_worker_write_canonical("M5_Amygdala", staged_tx_id):
            extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_hidden": getattr(live_state, 'num_hidden', 64)}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
            CheckpointManager.save("M5_Amygdala", live_state, optimizer, is_torch=True, extra_states=extra_st)
        bus.shutdown(); upstream_sock.close()
        
# ================= 6. M6_LANGUAGE =================
def m6_language_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M6_Language")
    spec = MODEL_SPECS["M6_Language"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID", "SEMANTICS", "MEM_RESULT", "VOCAL_FEEDBACK", "CMD_VOCAL", "THALAMUS_M6", "THALAMUS_TOPDOWN", "DREAM_REPLAY"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "03_Cognitive_Core")
    if code_dir not in sys.path: sys.path.append(code_dir)
    from importlib import import_module
    ProgressiveLiquidBrain = import_module("liquid_progressive_brain").ProgressiveLiquidBrain
    chkpt, load_mode = load_worker_checkpoint_strict("M6_Language", is_torch=True)
    live_state = ProgressiveLiquidBrain(input_dim=128, hidden_dim=128, output_dim=133, name="M6_Language")
    if chkpt:
        topology = chkpt.get("topology", {})
        state = chkpt.get("model_state", chkpt)
        try:
            live_state.load_state_dict(state, strict=True, topology_meta=topology)
        except Exception as e:
            logger.critical(f"💀 [RECOVERY_REQUIRED] M6_Language Mismatch! {e}")
            raise RuntimeError(f"RECOVERY_REQUIRED:M6_Language:{e}")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, live_state.parameters()), lr=1e-3)
        if chkpt.get("optimizer_state"):
            try: optimizer.load_state_dict(chkpt["optimizer_state"])
            except Exception: pass
        generation, fitness, topology_version = chkpt.get("generation", 0), chkpt.get("fitness", 0.0), chkpt.get("topology_version", chkpt.get("architecture_version", 1))
    else:
        if hasattr(live_state, 'sprout_new_skill_column') and len(getattr(live_state, 'columns', [])) < 2:
            live_state.sprout_new_skill_column()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, live_state.parameters()), lr=1e-3)
        topology_version, generation, fitness = 1, 0, 0.0
    parent_model_backup = None; parent_state_backup = None; parent_opt_backup = None
    staged_state = None; staged_opt = None; staged_tx_id = None; staged_fitness_inc = 0.0
    live_state.train(); is_dirty, is_sleeping, last_heartbeat = False, True, time.monotonic()
    eligibility_trace, last_save = None, time.monotonic()
    local_episode = "UNKNOWN"; lateral_mem = None; lateral_vocal_fb = None; lateral_cmd_vocal = None; lateral_topdown = None
    last_m6_cross_inputs = None
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M6_Language', 'schema_verified': True, 'load_mode': load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M6_Language", local_episode)
    local_tick = 0
    
    try:
        while not stop_event.is_set():
            
            socks = dict(poller.poll(50))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M6_Language", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", generation + 1)
                                if staged_tx_id != tx_id:
                                    if staged_tx_id is not None:
                                        staged_state, staged_opt, staged_fitness_inc = None, None, 0.0
                                    staged_tx_id = tx_id
                                    parent_gen_backup = generation
                                    parent_topo_ver_backup = topology_version
                                    parent_fit_backup = fitness
                                    parent_model_backup = copy.deepcopy(live_state)
                                    parent_state_backup = copy.deepcopy(live_state.state_dict())
                                    parent_opt_backup = copy.deepcopy(optimizer.state_dict())
                                cand_ref = staged_state if staged_state is not None else live_state
                                topo_d = {"num_columns": len(getattr(cand_ref, 'columns', [1]))}
                                cand_topo_ver = topology_version + (1 if (staged_state is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in live_state.parameters())) else 0)
                                h = stage_pytorch_candidate_file("M6_Language", live_state, staged_state, optimizer, staged_opt, tx_id, generation, target_gen, fitness + staged_fitness_inc, cand_topo_ver, topo_d)
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M6_Language', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M6_Language", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                if 'last_committed_tx_id' in locals() and tx_id == last_committed_tx_id:
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    continue
                                if staged_tx_id != tx_id:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                                    continue
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                generation = payload.get("brain_generation", generation + 1)
                                if staged_state is not None:
                                    if parent_model_backup is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in parent_model_backup.parameters()):
                                        topology_version += 1 
                                    live_state = staged_state
                                    if staged_opt is not None: optimizer = staged_opt
                                    fitness += staged_fitness_inc
                                staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                last_committed_tx_id = tx_id
                                is_dirty = False; last_save = time.monotonic()
                                _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                CheckpointManager.discard_candidate("M6_Language", tx_id)
                                if staged_tx_id == tx_id or locals().get('last_committed_tx_id') == tx_id:
                                    if locals().get('last_committed_tx_id') == tx_id:
                                        last_committed_tx_id = None
                                    if 'parent_gen_backup' in locals(): generation = parent_gen_backup
                                    if 'parent_topo_ver_backup' in locals(): topology_version = parent_topo_ver_backup
                                    if 'parent_fit_backup' in locals(): fitness = parent_fit_backup
                                    if parent_model_backup is not None:
                                        live_state = copy.deepcopy(parent_model_backup)
                                        if parent_opt_backup is not None:
                                            optimizer = type(optimizer)(filter(lambda p: p.requires_grad, live_state.parameters()), **optimizer.defaults)
                                            try: optimizer.load_state_dict(parent_opt_backup)
                                            except Exception: pass
                                    staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif topic == "MEM_RESULT":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict): lateral_mem = payload.get("retrieved_memory", "")
                        elif topic == "VOCAL_FEEDBACK": lateral_vocal_fb = packet.get("payload", packet) if isinstance(packet, dict) else packet
                        elif topic == "CMD_VOCAL":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict): lateral_cmd_vocal = payload.get("action", "")
                        elif not is_sleeping and topic == "THALAMUS_M6":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            if isinstance(packet, dict):
                                try:
                                    fused = payload.get("fused_latent", [])
                                    # [STEP 1.6 BUG #2 FIX]: Strict canonical contract — reject non-128D fused_latent
                                    if not isinstance(fused, (list, np.ndarray)) or len(fused) != 128:
                                        logger.warning(f"⚠️ [M6] Rejected invalid fused_latent: expected 128D, got {len(fused) if isinstance(fused, (list, np.ndarray)) else type(fused).__name__}. Skipping THALAMUS_M6 payload.")
                                        continue
                                    # [STEP 1.6 BUG #3 FIX]: Reject NaN/Inf in fused_latent at M6 boundary
                                    fused_arr = np.array(fused, dtype=np.float32)
                                    if not np.all(np.isfinite(fused_arr)):
                                        logger.warning(f"⚠️ [M6] Rejected fused_latent containing NaN/Inf. Skipping THALAMUS_M6 payload.")
                                        continue
                                    thalamus_x = torch.tensor([fused], dtype=torch.float32)
                                    mem_x = torch.tensor([numeric_encoder(lateral_mem, 39)], dtype=torch.float32)
                                    voc_x = torch.tensor([numeric_encoder(lateral_vocal_fb, 2)], dtype=torch.float32)
                                    cmd_x = torch.tensor([numeric_encoder(lateral_cmd_vocal, 10)], dtype=torch.float32)
                                    if eligibility_trace is None: eligibility_trace = thalamus_x
                                    else: eligibility_trace = 0.9 * eligibility_trace.detach() + 0.1 * thalamus_x
                                    cross_inputs = {"memory": mem_x, "vocal_fb": voc_x, "logic_cmd": cmd_x}
                                    last_m6_cross_inputs = cross_inputs
                                    live_state.eval()
                                    with torch.no_grad():
                                        last_out_tensor = live_state(eligibility_trace, cross_modal_inputs=cross_inputs, integration_time=1.0)
                                        intent_logits = last_out_tensor[0, :5]
                                        intent_idx = torch.argmax(intent_logits).item()
                                        semantic_embedding = last_out_tensor[0, 5:].tolist()
                                    live_state.train()
                                    intent_labels = ["GREETING", "WARNING", "COMMAND", "QUESTION", "NEUTRAL"]
                                    real_intent_idx = 1 if float(payload.get("amygdala_inhibition", 0.0)) > 0.5 else (0 if float(payload.get("salience", 0.0)) > 0.1 else 4)
                                    if lateral_cmd_vocal:
                                        raw_cmd = lateral_cmd_vocal.replace("_APPROVED", "")
                                        if raw_cmd in intent_labels: real_intent_idx = intent_labels.index(raw_cmd)
                                    if intent_idx == 4 and real_intent_idx != 4:
                                        intent_idx = int(torch.argmax(intent_logits[:4]).item())

                                    target_intent = torch.tensor([real_intent_idx], dtype=torch.long)
                                    target_semantic = torch.tensor([embedding_encoder(payload.get("fused_latent", []), 128)], dtype=torch.float32)

                                    def m6_live_loss(model, x):
                                        out = model(x, cross_modal_inputs=cross_inputs, integration_time=1.0)
                                        loss_intent = F.cross_entropy(out[0, :5].unsqueeze(0), target_intent.to(out.device))
                                        loss_semantic = F.mse_loss(out[0, 5:], target_semantic.to(out.device)[0])
                                        return loss_intent + 0.5 * loss_semantic
                                    m6_mut = lambda net, opt=None: net.sprout_new_skill_column(optimizer=opt) if hasattr(net, 'sprout_new_skill_column') else None
                                    if staged_tx_id is None:
                                        base_m6 = staged_state if staged_state is not None else live_state
                                        base_m6_opt = staged_opt if staged_opt is not None else optimizer
                                        is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m6, eligibility_trace, m6_live_loss, base_m6_opt, rpe=0.0, mutate_func=m6_mut)
                                        if is_valid:
                                            staged_state, staged_opt = new_staged, new_opt
                                    m6_payload = { "semantic_vector": semantic_embedding, "semantic_embedding": semantic_embedding, "attention_context": payload, "inferred_intent": intent_labels[intent_idx] }
                                    bus.publish("SEMANTICS", m6_payload, modality="semantics", parent_event_id=parent_id, episode_id=local_episode)
                                    if intent_idx != 4:
                                        vocal_payload = {"task_id": f"VOC_{uuid.uuid4().hex[:8]}", "action": intent_labels[intent_idx]}
                                        bus.publish("VOCAL_INTENT", vocal_payload, modality="vocal_cmd", parent_event_id=parent_id, episode_id=local_episode)
                                except Exception as e:
                                    if diagnose_error(e, "M6_Language") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif topic == "THALAMUS_TOPDOWN":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "topdown_attention" in payload:
                                lateral_topdown = payload["topdown_attention"]
                        
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                novelty_val = float(dopa_payload.get("novelty", 0.0))
                                local_plasticity = compute_local_plasticity("M6_Language", dopa_payload)
                                if abs(local_plasticity) > 0.05 and eligibility_trace is not None:
                                    try:
                                        target_semantic_dopa = torch.tensor([embedding_encoder(dopa_payload, 128)], dtype=torch.float32)
                                        def m6_dopa_loss(model, x):
                                            out = model(x, cross_modal_inputs=last_m6_cross_inputs, integration_time=1.0)
                                            loss_intent = F.cross_entropy(out[0, :5].unsqueeze(0), torch.tensor([4], dtype=torch.long, device=out.device)) 
                                            loss_semantic = F.mse_loss(out[0, 5:], target_semantic_dopa.to(x.device)[0])
                                            return loss_intent + 0.5 * loss_semantic
                                        m6_mut = lambda net, opt=None: net.sprout_new_skill_column(optimizer=opt) if hasattr(net, "sprout_new_skill_column") else None
                                        if staged_tx_id is None:
                                            base_m6 = staged_state if staged_state is not None else live_state
                                            base_m6_opt = staged_opt if staged_opt is not None else optimizer
                                            is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m6, eligibility_trace, m6_dopa_loss, base_m6_opt, rpe=local_plasticity, novelty=novelty_val, mutate_func=m6_mut)
                                            if is_valid:
                                                staged_state, staged_opt = new_staged, new_opt
                                                staged_fitness_inc = local_plasticity
                                    except Exception as e:
                                        if diagnose_error(e, "M6_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif is_sleeping and topic == "DREAM_REPLAY":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "vector" in payload:
                                try: live_state(torch.tensor([numeric_encoder(payload["vector"][:128], 128)], dtype=torch.float32), integration_time=0.5)
                                except Exception as e:
                                    if diagnose_error(e, "M6_Dream") == BrainErrorCategory.FATAL: stop_event.set(); break
            current_time = time.monotonic()
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M6_Language'})}")
                _publish_heartbeat(bus, "M6_Language", local_episode)
                last_heartbeat = current_time
            if is_dirty and current_time - last_save > 60 and can_worker_write_canonical("M6_Language", staged_tx_id):
                extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_columns": len(getattr(live_state, 'columns', [1]))}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
                CheckpointManager.save("M6_Language", live_state, optimizer, is_torch=True, extra_states=extra_st); is_dirty = False; last_save = current_time
    finally:
        if is_dirty and can_worker_write_canonical("M6_Language", staged_tx_id):
            extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_columns": len(getattr(live_state, 'columns', [1]))}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
            CheckpointManager.save("M6_Language", live_state, optimizer, is_torch=True, extra_states=extra_st)
        bus.shutdown(); upstream_sock.close()

# ================= 7. M7_LOGIC =================
def m7_prefrontal_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M7_Logic")
    spec = MODEL_SPECS["M7_Logic"]
    poller = zmq.Poller()
    bus.subscribe(spec.get("input_contract", []) + ["SYSTEM_CONTROL", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID"]); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "03_Cognitive_Core")
    if code_dir not in sys.path: sys.path.append(code_dir)
    from importlib import import_module
    ProgressiveLiquidBrain = import_module("liquid_progressive_brain").ProgressiveLiquidBrain
    chkpt, load_mode = load_worker_checkpoint_strict("M7_Logic", is_torch=True)
    live_state = ProgressiveLiquidBrain(input_dim=128, hidden_dim=64, output_dim=6, name="M7_Prefrontal")
    if chkpt:
        topology = chkpt.get("topology", {})
        state = chkpt.get("model_state", chkpt)
        try:
            live_state.load_state_dict(state, strict=True, topology_meta=topology)
        except Exception as e:
            logger.critical(f"💀 [RECOVERY_REQUIRED] M7_Logic Mismatch! {e}")
            raise RuntimeError(f"RECOVERY_REQUIRED:M7_Logic:{e}")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, live_state.parameters()), lr=1e-3)
        if chkpt.get("optimizer_state"):
            try: optimizer.load_state_dict(chkpt["optimizer_state"])
            except Exception: pass
        generation, fitness, topology_version = chkpt.get("generation", 0), chkpt.get("fitness", 0.0), chkpt.get("topology_version", chkpt.get("architecture_version", 1))
    else:
        if hasattr(live_state, 'sprout_new_skill_column') and len(getattr(live_state, 'columns', [])) < 2:
            live_state.sprout_new_skill_column()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, live_state.parameters()), lr=1e-3)
        topology_version, generation, fitness = 1, 0, 0.0
    parent_model_backup = None; parent_state_backup = None; parent_opt_backup = None
    staged_state = None; staged_opt = None; staged_tx_id = None; staged_fitness_inc = 0.0
    live_state.train()
    working_memory = []; pending_tasks = {}
    is_dirty, is_sleeping, planner_inhibited = False, True, False
    last_task_time, last_heartbeat = time.monotonic(), time.monotonic()
    total_episodes, local_tick, local_episode = 0, 0, "UNKNOWN"; lateral_topdown = None
    current_hormones = {}; local_credit_ledger = {}
    global_replay = ReplayBuffer(capacity=5000)
    last_save = time.monotonic()
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M7_Logic', 'schema_verified': True, 'load_mode': load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M7_Logic", local_episode)
    local_tick = 0
    
    try:
        while not stop_event.is_set():
            
            socks = dict(poller.poll(50))
            current_time = time.monotonic()
            for tid in list(pending_tasks.keys()):
                if current_time - pending_tasks[tid].get("timestamp", current_time) > 5.0: del pending_tasks[tid]
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M7_Logic", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                            if "payload" in packet and isinstance(packet["payload"], dict) and "hormonal_state" in packet["payload"]:
                                current_hormones = packet["payload"]["hormonal_state"]
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping, planner_inhibited = False, False
                            elif st == "EMERGENCY_STOP": planner_inhibited = True; pending_tasks.clear(); working_memory.clear()
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", generation + 1)
                                if staged_tx_id != tx_id:
                                    if staged_tx_id is not None:
                                        staged_state, staged_opt, staged_fitness_inc = None, None, 0.0
                                    staged_tx_id = tx_id
                                    parent_gen_backup = generation
                                    parent_topo_ver_backup = topology_version
                                    parent_fit_backup = fitness
                                    parent_model_backup = copy.deepcopy(live_state)
                                    parent_state_backup = copy.deepcopy(live_state.state_dict())
                                    parent_opt_backup = copy.deepcopy(optimizer.state_dict())
                                cand_ref = staged_state if staged_state is not None else live_state
                                topo_d = {"num_columns": len(getattr(cand_ref, 'columns', [1]))}
                                cand_topo_ver = topology_version + (1 if (staged_state is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in live_state.parameters())) else 0)
                                h = stage_pytorch_candidate_file("M7_Logic", live_state, staged_state, optimizer, staged_opt, tx_id, generation, target_gen, fitness + staged_fitness_inc, cand_topo_ver, topo_d)
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M7_Logic', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M7_Logic", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                if 'last_committed_tx_id' in locals() and tx_id == last_committed_tx_id:
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    continue
                                if staged_tx_id != tx_id:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                                    continue
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                generation = payload.get("brain_generation", generation + 1)
                                if staged_state is not None:
                                    if parent_model_backup is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in parent_model_backup.parameters()):
                                        topology_version += 1 
                                    live_state = staged_state
                                    if staged_opt is not None: optimizer = staged_opt
                                    fitness += staged_fitness_inc
                                staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                last_committed_tx_id = tx_id
                                is_dirty = False; last_save = time.monotonic()
                                _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                CheckpointManager.discard_candidate("M7_Logic", tx_id)
                                if staged_tx_id == tx_id or locals().get('last_committed_tx_id') == tx_id:
                                    if locals().get('last_committed_tx_id') == tx_id:
                                        last_committed_tx_id = None
                                    if 'parent_gen_backup' in locals(): generation = parent_gen_backup
                                    if 'parent_topo_ver_backup' in locals(): topology_version = parent_topo_ver_backup
                                    if 'parent_fit_backup' in locals(): fitness = parent_fit_backup
                                    if parent_model_backup is not None:
                                        live_state = copy.deepcopy(parent_model_backup)
                                        if parent_opt_backup is not None:
                                            optimizer = type(optimizer)(filter(lambda p: p.requires_grad, live_state.parameters()), **optimizer.defaults)
                                            try: optimizer.load_state_dict(parent_opt_backup)
                                            except Exception: pass
                                    staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                            elif st == "UPDATE_CREDIT": local_credit_ledger = packet.get("credit", {})
                            if "episode_id" in packet and packet["episode_id"] != "UNKNOWN": local_episode = packet["episode_id"]
                        elif topic == "AMYGDALA_THREAT":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict): current_hormones["threat"] = payload.get("threat_probability", 0.0)
                        elif topic == "THALAMUS_TOPDOWN":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "topdown_attention" in payload:
                                lateral_topdown = payload["topdown_attention"]
                        elif not is_sleeping and not planner_inhibited:
                            if topic == "SEMANTICS":
                                payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                                parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                                if isinstance(payload, dict):
                                    m7_think_id = f"M7_THINK_{uuid.uuid4().hex[:8]}"
                                    bus.publish("QUERY_MEM", {"task_id": m7_think_id, "context": payload}, modality="memory_query", parent_event_id=parent_id, episode_id=local_episode)
                            elif topic == "MEM_RESULT":
                                payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                                parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                                if isinstance(payload, dict): working_memory.append({"snapshot": payload, "salience": 1.0, "timestamp": time.monotonic(), "parent_id": parent_id})
                            elif topic == "THALAMUS_M7":
                                payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                                parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                                if isinstance(payload, dict):
                                    snap = payload.copy()
                                    snap["topdown_focus"] = lateral_topdown if lateral_topdown is not None else {}
                                    thal_salience = max(1.2, float(payload.get("salience", 0.5)) + 1.0)
                                    working_memory.append({"snapshot": snap, "salience": thal_salience, "timestamp": time.monotonic(), "parent_id": parent_id})
                                    if payload.get("amygdala_inhibition", 0.0) > 0.8:
                                        working_memory.append({"snapshot": {"type": "THREAT_ALERT", "threat_level": payload.get("amygdala_inhibition")}, "salience": 5.0, "timestamp": time.monotonic(), "parent_id": parent_id})
                            elif topic == "VOCAL_INTENT":
                                payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                                parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                                if isinstance(payload, dict):
                                    v_task_id = payload.get("task_id", f"VOC_{uuid.uuid4().hex[:8]}")
                                    
                                    state_snap = working_memory[0]["snapshot"] if working_memory else {}
                                    tensor_in = torch.tensor([numeric_encoder(state_snap, 128)], dtype=torch.float32)
                                    live_state.eval()
                                    with torch.no_grad():
                                        out = live_state(tensor_in, integration_time=1.0)
                                    action_logits = out[0, :5]; value_pred = torch.tanh(out[0, 5])
                                    dist = torch.distributions.Categorical(logits=action_logits)
                                    action_idx = dist.sample()
                                    log_prob = dist.log_prob(action_idx)
                                    live_state.train()
                                    
                                    intent_labels = ["GREETING", "WARNING", "COMMAND", "QUESTION", "NEUTRAL"]
                                    sampled_action_str = intent_labels[action_idx.item()]
                                    
                                    pending_tasks[v_task_id] = { "task_id": v_task_id, "action": sampled_action_str, "status": "PENDING", "type": "VOCAL",
                                        "action_idx": action_idx, "confidence": value_pred.item(), "value_tensor": value_pred,
                                        "log_prob": log_prob, "timestamp": time.monotonic(),
                                        "state_snapshot": state_snap, "parent_id": parent_id }
                                    bus.publish("CMD_VOCAL", {"task_id": v_task_id, "action": f"{sampled_action_str}_APPROVED"}, modality="vocal_cmd", parent_event_id=parent_id, episode_id=local_episode)
                            elif topic == "MOTOR_FEEDBACK":
                                payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                                if isinstance(payload, dict) and payload.get("task_id") in pending_tasks:
                                    if "reward" in payload:
                                        pending_tasks[payload["task_id"]]["ext_reward"] = float(payload["reward"])
                                        pending_tasks[payload["task_id"]]["status"] = "DONE"
                            elif topic == "VOCAL_FEEDBACK":
                                payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                                if isinstance(payload, dict) and payload.get("task_id") in pending_tasks:
                                    ac = payload.get("acoustic_output", [0.5, 0.5])
                                    vol = float(ac[1]) if isinstance(ac, list) and len(ac) > 1 else 0.0
                                    pending_tasks[payload["task_id"]].setdefault("ext_reward", 0.1 if vol > 0.1 else 0.0)
                                    pending_tasks[payload["task_id"]]["status"] = "DONE"
                            elif topic == "ENV_REWARD":
                                payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                                if packet.get("source") != "Sandbox_Core":
                                    logger.warning(f"⚠️ [M7_Logic] Từ chối ENV_REWARD giả mạo từ nguồn: {packet.get('source')}")
                                    continue
                                if isinstance(payload, dict) and payload.get("task_id") in pending_tasks:
                                    pending_tasks[payload["task_id"]]["ext_reward"] = float(payload.get("reward", 0.5))
                                    pending_tasks[payload["task_id"]]["status"] = "DONE"
                        elif is_sleeping and topic == "DREAM_REPLAY":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "vector" in payload:
                                global_replay.push(payload["vector"][:128], "DREAM", 0.1, payload["vector"][:128], True)
            
            for t_id in list(pending_tasks.keys()):
                task_info = pending_tasks[t_id]
                if task_info.get("status") == "DONE":
                    reward = float(np.clip(task_info.get("ext_reward", 0.0), -1.0, 1.0))
                    predicted_value = task_info.get("confidence", 0.0)
                    rpe = float(np.clip(reward - predicted_value, -2.0, 2.0))
                    log_prob, value_tensor = task_info.get("log_prob"), task_info.get("value_tensor")
                    if live_state is not None and log_prob is not None and value_tensor is not None:
                        current_state = task_info.get("state_snapshot")
                        next_state = working_memory[0]["snapshot"] if working_memory else current_state
                        global_replay.push(current_state, task_info.get("action"), reward, next_state, True)
                        neuromodulation_payload = { "signal": "NEUROMODULATION", "task_id": t_id, "episode_id": local_episode,
                            "reward": reward, "rpe": rpe, "threat": current_hormones.get("threat", 0.0),
                            "novelty": current_hormones.get("novelty", 0.0), "salience": current_hormones.get("salience", 0.0),
                            "arousal": current_hormones.get("adrenaline", 0.0), "sleep_state": 1.0 if is_sleeping else 0.0,
                            "credit": local_credit_ledger }
                        local_plasticity = compute_local_plasticity("M7_Logic", neuromodulation_payload)
                        novelty_val = float(current_hormones.get("novelty", 0.0))
                        if abs(local_plasticity) > 0.05:
                            try:
                                eligibility_trace = torch.tensor([numeric_encoder(current_state, 128)], dtype=torch.float32)
                                act_val = task_info.get("action_idx", 0)
                                if torch.is_tensor(act_val): act_val = act_val.item()
                                target_act = torch.tensor(int(act_val), dtype=torch.long)
                                cross_m7 = task_info.get("cross_inputs", None)
                                def m7_online_loss(model, x):
                                    out = model(x, cross_modal_inputs=cross_m7, integration_time=1.0) # [BẢN VÁ P0]: M7 học qua Projector
                                    act_l = -torch.distributions.Categorical(logits=out[0, :5]).log_prob(target_act.to(out.device)) * local_plasticity
                                    return act_l + torch.nn.functional.mse_loss(torch.tanh(out[0, 5]), torch.tensor(reward, dtype=torch.float32, device=out.device))
                                m7_mut = lambda net, opt=None: net.sprout_new_skill_column(optimizer=opt) if hasattr(net, 'sprout_new_skill_column') else None
                                if staged_tx_id is None:
                                    base_m7 = staged_state if staged_state is not None else live_state
                                    base_m7_opt = staged_opt if staged_opt is not None else optimizer
                                    is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m7, eligibility_trace, m7_online_loss, base_m7_opt, rpe=local_plasticity, novelty=novelty_val, mutate_func=m7_mut)
                                    if is_valid:
                                        staged_state, staged_opt = new_staged, new_opt
                                        staged_fitness_inc = local_plasticity
                            except Exception as e:
                                if diagnose_error(e, "M7_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
                        # [BẢN VÁ KIẾN TRÚC]: M7 tiết Dopamine gửi về M0, M0 mới là người phân phối Plasticity!
                        bus.publish("GLOBAL_DOPAMINE", neuromodulation_payload, modality="dopamine", parent_event_id=task_info.get("parent_id"), episode_id=local_episode)
                    total_episodes += 1; del pending_tasks[t_id]
            has_pending_motor = any(t.get("type", "MOTOR") == "MOTOR" for t in pending_tasks.values())
            if not is_sleeping and not planner_inhibited and len(working_memory) > 0 and not has_pending_motor and current_time - last_task_time > 0.5:
                for item in working_memory: 
                    item["salience"] *= np.exp(-0.5 * (current_time - item["timestamp"]))
                    item["timestamp"] = current_time
                working_memory = sorted([i for i in working_memory if i["salience"] > 0.1], key=lambda x: x["salience"], reverse=True)[:5]
                if working_memory:
                    task_id = f"TSK_{uuid.uuid4().hex[:8]}"
                    try:
                        tensor_in = torch.tensor([numeric_encoder(working_memory[0]["snapshot"], 128)], dtype=torch.float32)
                        threat_x = torch.tensor([numeric_encoder(current_hormones.get("threat", 0.0), 2)], dtype=torch.float32)
                        mem_val = working_memory[0]["snapshot"].get("retrieved_memory", "") if isinstance(working_memory[0]["snapshot"], dict) else ""
                        mem_x = torch.tensor([numeric_encoder(mem_val, 39)], dtype=torch.float32)
                        cross_m7 = {"amygdala": threat_x, "memory": mem_x}

                        live_state.eval()
                        with torch.no_grad():
                            out = live_state(tensor_in, cross_modal_inputs=cross_m7, integration_time=1.0)
                        action_logits = out[0, :5]; value_pred = torch.tanh(out[0, 5])
                        dist = torch.distributions.Categorical(logits=action_logits)
                        action_idx = dist.sample()
                        live_state.train()
                        wm_parents = [str(item.get("parent_id")) for item in working_memory if item.get("parent_id")]
                        combined_wm_parent = ",".join(dict.fromkeys(wm_parents[:5])) if wm_parents else working_memory[0].get("parent_id")
                        pending_tasks[task_id] = { "task_id": task_id, "action": f"Action_{action_idx.item()}", "type": "MOTOR", "action_idx": action_idx, "confidence": value_pred.item(),
                            "value_tensor": value_pred, "status": "PENDING", "log_prob": dist.log_prob(action_idx), "timestamp": current_time,
                            "state_snapshot": working_memory[0]["snapshot"], "cross_inputs": cross_m7, "parent_id": combined_wm_parent }
                        cmd_payload = {"task_id": task_id, "action": f"Action_{action_idx.item()}", "state_snapshot": working_memory[0]["snapshot"]}
                        bus.publish("CMD_MOTOR", cmd_payload, modality="motor_cmd", parent_event_id=combined_wm_parent, episode_id=local_episode)
                        working_memory[0]["salience"] *= 0.01
                        last_task_time = current_time
                    except Exception as e:
                        if diagnose_error(e, "M7_Logic") == BrainErrorCategory.FATAL: stop_event.set(); break
            if is_sleeping and len(global_replay) >= 16 and current_time - last_task_time > 1.0:
                batch = global_replay.sample(8)
                if isinstance(batch, dict) and "rewards" in batch:
                    batch_size = len(batch["rewards"])
                    for i in range(batch_size):
                        try:
                            obs = batch["states"][i]; rew = float(batch["rewards"][i])
                            tensor_in = torch.tensor([numeric_encoder(obs, 128)], dtype=torch.float32)
                            out = live_state(tensor_in, integration_time=1.0)
                            val_pred = torch.tanh(out[0, 5])
                            offline_rpe = float(np.clip(rew - val_pred.item(), -2.0, 2.0))
                            dream_payload = { "signal": "DREAM_CONSOLIDATION", "task_id": "OFFLINE_REPLAY", "episode_id": local_episode,
                                "reward": rew, "rpe": offline_rpe, "threat": current_hormones.get("threat", 0.0),
                                "novelty": current_hormones.get("novelty", 0.0), "salience": current_hormones.get("salience", 0.0),
                                "arousal": current_hormones.get("adrenaline", 0.0), "sleep_state": 1.0, "credit": local_credit_ledger }
                            local_plasticity = compute_local_plasticity("M7_Logic", dream_payload)
                            novelty_val = float(current_hormones.get("novelty", 0.0))
                            if abs(local_plasticity) > 0.05:
                                def m7_dream_loss(model, x): return torch.nn.functional.mse_loss(torch.tanh(model(x, integration_time=1.0)[0, 5]), torch.tensor(rew, dtype=torch.float32))
                                m7_mut = lambda net, opt=None: net.sprout_new_skill_column(optimizer=opt) if hasattr(net, 'sprout_new_skill_column') else None
                                base_m7 = staged_state if staged_state is not None else live_state
                                base_m7_opt = staged_opt if staged_opt is not None else optimizer
                                is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m7, tensor_in, m7_dream_loss, base_m7_opt, rpe=local_plasticity, novelty=novelty_val, mutate_func=m7_mut)
                                if is_valid and staged_tx_id is None:
                                    staged_state = new_staged
                                    staged_opt = new_opt
                                    staged_fitness_inc = local_plasticity
                            bus.publish("GLOBAL_DOPAMINE", dream_payload, modality="dopamine", episode_id=local_episode)
                        except Exception as e:
                            if diagnose_error(e, "M7_Dream") == BrainErrorCategory.FATAL: stop_event.set(); break
                    last_task_time = current_time
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M7_Logic'})}")
                _publish_heartbeat(bus, "M7_Logic", local_episode)
                last_heartbeat = current_time
            if is_dirty and time.monotonic() - last_save > 60 and can_worker_write_canonical("M7_Logic", staged_tx_id):
                extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_columns": len(getattr(live_state, 'columns', [1]))}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
                CheckpointManager.save("M7_Logic", live_state, optimizer, is_torch=True, extra_states=extra_st); is_dirty = False; last_save = time.monotonic()
    finally:
        if is_dirty and can_worker_write_canonical("M7_Logic", staged_tx_id):
            extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_columns": len(getattr(live_state, 'columns', [1]))}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
            CheckpointManager.save("M7_Logic", live_state, optimizer, is_torch=True, extra_states=extra_st)
        bus.shutdown(); upstream_sock.close()

# ================= 8. M8_HIPPOCAMPUS =================
def m8_memory_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    epoch_file = os.path.join(BASE_DIR, "models", "M8_Memory", ".worker_epoch")
    if os.path.exists(epoch_file):
        try:
            with open(epoch_file, "r", encoding="utf-8") as ef:
                boot_epoch = int(ef.read().strip())
            os.environ["ASUNA_WORKER_EPOCH"] = str(boot_epoch)
            os.environ["ASUNA_WORKER_INSTANCE_ID"] = f"M8_Memory_ep{boot_epoch}_{os.getpid()}"
        except Exception:
            pass
    m8_dir = os.path.join(BASE_DIR, "04_Memory_System", "M8_Hippocampus")
    if m8_dir not in sys.path: sys.path.append(m8_dir)
    try:
        import m8_art_memory
        m8_art_memory.start_node(stop_event)
    except Exception as e:
        logger.error(f"❌ Lỗi khởi động M8 ngầm: {e}")
        raise

# ================= 9. M9_THALAMUS =================
def m9_thalamus_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M9_Thalamus")
    spec = MODEL_SPECS["M9_Thalamus"]
    poller = zmq.Poller()
    bus.subscribe(list(dict.fromkeys(spec.get("input_contract", []) + ["SYSTEM_CONTROL"]))); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "00_Central_Router/M9_Thalamus")
    if code_dir not in sys.path: sys.path.append(code_dir)
    INPUT_DIM = 128
    try: from m9_dynamic_som import DynamicSOM
    except ImportError as e: raise RuntimeError(f"🛑 Lỗi M9: {e}")
    chkpt, load_mode = load_worker_checkpoint_strict("M9_Thalamus", is_torch=True)
    if chkpt:
        topology = chkpt.get("topology", {}); num_nodes = topology.get("num_nodes", 4)
        state = chkpt.get("model_state", chkpt)
        if isinstance(state, dict) and "weights" in state:
            num_nodes = int(state["weights"].shape[0])
        live_state = DynamicSOM(input_dim=INPUT_DIM, initial_nodes=num_nodes)
        try: live_state.load_state_dict(state, strict=True)
        except Exception as e:
            logger.critical(f"💀 [RECOVERY_REQUIRED] M9_Thalamus Mismatch! {e}")
            raise RuntimeError(f"RECOVERY_REQUIRED:M9_Thalamus:{e}")
        generation = chkpt.get("generation", 0); fitness = chkpt.get("fitness", 0.0)
        topology_version = chkpt.get("topology_version", chkpt.get("architecture_version", 1)); step_count = chkpt.get("step_count", 0)
    else:
        live_state = DynamicSOM(input_dim=INPUT_DIM, initial_nodes=4)
        topology_version, generation, fitness = 1, 0, 0.0; step_count = 0
    if not hasattr(live_state, 'topology_version'): live_state.topology_version = topology_version
    parent_model_backup = copy.deepcopy(live_state)
    staged_tx_id = None
    is_dirty, is_sleeping, last_heartbeat = False, True, time.monotonic()
    local_tick = 0; local_episode = "UNKNOWN"; last_save, last_bmu_id = time.monotonic(), None
    latest_reward = 0.0; latest_rpe = 0.0; latest_novelty = 0.0; latest_task_success = 0.0
    prev_num_nodes = live_state.num_nodes
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M9_Thalamus', 'schema_verified': True, 'load_mode': load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M9_Thalamus", local_episode)
    current_context = { 
        "vision_where": [0.0]*10, "vision_what": [0.0]*10, "audio": [0.0]*10, 
        "spatial": [0.0]*10, "threat": [0.0]*10, "semantics": [0.0]*39, 
        "memory": [0.0]*39, "motor_fb": [0.0]*10, "vocal_fb": [0.0]*2, 
        "self_audio": [0.0]*10, "dream_replay": [0.0]*128 
    }
    
    # [STEP 1: CANONICAL 128D MODALITY-SEPARATED REPRESENTATION]
    modality_specs = {
        "vision_where": (10, 16), "vision_what": (10, 16), "audio": (10, 16),
        "spatial": (10, 8), "threat": (10, 4), "semantics": (39, 16),
        "memory": (39, 16), "motor_fb": (10, 8), "vocal_fb": (2, 4),
        "self_audio": (10, 8), "dream_replay": (128, 16)
    }
    m9_projectors = {}
    m9_rng = np.random.RandomState(42)
    for mod, (in_dim, out_dim) in modality_specs.items():
        mat = m9_rng.randn(out_dim, in_dim).astype(np.float32) / np.sqrt(in_dim)
        m9_projectors[mod] = mat
    inhibition_level = 0.0
    try:
        while not stop_event.is_set():
            
            socks = dict(poller.poll(50))
            has_new_data = False; parent_ids, task_ids = [], []
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M9_Thalamus", local_episode)
                        if isinstance(packet.get("tick_id"), int) and packet.get("tick_id", 0) > local_tick:
                            local_tick = packet["tick_id"]
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                master_tick = p_load.get("target_tick", local_tick)
                                master_ep = p_load.get("target_episode", local_episode)
                                target_gen = p_load.get("target_generation", generation + 1)
                                if staged_tx_id != tx_id:
                                    staged_tx_id = tx_id
                                    parent_gen_backup = generation
                                    parent_topo_ver_backup = topology_version
                                    parent_prev_nodes_backup = prev_num_nodes
                                    parent_model_backup = copy.deepcopy(live_state)
                                parent_hash = tensor_state_hash(live_state.state_dict())
                                cand_info = live_state.prepare_candidate(tx_id, target_tick=master_tick, target_episode=master_ep)
                                base_sd = live_state.state_dict()
                                raw_cand = getattr(live_state, "candidate_state", None)
                                if hasattr(raw_cand, "state_dict"):
                                    cand_sd = raw_cand.state_dict()
                                elif isinstance(raw_cand, dict):
                                    cand_sd = {k: raw_cand[k] for k in base_sd.keys() if k in raw_cand and isinstance(raw_cand[k], torch.Tensor)}
                                    if set(cand_sd.keys()) != set(base_sd.keys()):
                                        cand_sd = base_sd
                                else:
                                    cand_sd = base_sd
                                raw_h = tensor_state_hash(cand_sd)
                                is_unchanged = str(cand_info.get("candidate_hash", "")).startswith("UNCHANGED")
                                h = f"UNCHANGED_{raw_h}" if is_unchanged else raw_h
                                cand_nodes = int(cand_sd["weights"].shape[0]) if isinstance(cand_sd, dict) and "weights" in cand_sd else getattr(live_state, 'num_nodes', 4)
                                cand_topo_ver = topology_version + (1 if cand_nodes > prev_num_nodes else 0)
                                CheckpointManager.save("M9_Thalamus", cand_sd, is_torch=True, extra_states={
                                    "is_candidate": True, "phase": "PREPARED", "transaction_id": tx_id,
                                    "candidate_transaction_id": tx_id, "parent_state_hash": parent_hash,
                                    "candidate_hash": h, "generation": target_gen, "parent_generation": generation,
                                    "fitness": fitness, "topology_version": cand_topo_ver,
                                    "topology": {"num_nodes": cand_nodes}, "step_count": getattr(live_state, 'step_count', 0),
                                    "worker_epoch": worker_epoch
                                })
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M9_Thalamus', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M9_Thalamus", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                if 'last_committed_tx_id' in locals() and tx_id == last_committed_tx_id:
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    continue
                                if staged_tx_id != tx_id:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                                    continue
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                generation = payload.get("brain_generation", generation + 1)
                                master_tick = payload.get("target_tick", local_tick)
                                master_episode = payload.get("target_episode", local_episode)
                                if not live_state.commit_candidate(transaction_id=tx_id, target_generation=generation, target_tick=master_tick, target_episode=master_episode):
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                                    continue
                                if getattr(live_state, 'num_nodes', 4) > prev_num_nodes:
                                    topology_version += 1
                                    prev_num_nodes = getattr(live_state, 'num_nodes', 4)
                                if hasattr(live_state, 'topology_version'): live_state.topology_version = topology_version
                                staged_tx_id = None
                                last_committed_tx_id = tx_id
                                is_dirty = False; last_save = time.monotonic()
                                _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                CheckpointManager.discard_candidate("M9_Thalamus", tx_id)
                                if staged_tx_id == tx_id or locals().get('last_committed_tx_id') == tx_id:
                                    if locals().get('last_committed_tx_id') == tx_id:
                                        last_committed_tx_id = None
                                    if 'parent_gen_backup' in locals(): generation = parent_gen_backup
                                    if 'parent_topo_ver_backup' in locals(): topology_version = parent_topo_ver_backup
                                    if 'parent_prev_nodes_backup' in locals(): prev_num_nodes = parent_prev_nodes_backup
                                    if not live_state.rollback_candidate(transaction_id=tx_id) and parent_model_backup is not None:
                                        live_state = copy.deepcopy(parent_model_backup)
                                    staged_tx_id = None
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                latest_rpe = float(dopa_payload.get("rpe", 0.0))
                                latest_novelty = float(dopa_payload.get("novelty", 0.0))
                                latest_reward = float(dopa_payload.get("reward", latest_rpe))
                                latest_task_success = float(dopa_payload.get("task_success", 0.0))
                                local_plasticity = compute_local_plasticity("M9_Thalamus", dopa_payload)
                                if abs(local_plasticity) > 0.05 and last_bmu_id is not None and staged_tx_id is None:
                                    try:
                                        live_state.apply_plasticity(local_plasticity, last_bmu_id)
                                        fitness += local_plasticity
                                    except Exception as e: diagnose_error(e, "M9_Plasticity")
                        elif not is_sleeping:
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            evt_id = packet.get("event_id") if isinstance(packet, dict) else None
                            if evt_id and topic in ("RAW_VISION_M1A", "RAW_VISION_M1B", "RAW_AUDIO", "RAW_SPATIAL"):
                                parent_ids.insert(0, evt_id)
                            elif evt_id and topic in ("SEMANTICS", "MEM_RESULT", "MOTOR_FEEDBACK", "VOCAL_FEEDBACK", "SELF_AUDIO", "DREAM_REPLAY", "AMYGDALA_THREAT"):
                                parent_ids.append(evt_id)
                            if isinstance(packet, dict) and packet.get("task_id") not in (None, "NO_TASK"):
                                task_ids.append(packet.get("task_id"))
                            if topic == "RAW_VISION_M1A" and isinstance(payload, dict):
                                current_context["vision_where"] = numeric_encoder(payload.get("latent_vector"), 10)
                                if isinstance(payload.get("target"), dict):
                                    latest_m1a_target = payload["target"]
                                has_new_data = True
                            elif topic == "RAW_VISION_M1B" and isinstance(payload, dict): current_context["vision_what"] = numeric_encoder(payload.get("latent_vector"), 10); has_new_data = True
                            elif topic == "RAW_AUDIO" and isinstance(payload, dict): current_context["audio"] = numeric_encoder(payload.get("acoustic_features"), 10); has_new_data = True
                            elif topic == "RAW_SPATIAL" and isinstance(payload, dict):
                                current_context["spatial"] = numeric_encoder(payload.get("current_pos", {"x":0.5, "y":0.5}), 10)
                                if isinstance(payload.get("current_pos"), dict):
                                    latest_m3_pos = payload["current_pos"]
                                has_new_data = True
                            elif topic == "SEMANTICS" and isinstance(payload, dict): current_context["semantics"] = embedding_encoder(payload.get("semantic_embedding", payload.get("latent_vector")), 39); has_new_data = True
                            elif topic == "MEM_RESULT" and isinstance(payload, dict): current_context["memory"] = embedding_encoder(payload.get("retrieved_memory", ""), 39); has_new_data = True
                            elif topic == "MOTOR_FEEDBACK" and isinstance(payload, dict): 
                                vec = [payload.get("actual_dx", 0.0), payload.get("actual_dy", 0.0)]
                                current_context["motor_fb"] = numeric_encoder(vec, 10); has_new_data = True
                            elif topic == "VOCAL_FEEDBACK" and isinstance(payload, dict):
                                current_context["vocal_fb"] = numeric_encoder(payload.get("acoustic_output", []), 2); has_new_data = True
                            elif topic == "SELF_AUDIO" and isinstance(payload, dict):
                                current_context["self_audio"] = encode_audio_stft(payload.get("data", []), dim=10); has_new_data = True
                            elif topic == "DREAM_REPLAY" and isinstance(payload, dict):
                                current_context["dream_replay"] = numeric_encoder(payload.get("vector", []), 128); has_new_data = True
                            elif topic == "AMYGDALA_THREAT" and isinstance(payload, dict):
                                current_context["threat"] = numeric_encoder(payload.get("threat_vector", [0, 0]), 10)
                                inhibition_level = payload.get("inhibition_strength", 0.0); has_new_data = True
            if has_new_data and not is_sleeping:
                step_count += 1
                
                now_mono = time.monotonic()
                if not hasattr(live_state, '_last_broadcast_tick'):
                    live_state._last_broadcast_tick = -1
                    live_state._broadcasts_this_tick = 0
                    live_state._last_broadcast_wall = now_mono
                
                if live_state._last_broadcast_tick != local_tick or (now_mono - getattr(live_state, '_last_broadcast_wall', 0.0) > 0.25):
                    live_state._last_broadcast_tick = local_tick
                    live_state._broadcasts_this_tick = 0
                    live_state._last_broadcast_wall = now_mono
                    
                if live_state._broadcasts_this_tick >= 3:
                    if time.monotonic() - last_heartbeat > 5.0:
                        upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M9_Thalamus'})}")
                        _publish_heartbeat(bus, "M9_Thalamus", local_episode)
                        last_heartbeat = time.monotonic()
                    continue
                
                live_state._broadcasts_this_tick += 1

                for k in list(current_context.keys()):
                    vec = current_context[k]
                    vec = [v * 0.9 for v in vec]
                    current_context[k] = vec
                    vec_arr = np.array(vec, dtype=np.float32)
                    if not np.all(np.isfinite(vec_arr)):
                        logger.warning(f"⚠️ [M9] Non-finite values detected in modality '{k}', holding previous valid state.")
                        if '_prev_context' in locals() and k in _prev_context:
                            current_context[k] = _prev_context[k]
                        else:
                            current_context[k] = [0.0] * len(vec)
                
                energies = {k: sum(abs(v) for v in val) for k, val in current_context.items()}
                total_energy = sum(energies.values()) + 1e-6
                attention_routing = {k: v / total_energy for k, v in energies.items()}
                salience = float(min(1.0, max(0.35, (total_energy / 10.0) + inhibition_level)))
                weighted_context_128 = []
                for k in ["vision_where", "vision_what", "audio", "spatial", "threat", "semantics", "memory", "motor_fb", "vocal_fb", "self_audio", "dream_replay"]:
                    vec = current_context[k]
                    in_dim = modality_specs[k][0]
                    if len(vec) != in_dim: vec = numeric_encoder(vec, in_dim)
                    vec_np = np.array(vec, dtype=np.float32)
                    proj = np.dot(m9_projectors[k], vec_np).tolist()
                    weighted_context_128.extend(proj)
                
                weighted_context_128 = np.tanh(weighted_context_128).tolist()
                
                _prev_context = {k: list(v) for k, v in current_context.items()}
                
                try:
                    if not np.all(np.isfinite(np.array(weighted_context_128, dtype=np.float32))):
                        logger.warning("⚠️ [M9] Non-finite values in projected 128D vector. Skipping SOM update.")
                        continue
                    tensor_in = torch.tensor([weighted_context_128], dtype=torch.float32)
                    last_bmu_id, topology_changed = live_state.update_and_grow(
                        tensor_in, 
                        current_tick=local_tick, 
                        current_episode=local_episode
                    )
                    
                    destination_mask, raw_probs, topdown_attention = live_state.get_dynamic_routing(last_bmu_id, attention_routing, salience, inhibition_level, reward_signal=latest_reward, prediction_error=latest_rpe, novelty=latest_novelty, task_success=latest_task_success)
                    routing_payload = live_state.get_routing_payload(
                        bmu_idx=last_bmu_id,
                        attention_routing=attention_routing,
                        salience=salience,
                        inhibition_level=inhibition_level,
                        destination_mask=destination_mask,
                                            raw_probs=raw_probs,
                                            topdown_attention=topdown_attention
                                        )
                    curr_pos_dict = locals().get("latest_m3_pos", {"x": 0.5, "y": 0.5})
                    targ_pos_dict = locals().get("latest_m1a_target", {"x": float(topdown_attention["gaze"]["x"]), "y": float(topdown_attention["gaze"]["y"])})
                    m9_payload = {
                        "fused_latent": weighted_context_128,
                        "context_vector": weighted_context_128,
                        "attention_routing": attention_routing,
                        "salience": salience,
                        "destination_mask": destination_mask,
                        "neural_focus": [float(last_bmu_id)],
                        "amygdala_inhibition": inhibition_level,
                        "routing_probs": raw_probs,
                        "x": float(curr_pos_dict.get("x", 0.5)),
                        "y": float(curr_pos_dict.get("y", 0.5)),
                        "target_x": float(targ_pos_dict.get("x", 0.5)),
                        "target_y": float(targ_pos_dict.get("y", 0.5))
                    }
                    composite_parent = ",".join(parent_ids[:15]) if parent_ids else None
                    master_task = task_ids[0] if task_ids else "NO_TASK"
                    route_map = {
                        "M6_Language": "THALAMUS_M6",
                        "M7_Logic": "THALAMUS_M7",
                        "M8_Memory": "THALAMUS_M8",
                        "M4_Motor": "THALAMUS_M4",
                        "M10_Vocal": "THALAMUS_M10"
                    }
                    for dst, topic in route_map.items():
                        # Fallback for old keys just in case
                        if destination_mask.get(dst, False) or destination_mask.get(dst.split("_")[0], False):
                            bus.publish(topic, m9_payload, modality="routing", parent_event_id=composite_parent, episode_id=local_episode)
                    
                    bus.publish("THALAMUS_TOPDOWN", {"topdown_attention": topdown_attention, "modality": "topdown", "tick_id": local_tick}, modality="topdown", parent_event_id=composite_parent, episode_id=local_episode)
                except Exception as e:
                    if diagnose_error(e, "M9_Thalamus") == BrainErrorCategory.FATAL: stop_event.set(); break
            if time.monotonic() - last_heartbeat > 5.0:
                if is_dirty and time.monotonic() - last_save > 60 and can_worker_write_canonical("M9_Thalamus", staged_tx_id):
                    extra_st = { "transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_nodes": getattr(live_state, 'num_nodes', 4)},
                                "generation": generation, "fitness": fitness,
                                "topology_version": topology_version, "step_count": step_count }
                    CheckpointManager.save("M9_Thalamus", live_state, is_torch=True, extra_states=extra_st)
                    is_dirty = False; last_save = time.monotonic()
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M9_Thalamus'})}")
                _publish_heartbeat(bus, "M9_Thalamus", local_episode)
                last_heartbeat = time.monotonic()
    finally:
        if is_dirty and can_worker_write_canonical("M9_Thalamus", staged_tx_id):
            extra_st = { "transaction_id": locals().get('last_committed_tx_id'), "topology": {"num_nodes": getattr(live_state, 'num_nodes', 4)},
                        "generation": generation, "fitness": fitness,
                        "topology_version": topology_version, "step_count": step_count }
            CheckpointManager.save("M9_Thalamus", live_state, is_torch=True, extra_states=extra_st)
        bus.shutdown(); upstream_sock.close()

# ================= 10. M10_VOCAL =================
def m10_vocal_process(stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN); reseed_brain()
    bus = CoreBus(node_id="M10_Vocal")
    spec = MODEL_SPECS["M10_Vocal"]
    poller = zmq.Poller()
    bus.subscribe(list(dict.fromkeys(spec.get("input_contract", []) + ["SYSTEM_CONTROL"]))); poller.register(bus.sub_socket, zmq.POLLIN); poller.register(bus.ctrl_socket, zmq.POLLIN) if hasattr(bus, 'ctrl_socket') else None
    upstream_sock = zmq.Context().socket(zmq.PUSH); upstream_sock.connect("tcp://127.0.0.1:5554")
    code_dir = os.path.join(BASE_DIR, "05_Vocal_System/M10_Core_Engine")
    if code_dir not in sys.path: sys.path.append(code_dir)
    
    try:
        from m10_vocal_tract import VocalTractODE
        import torch
    except ImportError as e:
        logger.critical(f"🛑 Lỗi nạp thư viện M10_Vocal: {e}"); stop_event.set(); return

    chkpt, load_mode = load_worker_checkpoint_strict("M10_Vocal", is_torch=True)
    if chkpt:
        topology = chkpt.get("topology", {}); motor_dim = topology.get("motor_dim", 3); physical_dim = topology.get("physical_dim", 2)
        live_state = VocalTractODE(motor_dim=motor_dim, physical_dim=physical_dim)
        try: live_state.load_state_dict(chkpt["model_state"], strict=True)
        except Exception as e:
            logger.critical(f"💀 [RECOVERY_REQUIRED] M10_Vocal Mismatch! {e}")
            raise RuntimeError(f"RECOVERY_REQUIRED:M10_Vocal:{e}")
        optimizer = torch.optim.Adam(live_state.parameters(), lr=1e-3)
        if chkpt.get("optimizer_state"):
            try: optimizer.load_state_dict(chkpt["optimizer_state"])
            except Exception: pass
        generation, fitness, topology_version = chkpt.get("generation", 0), chkpt.get("fitness", 0.0), chkpt.get("topology_version", chkpt.get("architecture_version", 1))
    else:
        live_state = VocalTractODE(motor_dim=3, physical_dim=2)
        optimizer = torch.optim.Adam(live_state.parameters(), lr=1e-3)
        topology_version, generation, fitness = 1, 0, 0.0
    parent_model_backup = None; parent_state_backup = None; parent_opt_backup = None
    staged_state = None; staged_opt = None; staged_tx_id = None; staged_fitness_inc = 0.0
    live_state.train(); is_dirty, is_sleeping, last_heartbeat = False, True, time.monotonic()
    eligibility_trace, last_action_str = None, ""; lateral_topdown = None
    local_tick = 0; local_episode = "UNKNOWN"; last_save = time.monotonic()
    worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
    upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'READY', 'module': 'M10_Vocal', 'schema_verified': True, 'load_mode': load_mode, 'worker_epoch': worker_epoch})}")
    _publish_heartbeat(bus, "M10_Vocal", local_episode)
    local_tick = 0
    
    try:
        while not stop_event.is_set():
            socks = dict(poller.poll(50))
            for sock in socks:
                for msg in get_all_msgs(sock, bus):
                    topic = msg.split(" ", 1)[0]; packet = extract_data(msg)
                    if isinstance(packet, dict):
                        _publish_telemetry(bus, "M10_Vocal", local_episode)
                        if topic == "CEREBROSPINAL_FLUID":
                            local_tick, local_episode = packet.get("tick_id", local_tick), packet.get("episode_id", local_episode)
                        elif topic == "SYSTEM_CONTROL":
                            st = packet.get("state", "")
                            if st == "ASLEEP": is_sleeping = True
                            elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: is_sleeping = False
                            elif st == "EVOLUTION_PREPARE":
                                tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                                p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                                target_gen = p_load.get("target_generation", generation + 1)
                                if staged_tx_id != tx_id:
                                    if staged_tx_id is not None:
                                        staged_state, staged_opt, staged_fitness_inc = None, None, 0.0
                                    staged_tx_id = tx_id
                                    parent_gen_backup = generation
                                    parent_topo_ver_backup = topology_version
                                    parent_fit_backup = fitness
                                    parent_model_backup = copy.deepcopy(live_state)
                                    parent_state_backup = copy.deepcopy(live_state.state_dict())
                                    parent_opt_backup = copy.deepcopy(optimizer.state_dict())
                                cand_ref = staged_state if staged_state is not None else live_state
                                h_dim = cand_ref.fc1.out_features if hasattr(cand_ref, 'fc1') else getattr(cand_ref, 'hidden_dim', 64)
                                topo_d = {"hidden_dim": h_dim, "motor_dim": getattr(cand_ref, 'motor_dim', 3), "physical_dim": getattr(cand_ref, 'physical_dim', 2)}
                                cand_topo_ver = topology_version + (1 if (staged_state is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in live_state.parameters())) else 0)
                                h = stage_pytorch_candidate_file("M10_Vocal", live_state, staged_state, optimizer, staged_opt, tx_id, generation, target_gen, fitness + staged_fitness_inc, cand_topo_ver, topo_d)
                                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': 'M10_Vocal', 'candidate_hash': h, 'transaction_id': tx_id})}")
                                _publish_candidate_ready(bus, "M10_Vocal", h, local_episode, transaction_id=tx_id, command_id=packet.get("command_id", ""))
                            elif st == "GLOBAL_COMMIT":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                if 'last_committed_tx_id' in locals() and tx_id == last_committed_tx_id:
                                    _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                                    continue
                                if staged_tx_id != tx_id:
                                    _send_manual_ack(bus, "ERROR", tx_id, packet.get("command_id", ""))
                                    continue
                                payload = packet.get("payload", {}) if isinstance(packet, dict) else {}
                                generation = payload.get("brain_generation", generation + 1)
                                if staged_state is not None:
                                    if 'parent_model_backup' in locals() and parent_model_backup is not None and sum(p.numel() for p in staged_state.parameters()) != sum(p.numel() for p in parent_model_backup.parameters()):
                                        topology_version += 1 
                                    live_state = staged_state
                                    if staged_opt is not None: optimizer = staged_opt
                                    fitness += staged_fitness_inc
                                staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                last_committed_tx_id = tx_id
                                is_dirty = False; last_save = time.monotonic()
                                _send_manual_ack(bus, "COMMIT_APPLIED", tx_id, packet.get("command_id", ""))
                            elif st == "GLOBAL_ROLLBACK":
                                tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                                CheckpointManager.discard_candidate("M10_Vocal", tx_id)
                                if staged_tx_id == tx_id or locals().get('last_committed_tx_id') == tx_id:
                                    if locals().get('last_committed_tx_id') == tx_id:
                                        last_committed_tx_id = None
                                    if 'parent_gen_backup' in locals(): generation = parent_gen_backup
                                    if 'parent_topo_ver_backup' in locals(): topology_version = parent_topo_ver_backup
                                    if 'parent_fit_backup' in locals(): fitness = parent_fit_backup
                                    if parent_model_backup is not None:
                                        live_state = copy.deepcopy(parent_model_backup)
                                        if parent_opt_backup is not None:
                                            optimizer = type(optimizer)(filter(lambda p: p.requires_grad, live_state.parameters()), **optimizer.defaults)
                                            try: optimizer.load_state_dict(parent_opt_backup)
                                            except Exception: pass
                                    staged_state, staged_opt, staged_tx_id, staged_fitness_inc = None, None, None, 0.0
                                _send_manual_ack(bus, "ROLLED_BACK", tx_id, packet.get("command_id", ""))
                        elif topic == "THALAMUS_TOPDOWN":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "topdown_attention" in payload:
                                lateral_topdown = payload["topdown_attention"]
                        elif not is_sleeping and topic in ["CMD_VOCAL", "THALAMUS_M10", "VOCAL_INTENT"]:
                            parent_id = packet.get("event_id") if isinstance(packet, dict) else None
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            task_id = packet.get("task_id", payload.get("task_id", "NO_TASK")) if isinstance(packet, dict) else "NO_TASK"
                            if isinstance(payload, dict):
                                try:
                                    # [STEP 1.6 BUG FIX]: Properly distinguish semantic intent, reflex, and invalid messages
                                    has_action = "action" in payload
                                    has_thalamus_features = "fused_latent" in payload or "amygdala_inhibition" in payload
                                    
                                    if topic == "VOCAL_INTENT" or topic == "CMD_VOCAL":
                                        if not has_action:
                                            logger.warning(f"⚠️ [M10] Invalid {topic} message: missing 'action'. Ignoring.")
                                            continue
                                        action_str = payload.get("action", "SILENCE")
                                    elif topic == "THALAMUS_M10":
                                        if not has_thalamus_features:
                                            logger.warning(f"⚠️ [M10] Invalid THALAMUS_M10 message: missing features. Ignoring.")
                                            continue
                                        if payload.get("amygdala_inhibition", 0.0) > 0.5:
                                            action_str = "REFLEX_SOUND"
                                        elif payload.get("salience", 0.0) > 0.2:
                                            action_str = "VOCAL_RESPONSE"
                                        else:
                                            continue
                                    else:
                                        action_str = "SILENCE"
                                        
                                    last_action_str = action_str
                                    base_trace = categorical_encoder(action_str, 5)
                                    topdown_vec = numeric_encoder(lateral_topdown, 5) if lateral_topdown is not None else [0.0]*5
                                    fused_trace = [b + t for b, t in zip(base_trace, topdown_vec)]
                                    current_trace = torch.tensor([fused_trace], dtype=torch.float32)
                                    if eligibility_trace is None: eligibility_trace = current_trace
                                    else: eligibility_trace = 0.9 * eligibility_trace.detach() + 0.1 * current_trace
                                    live_state.eval()
                                    with torch.no_grad():
                                        live_state.motor_intent = eligibility_trace[:, :3]
                                        device = eligibility_trace.device
                                        state0 = torch.zeros(1, 2, dtype=torch.float32, device=device)
                                        t_span = torch.linspace(0, 1.0, 10, device=device)
                                        vocal_traj = live_state(t_span, state0)
                                        final_vocal_state = vocal_traj[-1]
                                    live_state.train()
                                    m10_payload = {"task_id": task_id if task_id != "NO_TASK" else payload.get("task_id", "NO_ID"), "status": "DONE", "acoustic_output": final_vocal_state.detach().tolist()[0]}
                                    bus.publish("VOCAL_FEEDBACK", m10_payload, modality="vocal_fb", parent_event_id=parent_id, episode_id=local_episode)
                                    v_pitch = float(torch.sigmoid(torch.tensor(final_vocal_state[0, 0].item())))
                                    v_vol = float(torch.sigmoid(torch.tensor(final_vocal_state[0, 1].item())))
                                    simulated_audio = (np.sin(np.linspace(0, 100 * v_pitch, 1600)) * v_vol).tolist()
                                    audio_payload = {"label": action_str, "data": simulated_audio, "type": "self_voice"}
                                    bus.publish("SELF_AUDIO", audio_payload, modality="audio", parent_event_id=parent_id, episode_id=local_episode)
                                except Exception as e:
                                    if diagnose_error(e, "M10_Vocal") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif is_sleeping and topic == "DREAM_REPLAY":
                            payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(payload, dict) and "vector" in payload:
                                try:
                                    dream_intent = torch.tensor([numeric_encoder(payload["vector"][-5:], 5)], dtype=torch.float32)
                                    live_state.motor_intent = dream_intent[:, :3]
                                    device = dream_intent.device
                                    state0 = torch.zeros(1, 2, dtype=torch.float32, device=device)
                                    t_span = torch.linspace(0, 0.2, 2, device=device)
                                    _ = live_state(t_span, state0)
                                except Exception as e:
                                    if diagnose_error(e, "M10_Dream") == BrainErrorCategory.FATAL: stop_event.set(); break
                        elif topic == "GLOBAL_PLASTICITY":
                            dopa_payload = packet.get("payload", packet) if isinstance(packet, dict) else packet
                            if isinstance(dopa_payload, dict):
                                novelty_val = float(dopa_payload.get("novelty", 0.0))
                                local_plasticity = compute_local_plasticity("M10_Vocal", dopa_payload)
                                if abs(local_plasticity) > 0.05 and eligibility_trace is not None:
                                    try:
                                        def m10_dopa_loss(model, x):
                                            model.motor_intent = x[:, :3]
                                            device = x.device
                                            state0 = torch.zeros(1, 2, dtype=torch.float32, device=device)
                                            t_span = torch.linspace(0, 1.0, 10, device=device)
                                            v_t = model(t_span, state0)
                                            return torch.nn.functional.mse_loss(v_t[-1], torch.tensor([categorical_encoder(last_action_str, 2)], dtype=torch.float32, device=device))
                                        m10_mut = None
                                        base_m10 = staged_state if staged_state is not None else live_state
                                        base_m10_opt = staged_opt if staged_opt is not None else optimizer
                                        is_valid, _, new_staged, new_opt = UniversalEvolutionProtocol.evolve_and_stage_pytorch(base_m10, eligibility_trace, m10_dopa_loss, base_m10_opt, rpe=local_plasticity, novelty=novelty_val, mutate_func=m10_mut)
                                        if is_valid and staged_tx_id is None:
                                            staged_state = new_staged
                                            staged_opt = new_opt
                                            staged_fitness_inc = local_plasticity
                                    except Exception as e:
                                        if diagnose_error(e, "M10_Plasticity") == BrainErrorCategory.FATAL: stop_event.set(); break
            current_time = time.monotonic()
            if current_time - last_heartbeat > 5.0:
                upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': 'M10_Vocal'})}")
                _publish_heartbeat(bus, "M10_Vocal", local_episode)
                last_heartbeat = time.monotonic()
            if is_dirty and time.monotonic() - last_save > 60 and can_worker_write_canonical("M10_Vocal", staged_tx_id):
                h_dim = getattr(live_state, 'hidden_dim', 64)
                extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"motor_dim": getattr(live_state, 'motor_dim', 3), "physical_dim": getattr(live_state, 'physical_dim', 2), "hidden_dim": h_dim}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
                CheckpointManager.save("M10_Vocal", live_state, optimizer, is_torch=True, extra_states=extra_st); is_dirty = False; last_save = time.monotonic()
    finally:
        if is_dirty and can_worker_write_canonical("M10_Vocal", staged_tx_id):
            h_dim = getattr(live_state, 'hidden_dim', 64)
            extra_st = {"transaction_id": locals().get('last_committed_tx_id'), "topology": {"motor_dim": getattr(live_state, 'motor_dim', 3), "physical_dim": getattr(live_state, 'physical_dim', 2), "hidden_dim": h_dim}, "generation": generation, "fitness": fitness, "topology_version": topology_version}
            CheckpointManager.save("M10_Vocal", live_state, optimizer, is_torch=True, extra_states=extra_st)
        bus.shutdown(); upstream_sock.close()
        
# =========================================================================
# EVOLUTION TRANSACTION MANAGER
# =========================================================================
class EvolutionTransactionManager:
    CANONICAL_11_MODELS = [
        "M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial",
        "M4_Motor", "M5_Amygdala", "M6_Language", "M7_Logic",
        "M8_Memory", "M9_Thalamus", "M10_Vocal"
    ]

    def __init__(self):
        self.manifest_path = os.path.join(BASE_DIR, "brain_manifest.json")
        self.arch_manifest_path = os.path.join(BASE_DIR, "architecture_manifest.json")
        self.current_generation, self.parent_hash = self.load_brain_lineage()
        self.current_transaction_id = None
        self.candidates = {}

    def get_brain_models(self):
        return list(self.CANONICAL_11_MODELS)

    def load_brain_lineage(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    if data.get("transaction_status") == "PREPARING":
                        logger.warning("⚠️ Phát hiện Master sập ngầm ở phiên trước. Tiến hành đối chiếu 11/11 Checkpoint...")
                        
                        target_gen = data.get("brain_generation", 0) + 1
                        brain_models = self.get_brain_models()
                        commit_count = 0
                        valid_nodes_count = len(brain_models)
                        
                        for model_id in brain_models:
                            chkpt_path = CheckpointManager.get_canonical_path(model_id)
                            if os.path.exists(chkpt_path):
                                try:
                                    fmt = MODEL_SPECS[model_id].get("checkpoint_format", "pkl")
                                    if fmt == "json" or chkpt_path.endswith(".json"):
                                        with open(chkpt_path, 'r', encoding="utf-8") as cf:
                                            c_data = json.load(cf)
                                    elif fmt in ("pt", "pth") or chkpt_path.endswith(".pt"):
                                        c_data = torch.load(chkpt_path, map_location=torch.device('cpu'), weights_only=False)
                                    elif fmt == "pkl":
                                        with open(chkpt_path, 'rb') as cf:
                                            import pickle
                                            c_data = pickle.load(cf)
                                    else:
                                        c_data = torch.load(chkpt_path, map_location=torch.device('cpu'), weights_only=False)
                                    
                                    if isinstance(c_data, dict):
                                        node_gen = c_data.get("generation", -1)
                                        node_tx = c_data.get("transaction_id", None)
                                    else:
                                        node_gen = -1
                                        node_tx = None
                                        
                                    target_tx = data.get("current_transaction_id")
                                    
                                    if node_gen == target_gen and (target_tx is None or node_tx == target_tx):
                                        commit_count += 1
                                except Exception as e:
                                    logger.error(f"Lỗi đọc Checkpoint của {model_id} khi đối chiếu: {e}")
                                    
                        if commit_count == valid_nodes_count and valid_nodes_count > 0:
                            logger.info(f"✅ Đối chiếu: 11/11 node đã Commit thành công. Chuyển trạng thái thành COMMITTED.")
                            data["transaction_status"] = "COMMITTED"
                            data["brain_generation"] = target_gen
                            data["committed_generation"] = target_gen
                        elif commit_count == 0:
                            logger.warning(f"⏪ Đối chiếu: 0/11 node Commit. Hủy giao dịch nháp và giữ nguyên thế hệ COMMITTED.")
                            data["transaction_status"] = "COMMITTED"
                        else:
                            logger.critical(f"🚨 [SPLIT-BRAIN CATASTROPHE] Đối chiếu: {commit_count}/{valid_nodes_count} node đã Commit. HỆ THỐNG ĐỨT GÃY ĐỒNG BỘ!")
                            data["transaction_status"] = "RECOVERY_REQUIRED"
                            
                        data["current_transaction_id"] = None
                        with open(self.manifest_path, "w", encoding="utf-8") as fw:
                            json.dump(data, fw, indent=4)
                            
                        if data["transaction_status"] == "RECOVERY_REQUIRED":
                            raise RuntimeError("Hệ thống yêu cầu chạy Migration/Repair script thủ công để đồng bộ lại các Checkpoint bị lệch!")
                            
                    return data.get("committed_generation", data.get("brain_generation", 0)), data.get("committed_brain_hash", data.get("brain_hash", "GENESIS"))
            except Exception as e:
                if isinstance(e, RuntimeError): raise e
                logger.error(f"Lỗi khởi động manifest: {e}")
        return 0, "GENESIS"

    def start_transaction(self):
        self.current_transaction_id = f"tx_{uuid.uuid4().hex[:8]}"
        self.candidates.clear()
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["current_transaction_id"] = self.current_transaction_id
                data["transaction_status"] = "PREPARING"
                with open(self.manifest_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
            except Exception: pass
        return self.current_transaction_id

    def register_candidate(self, model_id, model_hash, tx_id):
        if model_id not in self.CANONICAL_11_MODELS or self.current_transaction_id is None:
            return
        if tx_id != self.current_transaction_id:
            logger.warning(f"Từ chối Candidate {model_id} do lệch Transaction (Nhận: {tx_id}, Chờ: {self.current_transaction_id})")
            return
        if model_hash:
            self.candidates[model_id] = model_hash
            logger.debug(f"[EVO TX] Đã nhận candidate: {model_id} -> {str(model_hash)[:12]}")

    def verify_all_staged_candidates(self):
        brain_models = self.get_brain_models()
        missing = [m for m in brain_models if m not in self.candidates]
        if missing:
            return False, f"Thiếu ACK ứng viên từ: {missing}"
        for m_id in brain_models:
            if str(self.candidates.get(m_id, "")).startswith("UNCHANGED_"):
                continue
            cand_path = CheckpointManager.get_candidate_path(m_id, self.current_transaction_id)
            if not os.path.exists(cand_path):
                return False, f"Thiếu file candidate bất biến trên đĩa của {m_id} ({cand_path})"
        return True, "OK"

    def commit_global_evolution(self, fitness_score):
        brain_models = self.get_brain_models()
        ok_staged, reason = self.verify_all_staged_candidates()
        if not ok_staged:
            logger.error(f"⏳ [GLOBAL COMMIT ABORT] Ứng viên chưa hợp lệ: {reason}")
            return False

        tx_id = self.current_transaction_id
        target_gen = self.current_generation + 1
        verified_models_meta = {}
        clean_candidates = {}
        pre_commit_backups = {}

        def _restore_pre_commit_backups():
            for b_mod, (b_path, c_path) in pre_commit_backups.items():
                if b_path and os.path.exists(b_path):
                    try:
                        shutil.copy2(b_path, c_path)
                    except Exception as re_err:
                        logger.error(f"Lỗi khôi phục backup cho {b_mod}: {re_err}")

        for m_id in sorted(brain_models):
            is_torch = MODEL_SPECS.get(m_id, {}).get("checkpoint_format", "pkl") in ("pt", "pth")
            canon_path = CheckpointManager.get_canonical_path(m_id, use_manifest=False)
            bak_path = CheckpointManager.backup_checkpoint(m_id, reason=f"tx_{tx_id}")
            pre_commit_backups[m_id] = (bak_path, canon_path)

            cand_path = CheckpointManager.get_candidate_path(m_id, tx_id)
            if os.path.exists(cand_path):
                if not CheckpointManager.commit_candidate(m_id, transaction_id=tx_id, is_torch=is_torch):
                    logger.critical(f"💀 [GLOBAL COMMIT FAILED] Không thể promote candidate cho {m_id}! Hoàn tác đĩa...")
                    _restore_pre_commit_backups()
                    return False

            chkpt = CheckpointManager.load(m_id, is_torch=is_torch, validate_hash=True, enforce_manifest=False)
            if not chkpt:
                logger.critical(f"💀 [GLOBAL COMMIT FAILED] Không đọc được checkpoint chuẩn của {m_id}! Hoàn tác đĩa...")
                _restore_pre_commit_backups()
                return False

            if chkpt.get("generation") != target_gen:
                extra_sync = dict(chkpt.get("extra_states", {}))
                extra_sync.update({
                    "generation": target_gen,
                    "parent_generation": self.current_generation,
                    "topology_version": chkpt.get("topology_version", 1),
                    "topology": chkpt.get("topology", {}),
                    "fitness": chkpt.get("fitness", 0.0),
                    "transaction_id": tx_id,
                    "phase": "COMMITTED",
                    "is_candidate": False,
                    "candidate_transaction_id": None
                })
                if not CheckpointManager.save(m_id, chkpt["model_state"], chkpt.get("optimizer_state"), is_torch=is_torch, extra_states=extra_sync):
                    logger.critical(f"💀 [GLOBAL COMMIT FAILED] Không thể đồng bộ thế hệ {target_gen} cho {m_id}! Hoàn tác đĩa...")
                    _restore_pre_commit_backups()
                    return False
                chkpt = CheckpointManager.load(m_id, is_torch=is_torch, validate_hash=True, enforce_manifest=False)

            actual_hash = chkpt["state_hash"]
            clean_candidates[m_id] = actual_hash
            verified_models_meta[m_id] = {
                "generation": target_gen,
                "topology_version": chkpt.get("topology_version", 1),
                "checkpoint_path": os.path.relpath(canon_path, BASE_DIR).replace("\\", "/"),
                "state_hash": actual_hash
            }

        self.current_generation = target_gen
        brain_hash = tensor_state_hash(clean_candidates)
        now_ts = time.time()
        manifest_data = {}
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                pass
        manifest_data.update({
            "brain_generation": self.current_generation,
            "brain_hash": brain_hash,
            "parent_brain_hash": self.parent_hash,
            "committed_generation": self.current_generation,
            "committed_brain_hash": brain_hash,
            "committed_timestamp": now_ts,
            "fitness": fitness_score,
            "candidate_hashes": {m_id: None for m_id in brain_models},
            "model_hashes": clean_candidates,
            "models": verified_models_meta,
            "transaction_status": "COMMITTED",
            "current_transaction_id": None
        })
        if isinstance(manifest_data.get("metadata"), dict):
            manifest_data["metadata"]["last_updated"] = now_ts

        history = manifest_data.setdefault("rollback_lineage", {}).setdefault("history", [])
        history.append(self.parent_hash)
        if len(history) > manifest_data["rollback_lineage"].get("max_depth", 5):
            history.pop(0)

        tmp_path = self.manifest_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.manifest_path)
        except Exception as e:
            logger.error(f"💀 Ghi brain_manifest thất bại: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        self.parent_hash = brain_hash
        self.candidates.clear()
        self.current_transaction_id = None
        logger.info(f"🌟 [GLOBAL COMMIT] Thế hệ {self.current_generation} thành công! BrainHash: {brain_hash[:12]}...")
        return True

    def reset_candidates(self, rollback_manifest=True):
        tx_id = self.current_transaction_id
        for m_id in self.get_brain_models():
            CheckpointManager.discard_candidate(m_id, tx_id)
        if rollback_manifest and os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["transaction_status"] = "COMMITTED" if data.get("committed_generation") == self.current_generation else "ROLLED_BACK"
                data["current_transaction_id"] = None
                with open(self.manifest_path, "w", encoding="utf-8") as fw:
                    json.dump(data, fw, indent=4)
            except Exception:
                pass
        self.candidates.clear()
        self.current_transaction_id = None
        logger.debug("[EVO TX] Candidates discarded and transaction reset.")
    
def _start_sandbox_core_wrapper(stop_event):
    try:
        sandbox_dir = os.path.join(BASE_DIR, "06_Laboratory")
        for p in (BASE_DIR, sandbox_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
        for mod in list(sys.modules.keys()):
            if mod.startswith("sandbox_core") or mod.startswith("core_bus") or mod.startswith("sandbox_env"):
                del sys.modules[mod]
        import sandbox_core  # type: ignore
        if not hasattr(sandbox_core, "start_sandbox_core"):
            raise AttributeError(
                f"Module sandbox_core ({getattr(sandbox_core, '__file__', 'unknown')}) "
                f"KHÔNG CÓ 'start_sandbox_core'.\n"
                f"Các attr công khai: {[a for a in dir(sandbox_core) if not a.startswith('_')]}"
            )
        sandbox_core.start_sandbox_core(stop_event)
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Sandbox_Core failed to start: {e}")
        logger.error(traceback.format_exc())
        
# ================= MASTER SUPERVISOR =================
class MasterSupervisor:
    def __init__(self):
        self.stop_event = multiprocessing.Event()
        self.processes, self.external_processes = {}, {}
        self.process_targets = {
            "ZMQ_Broker": run_central_broker,
            "M0_Cerebrospinal": m0_cerebrospinal_fluid_process,
            "M1a_Dorsal": m1a_dorsal_process,
            "M1b_Ventral": m1b_ventral_process,
            "M2_Ear": m2_ear_process,
            "M3_Spatial": m3_spatial_process,
            "M4_Motor": m4_motor_process,
            "M5_Amygdala": m5_amygdala_process,
            "M6_Language": m6_language_process,
            "M7_Logic": m7_prefrontal_process,
            "M8_Memory": m8_memory_process,
            "M9_Thalamus": m9_thalamus_process,
            "M10_Vocal": m10_vocal_process,
            "Sandbox_Core": _start_sandbox_core_wrapper 
        }
        self.external_targets = {}
        all_targets = [name for name in list(self.process_targets.keys()) + list(self.external_targets.keys()) if name != "ZMQ_Broker"]
        self.health_registry = {name: time.time() for name in all_targets}
        self.worker_epochs = {name: 0 for name in all_targets}
        self.worker_instances = {}
        self.module_telemetry = {}
        self.WATCHDOG_TIMEOUT = 300.0
        self.required_modules = set(self.health_registry.keys())
        self.ready_modules = set()
        self.evo_manager = EvolutionTransactionManager()
        self.brain_generation = self.evo_manager.current_generation
        self.last_evolution_time = time.time()
        self.circadian_clock = 0.0; self.is_sleeping = False
        self.best_committed_fitness = 0.40
        self.skill_baselines = {"reward_norm": 0.45, "prediction_norm": 0.45, "survival": 0.7}
        self._reset_episode_metrics()
        self.credit_ledger = {name: 1.0 for name in self.process_targets.keys() if name not in ["ZMQ_Broker", "M0_Cerebrospinal", "Sandbox_Core"]}

    def _reset_episode_metrics(self):
        self.episode_metrics = {
            "train_reward_sum": 0.0, "train_pred_sum": 0.0, "train_updates": 0,
            "val_reward_sum": 0.0, "val_reward_updates": 0,
            "val_pred_sum": 0.0, "val_pred_updates": 0,
            "val_updates": 0,
            "survival": 1.0, "survival_sum": 0.0, "survival_updates": 0,
            "memory": 0.0, "total_events": 0
        }

    def _terminate_and_confirm_process(self, name, p):
        if p is None:
            return
        try:
            is_alive = p.is_alive() if isinstance(p, multiprocessing.Process) else (p.poll() is None)
            if is_alive:
                logger.warning(f"🛑 Đang dừng tiến trình cũ của {name} (PID={getattr(p, 'pid', 'N/A')})...")
                p.terminate()
                if isinstance(p, multiprocessing.Process):
                    p.join(timeout=3.0)
                    if p.is_alive():
                        logger.warning(f"⚠️ Tiến trình {name} không phản hồi terminate, cưỡng chế kill()...")
                        p.kill()
                        p.join(timeout=2.0)
                else:
                    try:
                        p.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=2.0)
        except Exception as e:
            logger.error(f"Lỗi khi dừng tiến trình cũ {name}: {e}")

    def _spawn_worker_with_epoch(self, name, target):
        old_p = self.processes.get(name)
        if old_p is not None:
            self._terminate_and_confirm_process(name, old_p)
        epoch = self.worker_epochs.get(name, 0) + 1
        self.worker_epochs[name] = epoch
        instance_id = f"{name}_ep{epoch}_{uuid.uuid4().hex[:6]}"
        self.worker_instances[name] = instance_id
        if name in MODEL_SPECS:
            epoch_dir = os.path.join(BASE_DIR, "models", name)
            os.makedirs(epoch_dir, exist_ok=True)
            try:
                with open(os.path.join(epoch_dir, ".worker_epoch"), "w", encoding="utf-8") as ef:
                    ef.write(str(epoch))
            except Exception:
                pass
        os.environ["ASUNA_WORKER_EPOCH"] = str(epoch)
        os.environ["ASUNA_WORKER_INSTANCE_ID"] = instance_id
        p = multiprocessing.Process(target=target, args=(self.stop_event,), name=name)
        p.start()
        self.processes[name] = p
        self.health_registry[name] = time.time()
        return p

    def _verify_and_mark_ready(self, map_name, data):
        if not map_name or map_name not in self.required_modules:
            return False
        if data.get("load_mode") == WorkerLoadMode.RECOVERY_REQUIRED:
            logger.critical(f"💀 [RECOVERY_REQUIRED] {map_name} báo lỗi checkpoint nghiêm trọng! Dừng hệ thống.")
            self.stop_event.set()
            return False
        if map_name in MODEL_SPECS and data.get("schema_verified") is False:
            logger.error(f"❌ Từ chối READY từ {map_name}: schema_verified=False")
            return False
        msg_epoch = data.get("worker_epoch")
        expected_epoch = self.worker_epochs.get(map_name, 1)
        if msg_epoch is not None and int(msg_epoch) < expected_epoch:
            logger.warning(f"⚠️ Bỏ qua tín hiệu READY cũ từ {map_name} (epoch {msg_epoch} < {expected_epoch})")
            return False
        self.health_registry[map_name] = time.time()
        if map_name not in self.ready_modules:
            self.ready_modules.add(map_name)
            logger.info(f"🟢 {map_name} READY (epoch={expected_epoch}, mode={data.get('load_mode', 'READY')})! ({len(self.ready_modules)}/{len(self.required_modules)})")
        return True
        
    def start_all(self):
        self.bus = CoreBus(node_id="MasterSupervisor", is_master=True)
        self.upstream_sock = zmq.Context().socket(zmq.PULL)
        self.upstream_sock.bind("tcp://127.0.0.1:5554")
        logger.info("⚡ KHỞI ĐỘNG CƠ THỂ ASUNA (BẢN ĐẠI THỐNG NHẤT) ⚡")
        
        if "ZMQ_Broker" in self.process_targets:
            target = self.process_targets["ZMQ_Broker"]
            p = multiprocessing.Process(target=target, args=(self.stop_event,), name="ZMQ_Broker")
            p.start(); self.processes["ZMQ_Broker"] = p; self.health_registry["ZMQ_Broker"] = time.time()
            time.sleep(3.0) 
            
        for name, target in self.process_targets.items():
            if name == "ZMQ_Broker": continue
            self._spawn_worker_with_epoch(name, target)
            
        for name, script in self.external_targets.items():
            if os.path.exists(script):
                self.external_processes[name] = subprocess.Popen([sys.executable, script])
                self.health_registry[name] = time.time()
            else:
                logger.critical(f"🚨 [FATAL] Thiếu tệp ngoại vi {script}. Hủy khởi động!"); self.shutdown(); return
        self.wait_for_handshake()
        
    def wait_for_handshake(self):
        logger.info("⏳ Đang chờ Handshake toàn cục...")
        self.bus.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        start_wait = time.time()
        while len(self.ready_modules) < len(self.required_modules) and time.time() - start_wait < 600.0:
            if self.stop_event.is_set():
                self.shutdown()
                return
            for name, p in list(self.processes.items()):
                if not p.is_alive():
                    logger.critical(f"💀 [BOOT CRASH] Tiến trình '{name}' đã sập trong lúc khởi động (exitcode={p.exitcode}). Hủy Handshake!")
                    self.shutdown()
                    return
            try:
                msg = self.upstream_sock.recv_string(flags=zmq.NOBLOCK)
                data = extract_data(msg)
                if isinstance(data, dict):
                    state_val, map_name = data.get("state"), data.get("module")
                    if state_val == "HEARTBEAT" and map_name:
                        self.health_registry[map_name] = time.time()
                    elif state_val == "READY" and map_name:
                        self._verify_and_mark_ready(map_name, data)
            except zmq.Again: pass
            try:
                msg2 = self.bus.sub_socket.recv_string(flags=zmq.NOBLOCK)
                data2 = extract_data(msg2)
                if isinstance(data2, dict):
                    state_val, map_name = data2.get("state"), data2.get("module", data2.get("node_id"))
                    if state_val in ["READY", "MODULE_READY"] and map_name:
                        self._verify_and_mark_ready(map_name, data2)
            except zmq.Again: pass
            time.sleep(0.05)
            
        if len(self.ready_modules) == len(self.required_modules):
            logger.info(f"✅ {len(self.ready_modules)}/{len(self.required_modules)} THỰC THỂ ĐỒNG BỘ! PHÁT LỆNH START_TRAINING...")
            payload = {'state': 'START_TRAINING', "modality": "control"}
            for _ in range(5): self.bus.publish("SYSTEM_CONTROL", payload, modality="control"); time.sleep(0.1)
        else:
            logger.critical(f"❌ TIMEOUT HANDSHAKE! Thiếu: {self.required_modules - self.ready_modules}. Đang dọn dẹp và dừng hệ thống!")
            self.shutdown()
            
    def run_supervisor_loop(self):
        poller = zmq.Poller()
        poller.register(self.upstream_sock, zmq.POLLIN); poller.register(self.bus.sub_socket, zmq.POLLIN)
        if hasattr(self.bus, 'ctrl_socket'): poller.register(self.bus.ctrl_socket, zmq.POLLIN)
        loop_tick_time = time.time()
        try:
            while not self.stop_event.is_set():
                socks = dict(poller.poll(100))
                
                for sock_target in [self.upstream_sock, self.bus.sub_socket, getattr(self.bus, 'ctrl_socket', None)]:
                    if not sock_target or socks.get(sock_target) != zmq.POLLIN: continue
                    for msg in get_all_msgs(sock_target, self.bus):
                        topic = msg.split(" ", 1)[0] if isinstance(msg, str) else ""
                        data = extract_data(msg)
                        if isinstance(data, dict):
                            state_val, map_name = data.get("state", ""), data.get("module", data.get("source", ""))
                            
                            if topic == "ENV_REWARD":
                                if data.get("source") == "Sandbox_Core":
                                    rew_raw = float(np.clip(float(data.get("reward", 0.0)), -1.0, 1.0))
                                    rew_norm = (rew_raw + 1.0) * 0.5
                                    self.episode_metrics["total_events"] += 1
                                    if self.episode_metrics["total_events"] % 3 == 0:
                                        self.episode_metrics["val_reward_sum"] += rew_norm
                                        self.episode_metrics["val_reward_updates"] += 1
                                        self.episode_metrics["val_updates"] += 1
                                    else:
                                        self.episode_metrics["train_reward_sum"] += rew_norm
                                        self.episode_metrics["train_updates"] += 1
                            elif topic == "GLOBAL_DOPAMINE":
                                if data.get("source") == "M7_Logic":
                                    pred_acc = float(np.clip(1.0 - (abs(float(data.get("rpe", 0.0))) / 2.0), 0.0, 1.0))
                                    self.episode_metrics["total_events"] += 1
                                    if self.episode_metrics["total_events"] % 3 == 0:
                                        self.episode_metrics["val_pred_sum"] += pred_acc
                                        self.episode_metrics["val_pred_updates"] += 1
                                        self.episode_metrics["val_updates"] += 1
                                    else:
                                        self.episode_metrics["train_pred_sum"] += pred_acc
                                        self.episode_metrics["train_updates"] += 1
                            elif topic == "AMYGDALA_THREAT":
                                surv_val = float(np.clip(1.0 - float(data.get("threat_probability", 0.0)), 0.0, 1.0))
                                self.episode_metrics["survival_sum"] += surv_val
                                self.episode_metrics["survival_updates"] += 1
                                self.episode_metrics["survival"] = self.episode_metrics["survival_sum"] / self.episode_metrics["survival_updates"]
                            elif topic == "MEM_RESULT":
                                self.episode_metrics["memory"] += 1.0
                            elif topic == "TELEMETRY" and map_name:
                                self.module_telemetry[map_name] = {
                                    "latency_ms": float(data.get("latency_ms", 0.0)),
                                    "rss_mb": float(data.get("rss_mb", 0.0)),
                                    "timestamp": time.time()
                                }
                                
                            if state_val in ["READY", "MODULE_READY"] and map_name in self.health_registry:
                                if self._verify_and_mark_ready(map_name, data):
                                    self.bus.publish("SYSTEM_CONTROL", {"state": "RESUME" if not self.is_sleeping else "ASLEEP", "modality": "control"}, modality="control")
                            elif state_val == "HEARTBEAT" and map_name in self.health_registry:
                                msg_epoch = data.get("worker_epoch")
                                if msg_epoch is None or int(msg_epoch) == self.worker_epochs.get(map_name, 1):
                                    self.health_registry[map_name] = time.time()
                            elif state_val in ["EMERGENCY_STOP", "AWAKE_AND_SAFE"] and data.get("source") == "M5_Amygdala":
                                if getattr(self, "_last_emergency_relay", None) != state_val:
                                    self._last_emergency_relay = state_val
                                    self.bus.publish("SYSTEM_CONTROL", {"state": state_val, "source": "MasterSupervisor"}, modality="control")
                            elif state_val in ["CANDIDATE_READY", "PREPARED"]:
                                c_hash = data.get("candidate_hash")
                                tx_id = data.get("transaction_id")
                                if map_name and c_hash: self.evo_manager.register_candidate(map_name, c_hash, tx_id)
                current_time = time.time()
                dt = current_time - loop_tick_time; loop_tick_time = current_time
                for name, p in list(self.processes.items()) + list(self.external_processes.items()):
                    if name == "ZMQ_Broker": continue
                    is_dead = (isinstance(p, multiprocessing.Process) and not p.is_alive()) or (hasattr(p, 'poll') and p.poll() is not None)
                    if is_dead or (current_time - self.health_registry.get(name, current_time) > self.WATCHDOG_TIMEOUT):
                        logger.critical(f"🚨 [WATCHDOG] Node {name} bị sập hoặc treo quá hạn! Đang dừng tiến trình cũ và khởi động lại...")
                        if name in self.ready_modules:
                            self.ready_modules.remove(name)
                        if name in self.process_targets:
                            target = self.process_targets[name]
                            self._spawn_worker_with_epoch(name, target)
                self.circadian_clock += dt
                if self.circadian_clock > 300.0 and not self.is_sleeping:
                    payload = {"state": "ASLEEP", "modality": "control"}
                    self.bus.publish("SYSTEM_CONTROL", payload, modality="control"); self.is_sleeping = True
                elif self.circadian_clock > 330.0 and self.is_sleeping:
                    payload = {"state": "AWAKE", "modality": "control"}
                    self.bus.publish("SYSTEM_CONTROL", payload, modality="control"); self.is_sleeping = False; self.circadian_clock = 0.0
                
                # Global Evolution Loop - TRUE 2-PHASE COMMIT
                if current_time - self.last_evolution_time > 60.0 and len(self.ready_modules) == len(self.required_modules):
                    transaction_id = self.evo_manager.start_transaction()
                    logger.info(f"🔄 [PHASE 1] KHỞI TẠO 2-PHASE COMMIT. Transaction: {transaction_id}")
                    
                    target_nodes = self.evo_manager.get_brain_models()
                    master_target_tick = getattr(self.bus, 'current_tick', 0)
                    master_target_episode = getattr(self.bus, 'current_episode', "UNKNOWN")
                    # --- PHASE 1: PREPARE ---
                    all_prepared, prep_acks = self.bus.send_control_command(
                        command="EVOLUTION_PREPARE", 
                        payload={"transaction_id": transaction_id, "target_generation": self.brain_generation + 1, "target_tick": master_target_tick, "target_episode": master_target_episode}, 
                        expected_modules=target_nodes, transaction_id=transaction_id, timeout_sec=5.0
                    )
                    
                    for node, ack in prep_acks.items():
                        c_hash = ack.get("details", {}).get("candidate_hash")
                        if c_hash: self.evo_manager.register_candidate(node, c_hash, transaction_id)
                        
                    missing = set(target_nodes) - set(prep_acks.keys())
                    staged_ok, staged_reason = self.evo_manager.verify_all_staged_candidates()
                    if not all_prepared or missing or not staged_ok:
                        logger.warning(f"⚠️ [PHASE 1 FAILED] Thiếu/lỗi ứng viên ({missing}, {staged_reason}). HỦY TIẾN HÓA!")
                        all_rolled_back, rb_acks = self.bus.send_control_command(
                            command="GLOBAL_ROLLBACK", payload={"transaction_id": transaction_id},
                            expected_modules=target_nodes, transaction_id=transaction_id, timeout_sec=5.0
                        )
                        if not all_rolled_back: logger.error(f"💀 LỖI NGHIÊM TRỌNG: Lệch pha Rollback ở các node: {set(target_nodes) - set(rb_acks.keys())}")
                        self.evo_manager.reset_candidates()
                    else:
                        m = self.episode_metrics
                        val_updates = m["val_updates"]
                        val_rew_cnt = m["val_reward_updates"]
                        val_pred_cnt = m["val_pred_updates"]
                        total_updates = m["train_updates"] + val_updates
                        reward_norm = (m["val_reward_sum"] / val_rew_cnt) if val_rew_cnt > 0 else 0.5
                        pred_norm = (m["val_pred_sum"] / val_pred_cnt) if val_pred_cnt > 0 else 0.5
                        survival_norm = float(np.clip(m["survival"], 0.0, 1.0))
                        mem_norm = float(np.clip(m["memory"] / max(1.0, float(total_updates)), 0.0, 1.0))

                        avg_lat_ms = np.mean([v["latency_ms"] for v in self.module_telemetry.values()]) if self.module_telemetry else 10.0
                        total_rss_mb = sum(v["rss_mb"] for v in self.module_telemetry.values()) if self.module_telemetry else (psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
                        real_lat_penalty = float(np.clip(avg_lat_ms / 1000.0, 0.0, 0.25))
                        real_ram_penalty = float(np.clip(total_rss_mb / 8192.0, 0.0, 0.25))

                        brain_fitness = (
                            0.35 * reward_norm
                            + 0.30 * pred_norm
                            + 0.20 * survival_norm
                            + 0.15 * mem_norm
                            - 0.10 * (real_lat_penalty + real_ram_penalty)
                        )
                        skills_preserved = (
                            reward_norm >= self.skill_baselines["reward_norm"] - 0.05
                            and pred_norm >= self.skill_baselines["prediction_norm"] - 0.05
                            and survival_norm >= self.skill_baselines["survival"] - 0.05
                        )
                        logger.info(
                            f"⚖️ BrainFitness (Val): {brain_fitness:.4f} (Target > {self.best_committed_fitness:.4f}) | "
                            f"rew={reward_norm:.2f}, pred={pred_norm:.2f}, surv={survival_norm:.2f}, mem={mem_norm:.2f}, "
                            f"lat={avg_lat_ms:.1f}ms, ram={total_rss_mb:.1f}MB, skills_ok={skills_preserved}"
                        )
                        
                        if val_updates < 2 or brain_fitness < (self.best_committed_fitness - 0.02) or not skills_preserved:
                            logger.warning(
                                f"⚠️ [GLOBAL ROLLBACK] Không đạt tiêu chuẩn kiểm định (val_updates={val_updates}, "
                                f"fitness={brain_fitness:.4f} <= {self.best_committed_fitness:.4f}, skills_ok={skills_preserved}). Hủy chu kỳ."
                            )
                            self.bus.send_control_command(
                                command="GLOBAL_ROLLBACK", payload={"transaction_id": transaction_id},
                                expected_modules=target_nodes, transaction_id=transaction_id, timeout_sec=5.0
                            )
                            self.evo_manager.reset_candidates()
                        else:
                            # --- PHASE 2: COMMIT ---
                            logger.info("🚀 [PHASE 2] Gửi lệnh GLOBAL_COMMIT...")
                            all_committed, commit_acks = self.bus.send_control_command(
                                command="GLOBAL_COMMIT", payload={"transaction_id": transaction_id, "brain_generation": self.brain_generation + 1, "target_tick": master_target_tick, "target_episode": master_target_episode},
                                expected_modules=target_nodes, transaction_id=transaction_id, timeout_sec=5.0
                            )
                            
                            valid_commit_st = {"COMMIT_APPLIED", "COMMITTED_INFRA"}
                            missing_commit = set(target_nodes) - {k for k, v in commit_acks.items() if v.get("status") in valid_commit_st}
                            retry_count = 0
                            while missing_commit and retry_count < 3:
                                retry_count += 1
                                logger.warning(f"⚠️ [PHASE 2 RETRY {retry_count}] Thiếu ACK từ {missing_commit}...")
                                _, new_acks = self.bus.send_control_command(
                                    command="GLOBAL_COMMIT", payload={"transaction_id": transaction_id, "brain_generation": self.brain_generation + 1, "target_tick": master_target_tick, "target_episode": master_target_episode},
                                    expected_modules=list(missing_commit), transaction_id=transaction_id, timeout_sec=3.0
                                )
                                missing_commit -= {k for k, v in new_acks.items() if v.get("status") in valid_commit_st}
                            
                            if not missing_commit and self.evo_manager.commit_global_evolution(brain_fitness):
                                logger.info("✅ [PHASE 2 OK] Đã promote 11/11 Candidate và ghi Manifest thành công!")
                                self.brain_generation = self.evo_manager.current_generation
                                self.best_committed_fitness = 0.8 * self.best_committed_fitness + 0.2 * brain_fitness
                                self.skill_baselines["reward_norm"] = 0.8 * self.skill_baselines["reward_norm"] + 0.2 * reward_norm
                                self.skill_baselines["prediction_norm"] = 0.8 * self.skill_baselines["prediction_norm"] + 0.2 * pred_norm
                                self.skill_baselines["survival"] = 0.8 * self.skill_baselines["survival"] + 0.2 * survival_norm
                            else:
                                logger.critical(f"💀 [COMMIT ABORT] Các node {missing_commit} lỗi commit. Rollback toàn bộ về thế hệ {self.brain_generation}!")
                                self.bus.send_control_command(
                                    command="GLOBAL_ROLLBACK", payload={"transaction_id": transaction_id},
                                    expected_modules=target_nodes, transaction_id=transaction_id, timeout_sec=5.0
                                )
                                self.evo_manager.reset_candidates()

                    mean_credit = (sum(self.credit_ledger.values()) / max(1, len(self.credit_ledger))) + 1e-6
                    for k in self.credit_ledger: self.credit_ledger[k] = float(np.clip(self.credit_ledger[k] / mean_credit, 0.2, 2.0))
                    self.bus.publish("SYSTEM_CONTROL", {"state": "UPDATE_CREDIT", "credit": self.credit_ledger}, modality="control")
                    self._reset_episode_metrics()
                    self.last_evolution_time = current_time
        except KeyboardInterrupt: self.stop_event.set()
        self.shutdown()
        
    def shutdown(self):
        self.stop_event.set()
        if getattr(self, "evo_manager", None) is not None and self.evo_manager.current_transaction_id is not None:
            self.evo_manager.reset_candidates(rollback_manifest=True)
        for name, p in list(self.processes.items()):
            self._terminate_and_confirm_process(name, p)
        for name, p in list(self.external_processes.items()):
            self._terminate_and_confirm_process(name, p)
        try: self.upstream_sock.close()
        except Exception: pass
        try: self.bus.shutdown()
        except Exception: pass
        logger.info("🛑 HỆ THỐNG ĐÃ TẮT AN TOÀN.")


def _check_hard_dependencies():
    required_modules = {
        "torch": "torch>=2.3.0",
        "numpy": "numpy>=1.26.0",
        "zmq": "pyzmq>=25.0.0",
        "cv2": "opencv-python>=4.10.0",
        "loguru": "loguru>=0.7.0",
        "psutil": "psutil>=5.9.0",
        "neat": "neat-python",
        "snntorch": "snntorch",
        "torchdiffeq": "torchdiffeq",
    }

    missing = []
    for mod_name, pip_name in required_modules.items():
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        logger.critical("💀 [FATAL] THIẾU HARD DEPENDENCIES. HỆ THỐNG KHÔNG THỂ KHỞI ĐỘNG.")
        for m in missing:
            logger.critical(f"   - Thiếu: {m} (pip install {m})")
        logger.critical("   Vui lòng chạy: pip install -r requirements.txt")
        sys.exit(1)

    logger.info(f"✅ [BOOT] All {len(required_modules)} Hard Dependencies Satisfied.")

def validate_architecture_contracts():
    arch_path = os.path.join(BASE_DIR, "architecture_manifest.json")
    if not os.path.exists(arch_path):
        raise RuntimeError("Thiếu tệp architecture_manifest.json!")
    with open(arch_path, "r", encoding="utf-8") as f:
        arch = json.load(f)
    sys_topo = arch.get("system_topology", {})
    edges = sys_topo.get("data_flow_edges", [])
    modules_cfg = arch.get("modules", {})
    gov = sys_topo.get("resource_governor", {})
    if not edges or not modules_cfg:
        raise RuntimeError("architecture_manifest.json thiếu data_flow_edges hoặc modules!")

    node_caps = gov.get("max_nodes_per_module", {})
    param_caps = gov.get("max_parameters_per_module", {})
    for m_id in EvolutionTransactionManager.CANONICAL_11_MODELS:
        if m_id not in node_caps:
            raise RuntimeError(f"resource_governor.max_nodes_per_module thiếu giới hạn cho {m_id}!")
        if m_id not in param_caps:
            raise RuntimeError(f"resource_governor.max_parameters_per_module thiếu ngân sách cho {m_id}!")
    if sum(int(v) for v in param_caps.values()) > int(gov.get("global_max_parameters", 500000)):
        raise RuntimeError("Tổng ngân sách max_parameters_per_module vượt quá global_max_parameters!")

    all_mod_names = [m for m in modules_cfg.keys() if m != "ZMQ_Broker"]
    expected_in = {m: set() for m in all_mod_names}
    expected_out = {m: set() for m in all_mod_names}

    for edge in edges:
        src = edge.get("source")
        topic = edge.get("topic")
        targets = edge.get("target", [])
        if src not in modules_cfg:
            raise RuntimeError(f"Cạnh '{topic}' có source '{src}' không tồn tại trong modules!")
        expected_out[src].add(topic)
        if topic not in modules_cfg[src].get("output_contract", []):
            raise RuntimeError(f"Lệch output_contract: '{src}' phát '{topic}' trong data_flow_edges nhưng thiếu trong output_contract!")
        resolved_targets = all_mod_names if "ALL_MODULES" in targets else targets
        for dst in resolved_targets:
            if dst not in modules_cfg:
                raise RuntimeError(f"Cạnh '{topic}' có target '{dst}' không tồn tại trong modules!")
            expected_in[dst].add(topic)
            if topic not in modules_cfg[dst].get("input_contract", []):
                raise RuntimeError(f"Lệch input_contract: '{dst}' nhận '{topic}' trong data_flow_edges nhưng thiếu trong input_contract!")

    structural_runtimes = {"LiquidAuditoryNet", "LiquidAmygdalaNet", "LiquidProgressiveBrain", "ARTMemorySystem", "DynamicSOM"}
    for m_name, m_spec in modules_cfg.items():
        out_set = set(m_spec.get("output_contract", []))
        in_set = set(m_spec.get("input_contract", []))
        if out_set != expected_out.get(m_name, set()):
            raise RuntimeError(f"Module '{m_name}' có output_contract không khớp data_flow_edges: {out_set ^ expected_out.get(m_name, set())}")
        if in_set != expected_in.get(m_name, set()):
            raise RuntimeError(f"Module '{m_name}' có input_contract không khớp data_flow_edges: {in_set ^ expected_in.get(m_name, set())}")
        if m_spec.get("brain_model"):
            is_struct = bool(m_spec.get("evolution", {}).get("structural", False))
            rt_name = m_spec.get("runtime", "")
            if is_struct and rt_name not in structural_runtimes:
                raise RuntimeError(f"Module '{m_name}' (runtime={rt_name}) không hỗ trợ tiến hóa cấu trúc ở runtime nhưng khai báo evolution.structural=true!")

def validate_system_integrity():
    logger.info("🔍 [PREFLIGHT] Kiểm tra Manifest, Đồ thị Hợp đồng và Checkpoint trước khi khởi động (Fail-Closed)...")
    manifest_path = os.path.join(BASE_DIR, "brain_manifest.json")
    if not os.path.exists(manifest_path):
        logger.critical("💀 [RECOVERY_REQUIRED] Không tìm thấy brain_manifest.json! Từ chối khởi tạo lại im lặng.")
        sys.exit(1)
    try:
        validate_architecture_contracts()
        EvolutionTransactionManager()
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tx_status = data.get("transaction_status", "UNKNOWN")
        if tx_status != "COMMITTED":
            logger.critical(f"💀 [RECOVERY_REQUIRED] brain_manifest.json đang ở trạng thái '{tx_status}' (!= COMMITTED). Dừng khởi động!")
            sys.exit(1)
        for model_id in EvolutionTransactionManager.CANONICAL_11_MODELS:
            chkpt_path = CheckpointManager.get_canonical_path(model_id)
            if not os.path.exists(chkpt_path):
                logger.critical(f"💀 [RECOVERY_REQUIRED] Thiếu checkpoint chuẩn của {model_id} tại {chkpt_path}! Dừng khởi động.")
                sys.exit(1)
    except Exception as e:
        logger.critical(f"💀 [RECOVERY_REQUIRED] Lỗi xác minh tính toàn vẹn hệ thống: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    _check_hard_dependencies()
    validate_system_integrity()
    for model_id, spec in MODEL_SPECS.items():
        fmt = spec.get("checkpoint_format", "pkl")
        os.makedirs(os.path.join(BASE_DIR, "models", model_id), exist_ok=True)
    supervisor = MasterSupervisor()
    supervisor.start_all()
    if supervisor.stop_event.is_set():
        sys.exit(1)
    supervisor.run_supervisor_loop()