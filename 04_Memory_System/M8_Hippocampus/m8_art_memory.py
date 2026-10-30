#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import uuid
import json
import random
import threading
import copy
from typing import Dict, Optional
import numpy as np
import zmq
import hashlib, torch, pickle, shutil
from loguru import logger

# Import shared utilities to break circular imports
from shared_utils import BASE_DIR, MODEL_SPECS, tensor_state_hash, CheckpointManager, CheckpointError

# Import CoreBus mới (có Control Plane)
from core_bus import CoreBus

CHROMA_AVAILABLE = False
try:
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    try:
        import posthog  # type: ignore
        posthog.capture = lambda *args, **kwargs: None
        posthog.disabled = True
    except Exception:
        pass
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB không khả dụng. M8 sẽ chạy chế độ ART-only (RAM).")

def extract_data(msg):
    if not msg: return "NO_PAYLOAD"
    parts = msg.split(" ", 1)
    if len(parts) > 1:
        try:
            parsed = json.loads(parts[1])
            if isinstance(parsed, dict) and "payload" in parsed and isinstance(parsed["payload"], dict):
                for k, v in parsed["payload"].items():
                    if k not in parsed: parsed[k] = v
            return parsed
        except json.JSONDecodeError: return parts[1]
    return "NO_PAYLOAD"

def get_all_msgs(sock, bus=None):
    msgs = []
    # [BẢN VÁ P0]: Kiểm tra trực tiếp ctrl_socket gốc thay vì phụ thuộc alias dealer_socket
    if bus and hasattr(bus, 'ctrl_socket') and sock == bus.ctrl_socket:
        while True:
            try:
                frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) >= 1:
                    msg_bytes = frames[-1]
                    packet = json.loads(msg_bytes.decode('utf-8'))
                    cmd_type = packet.get("type")
                    if cmd_type == "CONTROL":
                        cmd_name = packet.get("command", "")
                        if bus and hasattr(bus, "_control_handlers") and cmd_name in bus._control_handlers:
                            res = bus._control_handlers[cmd_name](packet.get("payload", {}))
                            if packet.get("requires_ack", True):
                                det = dict(res) if isinstance(res, dict) else {}
                                st = det.pop("status", "OK") if isinstance(res, dict) else (res if isinstance(res, str) else "OK")
                                bus.send_ack(packet.get("command_id", ""), packet.get("transaction_id", ""), st, det)
                            continue
                        packet["state"] = cmd_name
                        if bus: bus.last_cmd_id = packet.get("command_id")
                    elif cmd_type == "CONTROL_ACK":
                        packet["state"] = packet.get("status", packet.get("details", {}).get("status", ""))
                    else:
                        packet["state"] = packet.get("command", "")
                    msgs.append(f"SYSTEM_CONTROL {json.dumps(packet)}")
            except zmq.Again: break
            except Exception as e:
                logger.error(f"ZMQ Control Socket Error: {e}")
                break
    else:
        if bus and hasattr(bus, '_replayed_queue') and sock == getattr(bus, 'sub_socket', None):
            while len(bus._replayed_queue) > 0:
                pkt = bus._replayed_queue.popleft()
                msgs.append(f"{pkt.get('topic', 'UNKNOWN')} {json.dumps(pkt, ensure_ascii=False)}")
        while True:
            try:
                raw_str = sock.recv_string(flags=zmq.NOBLOCK)
                if bus and hasattr(bus, '_parse_data_msg') and sock == getattr(bus, 'sub_socket', None):
                    bus._parse_data_msg(raw_str)
                msgs.append(raw_str)
            except zmq.Again:
                break
    return msgs

def _send_manual_ack(bus, status, tx_id, command_id, candidate_hash=None):
    if hasattr(bus, 'ctrl_socket'):
        ack = {
            "schema_version": 2, "type": "CONTROL_ACK", 
            "command_id": command_id, "transaction_id": tx_id, 
            "source": getattr(bus, 'node_id', 'M8_Memory'), "timestamp_ms": int(time.time() * 1000), "status": status
        }
        if candidate_hash:
            ack["details"] = {"candidate_hash": candidate_hash}
        bus._send_ctrl_msg([b'', json.dumps(ack).encode('utf-8')])

# =========================================================================
# SIMPLIFIED ART WITH STRUCTURAL EVOLUTION
# =========================================================================
class SimplifiedART:
    def __init__(self, vigilance=0.85, input_dim=128):
        self.vigilance = vigilance
        self.input_dim = input_dim
        # Node: {"weight": list[float], "count": int, "created_at": float, "last_accessed": float}
        self.memory_nodes = []
        self.functional_revision = 1
        self.topology_version = 1

    def _get_governor_limits(self) -> Dict[str, float]:
        limits = {
            "structural": True,
            "max_nodes": 2000,
            "max_parameters": 50000,
            "global_max_parameters": 500000,
            "max_rss_mb": 2048.0
        }
        arch_path = os.path.join(BASE_DIR, "architecture_manifest.json")
        if os.path.exists(arch_path):
            try:
                with open(arch_path, "r", encoding="utf-8") as f:
                    arch = json.load(f)
                gov = arch.get("system_topology", {}).get("resource_governor", arch.get("governor", {}))
                m8_cfg = arch.get("modules", {}).get("M8_Memory", {})
                evo_cfg = m8_cfg.get("evolution", {})
                limits["structural"] = bool(evo_cfg.get("structural", True))
                limits["max_nodes"] = min(
                    int(gov.get("max_nodes_per_module", {}).get("M8_Memory", limits["max_nodes"])),
                    int(evo_cfg.get("max_nodes", limits["max_nodes"]))
                )
                limits["max_parameters"] = min(
                    int(gov.get("max_parameters_per_module", {}).get("M8_Memory", limits["max_parameters"])),
                    int(evo_cfg.get("max_parameters", gov.get("max_params_per_model", limits["max_parameters"])))
                )
                limits["global_max_parameters"] = int(gov.get("global_max_parameters", limits["global_max_parameters"]))
                limits["max_rss_mb"] = float(gov.get("max_process_rss_mb", limits["max_rss_mb"]))
            except Exception:
                pass
        return limits

    def _can_grow_topology(self, target_nodes: int):
        limits = self._get_governor_limits()
        if not limits["structural"]:
            return False, "evolution.structural=false trong manifest"
        if target_nodes > limits["max_nodes"]:
            return False, f"Vượt giới hạn node của Governor ({target_nodes} > {limits['max_nodes']})"
        projected_params = target_nodes * self.input_dim
        if projected_params > max(limits["max_parameters"], limits["max_nodes"] * self.input_dim) or projected_params > limits["global_max_parameters"]:
            return False, f"Vượt ngân sách tham số Governor ({projected_params})"
        try:
            import psutil
            vm = psutil.virtual_memory()
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
            if vm.percent >= 90.0 or rss_mb >= limits["max_rss_mb"]:
                return False, f"Vượt ngưỡng RAM an toàn (SysRAM={vm.percent:.1f}%, ProcRSS={rss_mb:.1f}MB)"
        except Exception:
            pass
        return True, "OK"

    def process(self, vector, is_query=False):
        try:
            vector_np = np.array(vector, dtype=np.float32).flatten()
        except Exception:
            return None, "INVALID_INPUT"
            
        if not np.isfinite(vector_np).all():
            return None, "INVALID_INPUT"
            
        norm = np.linalg.norm(vector_np)
        if norm == 0: 
            return None, "BLANK"
        vector_np = vector_np / norm
        
        if len(vector_np) != self.input_dim:
            return None, "DIMENSION_MISMATCH"
            
        best_node = None
        max_sim = -1.0
        for node in self.memory_nodes:
            node_np = np.array(node["weight"]).flatten()
            if len(vector_np) != len(node_np): continue
            sim = float(np.dot(vector_np, node_np))
            if sim > max_sim: 
                max_sim = sim
                best_node = node
                
        if max_sim >= self.vigilance and best_node is not None:
            if not is_query:
                alpha = 0.1
                updated = (1 - alpha) * np.array(best_node["weight"]) + alpha * vector_np
                norm_up = np.linalg.norm(updated)
                if norm_up > 0:
                    best_node["weight"] = (updated / norm_up).tolist()
                best_node["count"] += 1
                best_node["last_accessed"] = time.time()
            return best_node.get("node_id"), "UPDATE"
        else:
            if is_query:
                return None, "NOT_FOUND"
            can_grow, reason = self._can_grow_topology(len(self.memory_nodes) + 1)
            if not can_grow:
                if best_node is not None:
                    alpha = 0.05
                    updated = (1 - alpha) * np.array(best_node["weight"]) + alpha * vector_np
                    norm_up = np.linalg.norm(updated)
                    if norm_up > 0:
                        best_node["weight"] = (updated / norm_up).tolist()
                    best_node["count"] += 1
                    best_node["last_accessed"] = time.time()
                    return best_node.get("node_id"), "UPDATE"
                logger.warning(f"⚠️ [M8_Memory] Từ chối tạo node ART mới: {reason}")
                return None, "GOVERNOR_CAP_REACHED"
            now = time.time()
            new_id = str(uuid.uuid4())
            self.memory_nodes.append({"node_id": new_id, "weight": vector_np.tolist(), "count": 1, "created_at": now, "last_accessed": now})
            self.functional_revision += 1
            self.topology_version += 1
            return new_id, "CREATE"

    def get_state(self):
        return {
            "vigilance": self.vigilance,
            "input_dim": self.input_dim,
            "functional_revision": self.functional_revision,
            "topology_version": self.topology_version,
            "nodes": copy.deepcopy(self.memory_nodes)
        }

    def load_state(self, state):
        if isinstance(state, dict):
            backup_nodes = copy.deepcopy(self.memory_nodes)
            backup_vig = self.vigilance
            backup_dim = self.input_dim
            backup_rev = self.functional_revision
            backup_topo = self.topology_version
            try:
                self.vigilance = float(state.get("vigilance", 0.85))
                self.input_dim = int(state.get("input_dim", 128))
                self.functional_revision = int(state.get("functional_revision", 1))
                self.topology_version = int(state.get("topology_version", self.topology_version))
                nodes = copy.deepcopy(state.get("nodes", []))
                valid_nodes = []
                for n in nodes:
                    w_arr = np.array(n.get("weight", []), dtype=np.float32).flatten()
                    if w_arr.size != self.input_dim or not np.isfinite(w_arr).all():
                        continue
                    if "last_accessed" not in n:
                        n["last_accessed"] = n.get("created_at", time.time())
                    if "node_id" not in n:
                        w_str = ",".join(f"{float(x):.5f}" for x in w_arr)
                        w_hash = hashlib.md5(w_str.encode('utf-8')).hexdigest()
                        n["node_id"] = f"legacy_{w_hash[:16]}"
                    valid_nodes.append(n)
                self.memory_nodes = valid_nodes
            except Exception as e:
                self.memory_nodes = backup_nodes
                self.vigilance = backup_vig
                self.input_dim = backup_dim
                self.functional_revision = backup_rev
                self.topology_version = backup_topo
                raise e

    # --- Structural Evolution Helpers ---
    def _cosine_sim(self, v1, v2):
        v1 = np.array(v1); v2 = np.array(v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def consolidate_merge(self, merge_threshold=0.95):
        merged = 0
        merged_map = {}
        i = 0
        while i < len(self.memory_nodes):
            j = i + 1
            while j < len(self.memory_nodes):
                sim = self._cosine_sim(self.memory_nodes[i]["weight"], self.memory_nodes[j]["weight"])
                if sim > merge_threshold:
                    n1, n2 = self.memory_nodes[i], self.memory_nodes[j]
                    total_count = n1["count"] + n2["count"]
                    if total_count > 0:
                        w1 = np.array(n1["weight"]); w2 = np.array(n2["weight"])
                        new_w = (w1 * n1["count"] + w2 * n2["count"]) / total_count
                        norm_w = np.linalg.norm(new_w)
                        if norm_w > 0:
                            n1["weight"] = (new_w / norm_w).tolist()
                        n1["count"] = total_count
                        n1["last_accessed"] = max(n1["last_accessed"], n2["last_accessed"])
                    if n2.get("node_id") and n1.get("node_id"):
                        merged_map[n2["node_id"]] = n1["node_id"]
                    self.memory_nodes.pop(j)
                    merged += 1
                else:
                    j += 1
            i += 1
        self.last_merged_map = merged_map
        return merged

    def prune_stale(self, max_age_sec=86400*30, min_count=1):
        now = time.time()
        original_len = len(self.memory_nodes)
        self.memory_nodes = [
            n for n in self.memory_nodes 
            if n["count"] > min_count or (now - n.get("last_accessed", now)) < max_age_sec
        ]
        return original_len - len(self.memory_nodes)

    def split_broad_categories(self, max_count_before_split=100):
        split_count = 0
        new_nodes = []
        for node in self.memory_nodes:
            if node["count"] > max_count_before_split:
                can_grow, reason = self._can_grow_topology(len(self.memory_nodes) + len(new_nodes) + 1)
                if not can_grow:
                    logger.warning(f"⚠️ [M8_Memory] Dừng split_broad_categories do chạm giới hạn Governor: {reason}")
                    break
                w = np.array(node["weight"])
                noise = np.random.randn(*w.shape) * 0.05
                child_w = w + noise
                norm_c = np.linalg.norm(child_w)
                if norm_c > 0:
                    child_w = child_w / norm_c
                
                new_nodes.append({
                    "node_id": str(uuid.uuid4()),
                    "weight": child_w.tolist(),
                    "count": max(1, node["count"] // 2),
                    "created_at": time.time(),
                    "last_accessed": time.time()
                })
                node["count"] = max(1, node["count"] // 2)
                split_count += 1
        self.memory_nodes.extend(new_nodes)
        return split_count

    def simulate_evolution(self, trigger="prepare"):
        log = {"trigger": trigger, "actions": [], "topology_changed": False, "merged_map": {}}
        limits = self._get_governor_limits()
        if not limits["structural"]:
            return log

        backup_state = self.get_state()
        try:
            merged = self.consolidate_merge(merge_threshold=0.95)
            if merged > 0:
                log["actions"].append(f"merge_{merged}_categories")
                log["merged_map"] = dict(getattr(self, "last_merged_map", {}))

            pruned = self.prune_stale(max_age_sec=86400*30, min_count=2)
            if pruned > 0: log["actions"].append(f"prune_{pruned}_stale_nodes")

            split = self.split_broad_categories(max_count_before_split=100)
            if split > 0: log["actions"].append(f"split_{split}_broad_categories")

            max_nodes = int(limits["max_nodes"])
            if len(self.memory_nodes) > max_nodes:
                self.memory_nodes.sort(key=lambda n: (n.get("count", 1), n.get("last_accessed", 0.0)), reverse=True)
                excess = len(self.memory_nodes) - max_nodes
                self.memory_nodes = self.memory_nodes[:max_nodes]
                pruned += excess
                log["actions"].append(f"governor_cap_prune_{excess}_nodes")

            if merged > 0 or pruned > 0 or split > 0:
                log["topology_changed"] = True
                self.topology_version += 1

            current_nodes = len(self.memory_nodes)
            if current_nodes > max_nodes * 0.9:
                self.vigilance = min(0.99, self.vigilance + 0.01)
                log["actions"].append(f"vigilance_up_{self.vigilance:.3f}")
            elif current_nodes < max_nodes * 0.5:
                self.vigilance = max(0.6, self.vigilance - 0.01)
                log["actions"].append(f"vigilance_down_{self.vigilance:.3f}")

            if log["actions"]:
                self.functional_revision += 1
                log["functional_revision"] = self.functional_revision
                log["topology_version"] = self.topology_version
                log["node_count"] = len(self.memory_nodes)
            return log
        except Exception as e:
            logger.error(f"❌ [M8_Memory] Lỗi khi đột biến cấu trúc ART ({e}), tự động khôi phục trạng thái trước đó!")
            self.load_state(backup_state)
            return {"trigger": trigger, "actions": [], "topology_changed": False, "merged_map": {}}

    def apply_plasticity(self, mod_signal: float, vector: np.ndarray = None):
        if abs(mod_signal) > 1e-4:
            if mod_signal > 0:
                self.vigilance = min(0.98, self.vigilance + 0.01 * mod_signal)
            else:
                self.vigilance = max(0.5, self.vigilance + 0.005 * mod_signal)
    
            if vector is not None and len(self.memory_nodes) > 0:
                vector_np = np.array(vector, dtype=np.float32).flatten()
                norm = np.linalg.norm(vector_np)
                if norm > 0:
                    vector_np = vector_np / norm
            
                    best_node, max_sim = None, -1.0
                    for node in self.memory_nodes:
                        node_np = np.array(node["weight"]).flatten()
                        if len(vector_np) != len(node_np): continue
                        sim = float(np.dot(vector_np, node_np))
                        if sim > max_sim:
                            max_sim = sim
                            best_node = node
            
                    if best_node is not None and max_sim > 0.5:
                        alpha = 0.05 * abs(mod_signal)
                        updated = (1 - alpha) * np.array(best_node["weight"]) + alpha * vector_np
                        norm_u = np.linalg.norm(updated)
                        if norm_u > 0:
                            best_node["weight"] = (updated / norm_u).tolist()
                        best_node["last_accessed"] = time.time()
            self.functional_revision += 1


# =========================================================================
# HIPPOCAMPUS MEMORY NODE (M8)
# =========================================================================
class HippocampusMemory:
    def __init__(self, node_id="M8_Memory"):
        self.node_id = node_id
        self.bus = CoreBus(self.node_id, is_master=False)
        self.art = SimplifiedART()
        self.working_art = SimplifiedART()
        self.generation = 0
        self.is_sleeping = False
        self.topology_version = 1
        self.mutation_log = []
        self.load_mode = "READY"
        self.worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
        
        self._evo_state = "LIVE"
        self._candidate_art_state = None
        self._candidate_topology_version = 0
        self._candidate_generation = 0
        self._current_transaction_id = None

        self.pending_memories = []
        self._staged_tx_memories: Dict[str, list] = {}
        self.collection = None
        if CHROMA_AVAILABLE:
            db_path = os.path.join(BASE_DIR, "models", self.node_id, "chroma_db")
            os.makedirs(db_path, exist_ok=True)
            try:
                from chromadb.config import Settings
                chroma_settings = Settings(anonymized_telemetry=False)
                self.chroma_client = chromadb.PersistentClient(path=db_path, settings=chroma_settings)
                self.collection = self.chroma_client.get_or_create_collection(name="episodic_memory")
                logger.info(f"🟢 {self.node_id}: ChromaDB connected at {db_path}")
            except Exception as e:
                logger.warning(f"⚠️ {self.node_id}: ChromaDB schema cũ ({e}), đang làm mới tại {db_path}...")
                try:
                    shutil.rmtree(db_path, ignore_errors=True)
                    os.makedirs(db_path, exist_ok=True)
                    from chromadb.config import Settings
                    self.chroma_client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
                    self.collection = self.chroma_client.get_or_create_collection(name="episodic_memory")
                    logger.info(f"🟢 {self.node_id}: ChromaDB recreated and connected at {db_path}")
                except Exception as e2:
                    logger.error(f"🔴 {self.node_id}: ChromaDB init failed: {e2}. Fallback ART-only.")
                    self.collection = None
        else:
            logger.warning(f"⚠️ {self.node_id}: Running ART-only (RAM) mode.")

        self.parent_art_state = None
        self.parent_generation = 0
        self.parent_topology_version = 1

        self.training_lock = threading.Lock()
        import queue
        self.training_queue = queue.Queue(maxsize=100)
        self.training_thread = threading.Thread(target=self._async_train_encoder, daemon=True)
        self.training_thread.start()

        self._load_checkpoint()
        self._reconcile_chroma_db()

        self.bus.subscribe(["QUERY_MEM", "SAVE_MEM", "THALAMUS_M8", "THALAMUS_TOPDOWN", "CEREBROSPINAL_FLUID", "GLOBAL_PLASTICITY", "SYSTEM_CONTROL"])
        self.upstream_sock = self.bus.context.socket(zmq.PUSH)
        self.upstream_sock.connect("tcp://127.0.0.1:5554")
        logger.info(f"🟢 {self.node_id} chuẩn bị hòa mạng...")

    def _reconcile_chroma_db(self, rollback_tx_id: str = None, merged_map: Optional[Dict[str, str]] = None):
        """Đồng bộ hóa ChromaDB với thế hệ đã commit, ánh xạ lại các node gộp và xóa node mồ côi."""
        if self.collection is None:
            return
        try:
            if rollback_tx_id:
                self.collection.delete(where={"transaction_id": rollback_tx_id})
            self.collection.delete(where={"generation": {"$gt": self.generation}})
            all_records = self.collection.get(include=["metadatas"])
            if all_records and all_records.get("ids"):
                active_node_ids = {n.get("node_id") for n in self.art.memory_nodes if n.get("node_id")}
                orphan_ids = []
                remap_ids, remap_metas = [], []
                for doc_id, meta in zip(all_records["ids"], all_records.get("metadatas", [])):
                    if isinstance(meta, dict):
                        node_ref = meta.get("art_node")
                        if merged_map and node_ref in merged_map and merged_map[node_ref] in active_node_ids:
                            new_meta = dict(meta)
                            new_meta["art_node"] = merged_map[node_ref]
                            remap_ids.append(doc_id)
                            remap_metas.append(new_meta)
                        elif node_ref and node_ref not in active_node_ids:
                            orphan_ids.append(doc_id)
                if remap_ids:
                    self.collection.update(ids=remap_ids, metadatas=remap_metas)
                if orphan_ids:
                    self.collection.delete(ids=orphan_ids)
                    logger.info(f"🧹 {self.node_id}: Đã dọn {len(orphan_ids)} bản ghi ChromaDB mồ côi không khớp với ART.")
        except Exception as e:
            logger.warning(f"⚠️ {self.node_id}: ChromaDB reconciliation warning: {e}")

    def _serialize_optimizer_state(self) -> Optional[Dict]:
        if not hasattr(self, 'encoder_optimizer') or self.encoder_optimizer is None:
            return getattr(self, '_encoder_optimizer_state', None)
        raw_opt = self.encoder_optimizer.state_dict()
        ser_state = {}
        for k, v in raw_opt.get("state", {}).items():
            ser_state[str(k)] = {}
            for sk, sv in v.items():
                ser_state[str(k)][sk] = sv.detach().cpu().numpy().tolist() if torch.is_tensor(sv) else copy.deepcopy(sv)
        return {"state": ser_state, "param_groups": copy.deepcopy(raw_opt.get("param_groups", []))}

    def _restore_encoder_and_optimizer(self, enc_state: Optional[Dict], opt_state: Optional[Dict]):
        if not hasattr(self, 'neural_encoder') or self.neural_encoder is None:
            self._encoder_state = copy.deepcopy(enc_state)
            self._encoder_optimizer_state = copy.deepcopy(opt_state)
            return
        with self.training_lock:
            if isinstance(enc_state, dict) and enc_state:
                loaded_enc = {
                    k: (v.to(self.device) if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32, device=self.device))
                    for k, v in enc_state.items()
                }
                self.neural_encoder.load_state_dict(loaded_enc)
            if isinstance(opt_state, dict) and opt_state and hasattr(self, 'encoder_optimizer'):
                restored_opt = {"state": {}, "param_groups": copy.deepcopy(opt_state.get("param_groups", self.encoder_optimizer.state_dict()["param_groups"]))}
                for k, v in opt_state.get("state", {}).items():
                    idx_k = int(k) if str(k).isdigit() else k
                    restored_opt["state"][idx_k] = {}
                    for sk, sv in v.items():
                        if isinstance(sv, list):
                            restored_opt["state"][idx_k][sk] = torch.tensor(sv, dtype=torch.float32, device=self.device)
                        elif torch.is_tensor(sv):
                            restored_opt["state"][idx_k][sk] = sv.to(self.device)
                        else:
                            restored_opt["state"][idx_k][sk] = copy.deepcopy(sv)
                try:
                    self.encoder_optimizer.load_state_dict(restored_opt)
                except Exception as e:
                    logger.warning(f"⚠️ {self.node_id}: Khôi phục encoder_optimizer state thất bại: {e}")

    def _load_checkpoint(self):
        try:
            chkpt = CheckpointManager.load(self.node_id, is_torch=False, validate_hash=True, enforce_manifest=True)
            if chkpt:
                self.art.load_state(chkpt.get("model_state", {}))
                self.generation = chkpt.get("generation", 0)
                self.topology_version = chkpt.get("topology_version", chkpt.get("architecture_version", self.art.topology_version))
                self.art.topology_version = self.topology_version
                self.working_art.load_state(self.art.get_state())
                
                raw_log = chkpt.get("mutation_log", [])
                self.mutation_log = raw_log if isinstance(raw_log, list) else [raw_log]
                extra = chkpt.get("extra_states", chkpt)

                self.parent_art_state = extra.get("parent_art_state")
                self.parent_generation = extra.get("parent_generation", self.generation)
                self.parent_topology_version = extra.get("parent_topology_version", self.topology_version)
                self.pending_memories = []
                self._encoder_state = extra.get("encoder_state")
                self._encoder_optimizer_state = extra.get("encoder_optimizer_state")
                self.load_mode = "READY"
                logger.info(f"🟢 {self.node_id}: Loaded Gen={self.generation}, TopoVer={self.topology_version}, Nodes={len(self.art.memory_nodes)}")
            else:
                if os.environ.get("ASUNA_ALLOW_BOOTSTRAP", "0") != "1":
                    raise RuntimeError(f"RECOVERY_REQUIRED:{self.node_id}:Missing canonical checkpoint")
                self.load_mode = "DEGRADED"
                logger.warning(f"🆕 [BOOTSTRAP_NEW_MODEL] {self.node_id}: Khởi tạo MODEL MỚI (DEGRADED).")
        except CheckpointError as e:
            logger.critical(f"🚨 [RECOVERY_REQUIRED] {self.node_id}: Checkpoint lỗi toàn vẹn ({e})!")
            raise RuntimeError(f"RECOVERY_REQUIRED:{self.node_id}:{e}")

    def _save_checkpoint(self, is_dirty=True, force=False, as_candidate=False, tx_id=None):
        if not is_dirty and not force: return False
        
        encoder_st = getattr(self, '_encoder_state', None)
        encoder_opt_st = getattr(self, '_encoder_optimizer_state', None)
        if hasattr(self, 'neural_encoder'):
            with self.training_lock:
                encoder_st = {k: v.detach().cpu().numpy().tolist() for k, v in self.neural_encoder.state_dict().items()}
                encoder_opt_st = self._serialize_optimizer_state()

        target_state = self._candidate_art_state if (as_candidate and self._candidate_art_state is not None) else self.art.get_state()
        target_gen = self._candidate_generation if as_candidate else self.generation
        target_topo_ver = self._candidate_topology_version if as_candidate else self.topology_version
        parent_hash = tensor_state_hash(self.art.get_state())
        cand_hash = tensor_state_hash(target_state)

        extra_st = {
            "is_candidate": bool(as_candidate),
            "phase": "PREPARED" if as_candidate else "COMMITTED",
            "transaction_id": tx_id,
            "candidate_transaction_id": tx_id if as_candidate else None,
            "parent_state_hash": parent_hash,
            "candidate_hash": cand_hash if as_candidate else None,
            "topology": {"name": "ART_Memory", "vigilance": target_state.get("vigilance", 0.85), "node_count": len(target_state.get("nodes", []))},
            "generation": target_gen,
            "parent_generation": self.generation if as_candidate else self.parent_generation,
            "topology_version": target_topo_ver,
            "mutation_log": self.mutation_log[-10:],
            "pending_memories": self._staged_tx_memories.get(tx_id, []) if as_candidate else [],
            "parent_art_state": self.parent_art_state,
            "parent_topology_version": self.parent_topology_version,
            "candidate_art_state": None,
            "candidate_topology_version": 0,
            "candidate_generation": 0,
            "encoder_state": encoder_st,
            "encoder_optimizer_state": encoder_opt_st,
            "worker_epoch": self.worker_epoch
        }
        return CheckpointManager.save(self.node_id, target_state, is_torch=False, extra_states=extra_st)

    def _async_train_encoder(self):
        import torch
        import torch.nn.functional as F
        while True:
            tensor_in = self.training_queue.get()
            if tensor_in is None: break
            
            try:
                if getattr(self, '_evo_state', 'LIVE') != 'LIVE':
                    continue
                with self.training_lock:
                    if not hasattr(self, 'neural_encoder') or not hasattr(self, 'encoder_optimizer'):
                        continue
                    train_model = copy.deepcopy(self.neural_encoder)
                    train_opt = type(self.encoder_optimizer)(train_model.parameters(), lr=0.001)
                    train_opt.load_state_dict(self.encoder_optimizer.state_dict())
                
                train_model.train()
                train_opt.zero_grad()
                noisy_in = tensor_in + torch.randn_like(tensor_in) * 0.05
                out_learn = train_model(noisy_in)
                loss = 1.0 - F.cosine_similarity(out_learn, tensor_in).mean()
                loss.backward()
                train_opt.step()
                
                new_active_model = copy.deepcopy(train_model)
                new_active_model.eval()
                
                with self.training_lock:
                    if getattr(self, '_evo_state', 'LIVE') == 'LIVE':
                        self.neural_encoder = new_active_model
                        self.encoder_optimizer.load_state_dict(train_opt.state_dict())
            except Exception as e:
                logger.error(f"Async training error: {e}")
            finally:
                self.training_queue.task_done()

    def encode_semantic(self, data, dim=128):
        # [STEP 1: CANONICAL 128D MODALITY-SEPARATED REPRESENTATION]
        if isinstance(data, dict):
            payload = data.get("payload", data)
            if isinstance(payload, dict):
                if "fused_latent" in payload and len(payload["fused_latent"]) == dim:
                    return payload["fused_latent"]
                if "semantic_embedding" in payload and len(payload["semantic_embedding"]) == dim:
                    return payload["semantic_embedding"]
                if "latent_vector" in payload and len(payload["latent_vector"]) == dim:
                    return payload["latent_vector"]
                context = payload.get("context", {})
                if isinstance(context, dict):
                    if "fused_latent" in context and len(context["fused_latent"]) == dim:
                        return context["fused_latent"]
                    if "semantic_embedding" in context and len(context["semantic_embedding"]) == dim:
                        return context["semantic_embedding"]
        text_str = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
        normalized_text = re.sub(r'[^\w\s]', ' ', text_str.lower())
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        raw_vec = np.zeros(dim, dtype=np.float32)
        
        words = normalized_text.split()
        for w in words:
            idx = sum(ord(c) for c in w) % dim
            raw_vec[idx] += len(w)
            
        for i in range(len(normalized_text) - 2):
            trigram = normalized_text[i:i+3]
            idx = sum(ord(c) * (pos+1) for pos, c in enumerate(trigram)) % dim
            raw_vec[idx] += 1.0
            
        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            raw_vec = (raw_vec / norm).tolist()
        else:
            digest = hashlib.sha256(text_str.encode('utf-8')).digest()
            det_vals = np.array([(float(b) / 127.5) - 1.0 for b in digest], dtype=np.float32)
            tiled = np.resize(det_vals, dim)
            det_norm = float(np.linalg.norm(tiled))
            raw_vec = (tiled / det_norm).tolist() if det_norm > 0 else ([1.0 / np.sqrt(dim)] * dim)
        
        # 2. Khởi tạo Neural Encoder (Lazy Init) có kiểm tra ngân sách Governor và khôi phục Optimizer
        if not hasattr(self, 'neural_encoder'):
            with self.training_lock:
                if not hasattr(self, 'neural_encoder'):
                    import torch.nn as nn
                    import torch.optim as optim
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    candidate_encoder = nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.GELU(),
                        nn.Linear(dim, dim)
                    ).to(self.device)
                    limits = self.art._get_governor_limits()
                    enc_params = sum(p.numel() for p in candidate_encoder.parameters())
                    max_allowed_params = min(int(limits["max_parameters"]), int(limits["global_max_parameters"]))
                    if enc_params > max_allowed_params:
                        raise RuntimeError(f"[{self.node_id}] Neural Encoder vượt ngân sách tham số Governor ({enc_params} > {max_allowed_params})!")
                    self.neural_encoder = candidate_encoder
                    self.encoder_optimizer = optim.AdamW(self.neural_encoder.parameters(), lr=1e-4)
            if getattr(self, '_encoder_state', None) or getattr(self, '_encoder_optimizer_state', None):
                self._restore_encoder_and_optimizer(
                    getattr(self, '_encoder_state', None),
                    getattr(self, '_encoder_optimizer_state', None)
                )
        
        # 3. [BẢN VÁ TỐI THƯỢNG]: Push to async queue instead of blocking
        tensor_in = torch.tensor([raw_vec], dtype=torch.float32, device=self.device)
        try:
            self.training_queue.put_nowait(tensor_in)
        except Exception: pass # Ignore if queue full

        # 4. [STEP 1.6 FIX - LOCK-FREE CONCURRENCY]: Chạy qua Mạng Neural (Inference Phase)
        # Capture the atomic reference to avoid race conditions with trainer thread
        active_encoder = self.neural_encoder
        active_encoder.eval()
        with torch.no_grad():
            tensor_out = active_encoder(tensor_in)
            final_vec = tensor_out[0].cpu().numpy().tolist()
            
        return final_vec

    # =========================================================================
    # CHROMA HELPERS
    # =========================================================================
    def _commit_chroma_pending(self, target_generation: int = None, tx_id: str = None) -> bool:
        """Chỉ ghi các bản ghi ứng viên của tx_id vào ChromaDB khi nhận GLOBAL_COMMIT."""
        staged_list = self._staged_tx_memories.get(tx_id, []) if tx_id else self.pending_memories
        merged_map = getattr(self, "_candidate_merged_map", {}) or {}
        if not self.collection:
            if tx_id: self._staged_tx_memories.pop(tx_id, None)
            return True
        try:
            active_node_ids = {n.get("node_id") for n in self.art.memory_nodes if n.get("node_id")}
            valid_memories = []
            for m in staged_list:
                node_ref = m.get("meta", {}).get("art_node")
                if node_ref in merged_map:
                    node_ref = merged_map[node_ref]
                    m["meta"]["art_node"] = node_ref
                if node_ref in active_node_ids:
                    valid_memories.append(m)
            if valid_memories:
                for m in valid_memories:
                    if target_generation is not None:
                        m["meta"]["generation"] = target_generation
                    if tx_id is not None:
                        m["meta"]["transaction_id"] = tx_id
                ids = [m["id"] for m in valid_memories]
                embeddings = [m["vector"] for m in valid_memories]
                documents = [m["doc"] for m in valid_memories]
                metadatas = [m["meta"] for m in valid_memories]
                self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                logger.debug(f"💾 {self.node_id}: Committed {len(valid_memories)} records to ChromaDB (gen={target_generation}, tx={tx_id}).")
            if tx_id:
                self._staged_tx_memories.pop(tx_id, None)
            self._reconcile_chroma_db(merged_map=merged_map)
            self._candidate_merged_map = {}
            return True
        except Exception as e:
            logger.error(f"🔴 {self.node_id}: ChromaDB commit failed: {e}")
            return False

    def _clear_pending(self):
        if self.pending_memories:
            logger.warning(f"⏪ {self.node_id}: Rollback {len(self.pending_memories)} pending Chroma memories.")
            self.pending_memories.clear()

    # =========================================================================
    # MAIN LOOP (CHUYỂN CHUNG CONTROL PLANE VỚI 10 MODEL KHÁC)
    # =========================================================================
    def run(self, stop_event=None):
        try:
            self.upstream_sock.send_string(
                f"UPSTREAM {json.dumps({'state': 'READY', 'module': self.node_id, 'schema_verified': True, 'load_mode': self.load_mode, 'worker_epoch': self.worker_epoch})}",
                flags=zmq.NOBLOCK
            )
            self.bus.publish("SYSTEM_CONTROL", {"state": "HEARTBEAT", "module": self.node_id, "worker_epoch": self.worker_epoch, "modality": "control"})
        except Exception as e:
            logger.warning(f"⚠️ {self.node_id}: Lỗi gửi tín hiệu READY ban đầu: {e}")
            
        last_dream = time.time()
        last_save = time.time()
        last_heartbeat = time.time()
        local_episode = "UNKNOWN"
        local_tick = 0
        is_dirty = False
        
        poller = zmq.Poller()
        poller.register(self.bus.sub_socket, zmq.POLLIN)
        if hasattr(self.bus, 'ctrl_socket'): poller.register(self.bus.ctrl_socket, zmq.POLLIN)
        
        while True:
            if stop_event and stop_event.is_set():
                logger.info(f"🛑 {self.node_id}: Stop event received. Shutting down without overwriting committed state...")
                break
                
            socks = dict(poller.poll(50))
            for sock in socks:
                for msg in get_all_msgs(sock, self.bus):
                    topic = msg.split(" ", 1)[0]
                    packet = extract_data(msg)
                    if not isinstance(packet, dict): continue
                    payload = packet.get("payload", packet)
                    event_id = packet.get("event_id")

                    if topic == "CEREBROSPINAL_FLUID":
                        local_episode = packet.get("episode_id", local_episode)
                        local_tick = packet.get("tick_id", local_tick)

                    elif topic == "SYSTEM_CONTROL":
                        st = packet.get("state", "")
                        if st == "ASLEEP": self.is_sleeping = True
                        elif st in ["AWAKE", "START_TRAINING", "AWAKE_AND_SAFE", "RESUME"]: self.is_sleeping = False
                        elif st == "EVOLUTION_PREPARE":
                            tx_id = packet.get("transaction_id", f"tx_{local_tick}")
                            command_id = packet.get("command_id", packet.get("event_id", "UNKNOWN"))
                            p_load = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
                            target_gen = p_load.get("target_generation", self.generation + 1)

                            if self._evo_state == "CANDIDATE_STAGED" and self._current_transaction_id == tx_id and self._candidate_art_state is not None:
                                h = getattr(self, "_last_prepared_hash", tensor_state_hash(self._candidate_art_state))
                                _send_manual_ack(self.bus, "PREPARED", tx_id, command_id, candidate_hash=h)
                                continue

                            self.parent_art_state = copy.deepcopy(self.art.get_state())
                            self.parent_generation = self.generation
                            self.parent_topology_version = self.topology_version
                            if hasattr(self, 'neural_encoder'):
                                with self.training_lock:
                                    self._parent_encoder_sd = copy.deepcopy(self.neural_encoder.state_dict())
                                    self._parent_encoder_opt_sd = copy.deepcopy(self.encoder_optimizer.state_dict()) if hasattr(self, 'encoder_optimizer') else None
                            
                            candidate_art = SimplifiedART()
                            candidate_art.load_state(self.working_art.get_state())
                            
                            evo_log = candidate_art.simulate_evolution(trigger="prepare")
                            self._candidate_merged_map = evo_log.get("merged_map", {})
                            if evo_log.get("actions"):
                                self.mutation_log.append({"timestamp": time.time(), "generation": target_gen, **evo_log})
                            
                            prev_node_ids = {n.get("node_id") for n in self.art.memory_nodes}
                            cand_node_ids = {n.get("node_id") for n in candidate_art.memory_nodes}
                            structural_changed = bool(evo_log.get("topology_changed", False) or (prev_node_ids != cand_node_ids))
                            self._candidate_topology_version = max(
                                self.topology_version + (1 if structural_changed else 0),
                                candidate_art.topology_version
                            )
                            candidate_art.topology_version = self._candidate_topology_version
                            self._candidate_art_state = candidate_art.get_state()
                            self.working_art.load_state(self._candidate_art_state)
                            self._candidate_generation = target_gen
                            self._evo_state = "CANDIDATE_STAGED"
                            self._current_transaction_id = tx_id

                            staged_for_tx = []
                            for m in self.pending_memories:
                                m_copy = copy.deepcopy(m)
                                m_copy["meta"]["transaction_id"] = tx_id
                                staged_for_tx.append(m_copy)
                            self._staged_tx_memories.clear()
                            self._staged_tx_memories[tx_id] = staged_for_tx
                            self.pending_memories.clear()
                            
                            self._save_checkpoint(is_dirty=True, force=True, as_candidate=True, tx_id=tx_id)
                            raw_h = tensor_state_hash(self._candidate_art_state)
                            parent_h = tensor_state_hash(self.art.get_state())
                            h = f"UNCHANGED_{raw_h}" if (raw_h == parent_h and not staged_for_tx) else raw_h
                            self._last_prepared_hash = h
                            
                            try: self.upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'CANDIDATE_READY', 'module': self.node_id, 'candidate_hash': h, 'transaction_id': tx_id})}", flags=zmq.NOBLOCK)
                            except Exception as e: logger.warning(f"⚠️ Upstream send failed: {e}")
                            self.bus.publish("SYSTEM_CONTROL", {"state": "CANDIDATE_READY", "module": self.node_id, "candidate_hash": h, "transaction_id": tx_id, "modality": "evolution"})
                            _send_manual_ack(self.bus, "PREPARED", tx_id, command_id, candidate_hash=h)
                            
                        elif st == "GLOBAL_COMMIT":
                            tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                            command_id = packet.get("command_id", packet.get("event_id", "UNKNOWN"))
                            payload_dict = packet.get("payload", {}) if isinstance(packet, dict) else {}
                            target_gen = payload_dict.get("brain_generation", getattr(self, '_candidate_generation', self.generation + 1))

                            if getattr(self, '_last_committed_tx_id', None) == tx_id:
                                _send_manual_ack(self.bus, "COMMIT_APPLIED", tx_id, command_id)
                                continue
                            
                            if self._evo_state == "CANDIDATE_STAGED" and getattr(self, '_current_transaction_id', None) == tx_id:
                                if getattr(self, '_candidate_art_state', None) is not None:
                                    self.topology_version = getattr(self, '_candidate_topology_version', self.topology_version)
                                    self.art.load_state(self._candidate_art_state)
                                    self.art.topology_version = self.topology_version
                                    if not self.pending_memories:
                                        self.working_art.load_state(self.art.get_state())
                                    self.generation = target_gen
                                    
                                chroma_ok = self._commit_chroma_pending(target_generation=self.generation, tx_id=tx_id)
                                if chroma_ok:
                                    self._candidate_art_state = None
                                    self._current_transaction_id = None
                                    self._evo_state = "LIVE"
                                    self._last_committed_tx_id = tx_id
                                    is_dirty = False
                                    last_save = time.time()
                                    _send_manual_ack(self.bus, "COMMIT_APPLIED", tx_id, command_id)
                                else:
                                    if getattr(self, 'parent_art_state', None) is not None:
                                        self.art.load_state(self.parent_art_state)
                                        self.working_art.load_state(self.parent_art_state)
                                        self.generation = self.parent_generation
                                        self.topology_version = self.parent_topology_version
                                    _send_manual_ack(self.bus, "ERROR", tx_id, command_id)
                            else:
                                _send_manual_ack(self.bus, "ERROR", tx_id, command_id)
                            
                        elif st == "GLOBAL_ROLLBACK":
                            tx_id = packet.get("payload", {}).get("transaction_id", "UNKNOWN") if isinstance(packet, dict) else "UNKNOWN"
                            command_id = packet.get("command_id", packet.get("event_id", "UNKNOWN"))
                            CheckpointManager.discard_candidate(self.node_id, tx_id)
                            if getattr(self, '_current_transaction_id', None) == tx_id or getattr(self, '_last_committed_tx_id', None) == tx_id:
                                if getattr(self, 'parent_art_state', None) is not None:
                                    self.art.load_state(self.parent_art_state)
                                    self.generation = self.parent_generation
                                    self.topology_version = self.parent_topology_version
                                    self.art.topology_version = self.topology_version
                                    self.working_art.load_state(self.art.get_state())
                                else:
                                    self.working_art.load_state(self.art.get_state())
                                if hasattr(self, 'neural_encoder') and getattr(self, '_parent_encoder_sd', None) is not None:
                                    with self.training_lock:
                                        self.neural_encoder.load_state_dict(self._parent_encoder_sd)
                                        if hasattr(self, 'encoder_optimizer') and getattr(self, '_parent_encoder_opt_sd', None) is not None:
                                            self.encoder_optimizer.load_state_dict(self._parent_encoder_opt_sd)
                                    
                                self._staged_tx_memories.pop(tx_id, None)
                                self._clear_pending()
                                self._reconcile_chroma_db(rollback_tx_id=tx_id)
                                if getattr(self, '_last_committed_tx_id', None) == tx_id:
                                    self._last_committed_tx_id = None
                                self._candidate_art_state = None
                                self._current_transaction_id = None
                                self._evo_state = "LIVE"
                                is_dirty = False
                                last_save = time.time()
                                
                            _send_manual_ack(self.bus, "ROLLED_BACK", tx_id, command_id)

                    elif topic == "GLOBAL_PLASTICITY":
                        rpe = float(payload.get("rpe", 0.0))
                        novelty = float(payload.get("novelty", 0.0))
                        if self._evo_state == "LIVE":
                            if novelty > 0.7 or rpe < -0.3:
                                self.working_art.vigilance = min(0.98, self.working_art.vigilance + 0.01)
                                self.working_art.functional_revision += 1
                                is_dirty = True
                            elif rpe > 0.5:
                                self.working_art.vigilance = max(0.5, self.working_art.vigilance - 0.005)
                                self.working_art.functional_revision += 1
                                is_dirty = True

                    elif topic == "SAVE_MEM":
                        vector = self.encode_semantic(payload)
                        mem_id = f"{local_episode}_{uuid.uuid4().hex[:8]}"
                        node_id, action = self.working_art.process(vector, is_query=False)
                            
                        if self.collection is not None and node_id is not None:
                            meta = {"episode_id": local_episode, "generation": self.generation + 1, "art_node": node_id, "type": payload.get("type", "episodic")}
                            text_desc = json.dumps(payload, ensure_ascii=False)
                            self.pending_memories.append({"id": mem_id, "vector": vector, "doc": text_desc, "meta": meta})
                        is_dirty = True

                    elif topic == "THALAMUS_TOPDOWN":
                        if isinstance(packet, dict) and "topdown_attention" in packet:
                            self.current_topdown = packet.get("topdown_attention", {})
                        
                    elif topic == "THALAMUS_M8":
                        query_context = {"payload": payload, "topdown_focus": getattr(self, 'current_topdown', {})}
                        vector = self.encode_semantic(query_context)
                        is_q = payload.get("is_query", False)
                        
                        if is_q:
                            node_id, action = self.art.process(vector, is_query=True)
                        else:
                            node_id, action = self.working_art.process(vector, is_query=False)
                            
                        if action in ["CREATE", "UPDATE"] and self.collection is not None and node_id is not None:
                            mem_id = f"{local_episode}_{uuid.uuid4().hex[:12]}"
                            meta = {"episode_id": local_episode, "generation": self.generation + 1, "art_node": node_id, "type": "sensory_observation"}
                            text_desc = json.dumps(payload, ensure_ascii=False)
                            self.pending_memories.append({"id": mem_id, "vector": vector, "doc": text_desc, "meta": meta})
                            is_dirty = True
                            
                        retrieved, confidence = "Không có ký ức liên quan.", 0.0
                        if self.collection is not None:
                            try:
                                res = self.collection.query(query_embeddings=[vector], n_results=1, where={"generation": {"$lte": self.generation}})
                                if res and res.get('documents') and len(res['documents'][0]) > 0:
                                    meta = res.get('metadatas', [[{}]])[0][0]
                                    art_node_id = meta.get("art_node")
                                    if any(n.get("node_id") == art_node_id for n in self.art.memory_nodes):
                                        retrieved = res['documents'][0][0]
                                        dist = res.get('distances', [[1.0]])[0][0]
                                        confidence = max(0.0, 1.0 - dist)
                                    else:
                                        retrieved, confidence = "Không có ký ức liên quan. (Deleted Node)", 0.0
                            except Exception as e:
                                logger.error(f"🔴 ChromaDB query failed: {e}")
                                
                        if confidence < 0.5 or self.collection is None:
                            q_node_id, _ = self.art.process(vector, is_query=True)
                            if q_node_id:
                                node = next((n for n in self.art.memory_nodes if n.get("node_id") == q_node_id), None)
                                if node:
                                    retrieved = f"ART_RECALL_NODE_{q_node_id}: " + json.dumps(node["weight"][:5])
                                    confidence = max(confidence, 0.6)
                                    
                        self.bus.publish("MEM_RESULT", {"retrieved_memory": retrieved, "confidence": confidence}, parent_event_id=event_id, episode_id=local_episode)

                    elif topic == "QUERY_MEM":
                        query_context = {"payload": payload, "topdown_focus": getattr(self, 'current_topdown', {})}
                        vector = self.encode_semantic(query_context)
                        retrieved, confidence = "Không có ký ức liên quan.", 0.0
                        if self.collection is not None:
                            try:
                                res = self.collection.query(query_embeddings=[vector], n_results=1, where={"generation": {"$lte": self.generation}})
                                if res and res.get('documents') and len(res['documents'][0]) > 0:
                                    meta = res.get('metadatas', [[{}]])[0][0]
                                    art_node_id = meta.get("art_node")
                                    if any(n.get("node_id") == art_node_id for n in self.art.memory_nodes):
                                        retrieved = res['documents'][0][0]
                                        dist = res.get('distances', [[1.0]])[0][0]
                                        confidence = max(0.0, 1.0 - dist)
                                    else:
                                        retrieved, confidence = "Không có ký ức liên quan. (Deleted Node)", 0.0
                            except Exception as e:
                                logger.error(f"🔴 ChromaDB query failed: {e}")
                        
                        if confidence < 0.5 or self.collection is None:
                            node_id, _ = self.art.process(vector, is_query=True)
                            if node_id:
                                node = next((n for n in self.art.memory_nodes if n.get("node_id") == node_id), None)
                                if node:
                                    retrieved = f"ART_RECALL_NODE_{node_id}: " + json.dumps(node["weight"][:5])
                                    confidence = max(confidence, 0.6)
                                    
                        self.bus.publish("MEM_RESULT", {"retrieved_memory": retrieved, "confidence": confidence}, parent_event_id=event_id, episode_id=local_episode)

            current_time = time.time()
            
            if current_time - last_heartbeat > 5.0:
                try:
                    self.upstream_sock.send_string(f"UPSTREAM {json.dumps({'state': 'HEARTBEAT', 'module': self.node_id, 'worker_epoch': self.worker_epoch})}", flags=zmq.NOBLOCK)
                    self.bus.publish("SYSTEM_CONTROL", {"state": "HEARTBEAT", "module": self.node_id, "worker_epoch": self.worker_epoch, "modality": "control"}, episode_id=local_episode)
                except Exception as e: logger.warning(f"⚠️ Heartbeat send failed: {e}")
                last_heartbeat = current_time

            if is_dirty and current_time - last_save > 60.0 and self._evo_state == "LIVE" and self._current_transaction_id is None:
                is_dirty = False
                last_save = current_time

            if self.is_sleeping and current_time - last_dream > 2.0:
                if len(self.art.memory_nodes) > 0:
                    dream_node = random.choice(self.art.memory_nodes)
                    self.bus.publish("DREAM_REPLAY", {"vector": dream_node["weight"], "status": "DREAMING"}, episode_id=local_episode)
                last_dream = current_time

def start_node(stop_event=None):
    node = HippocampusMemory()
    try:
        node.run(stop_event)
    except KeyboardInterrupt:
        logger.info(f"🛑 {node.node_id}: Bắt được KeyboardInterrupt, đang dọn dẹp...")
    finally:
        try: node.training_queue.put_nowait(None)
        except Exception: pass
        if hasattr(node, 'upstream_sock'): 
            try: node.upstream_sock.close(linger=0)
            except Exception: pass
        if hasattr(node, 'bus'):
            try: node.bus.shutdown()
            except Exception: pass
        logger.info(f"✅ {node.node_id}: Đã tắt hoàn toàn.")

if __name__ == "__main__":
    start_node()