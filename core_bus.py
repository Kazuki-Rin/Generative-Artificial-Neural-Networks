#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
core_bus.py — Unified Message Bus (Data Plane + Control Plane) — Windows Spawn Safe
- Data Plane: PUB/SUB via Broker (Port 5555/5556) -> High throughput, lossy.
- Control Plane: ROUTER/DEALER Direct (Port 5557) -> Reliable, ACK-based, Ordered.
- **FIX**: Tạo ZMQ Context riêng trong __init__ để tương thích Windows Spawn.
- **FIX**: Sử dụng zmq.proxy_steerable để shutdown graceful, tránh crash Thread-1 (proxy).
"""

import zmq
import json
import uuid
import time
import logging
import threading
import atexit
import copy
from typing import Dict, List, Optional, Callable, Any, Tuple

logger = logging.getLogger(__name__)

# =========================================================================
# PORT DEFINITIONS
# =========================================================================
BROKER_XPUB_PORT = 5555  # Models SUB connect here
BROKER_XSUB_PORT = 5556  # Models PUB connect here
CONTROL_ROUTER_PORT = 5557  # Master BINDs ROUTER here

# =========================================================================
# HELPERS
# =========================================================================
def _now_ms() -> int:
    return int(time.time() * 1000)

def create_control_packet(
    command: str,
    payload: dict,
    command_id: str = None,
    transaction_id: str = None,
    source: str = "unknown",
    requires_ack: bool = True,
    sequence: int = 0,
    tick_id: int = 0
) -> dict:
    return {
        "schema_version": 2,
        "type": "CONTROL",
        "command": command,
        "command_id": command_id or f"cmd_{uuid.uuid4().hex[:12]}",
        "transaction_id": transaction_id,
        "source": source,
        "timestamp_ms": int(time.time() * 1000),
        "tick_id": tick_id,
        "sequence": sequence,
        "requires_ack": requires_ack,
        "payload": payload
    }

def create_ack_packet(
    command_id: str,
    transaction_id: str,
    source: str,
    status: str = "OK",
    details: dict = None
) -> dict:
    return {
        "schema_version": 2,
        "type": "CONTROL_ACK",
        "command_id": command_id,
        "transaction_id": transaction_id,
        "source": source,
        "timestamp_ms": int(time.time() * 1000),
        "status": status,
        "details": details or {}
    }

# =========================================================================
# CORE BUS CLASS
# =========================================================================
class CoreBus:
    def __init__(self, node_id: str, is_master: bool = False):
        self.node_id = node_id
        self.is_master = is_master
        self._shutdown = False

        # Khởi tạo các biến theo dõi Sequence và Causal Metadata trung tâm
        self.current_tick = 0
        self.current_episode = "UNKNOWN"
        self._sequence_id = 0

        # TẠO CONTEXT RIÊNG TRONG CHILD PROCESS
        self.context = zmq.Context()

        # --- 1. DATA PLANE (PUB/SUB via Broker) ---
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.LINGER, 0)
        self.pub_socket.connect(f"tcp://127.0.0.1:{BROKER_XSUB_PORT}")

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        self.sub_socket.connect(f"tcp://127.0.0.1:{BROKER_XPUB_PORT}")

        from collections import OrderedDict, deque
        self.CRITICAL_TOPICS = {
            "ENV_REWARD", "GLOBAL_DOPAMINE", "GLOBAL_PLASTICITY",
            "SAVE_MEM", "MOTOR_FEEDBACK", "AMYGDALA_THREAT",
            "THALAMUS_M4", "THALAMUS_M6", "THALAMUS_M7", "THALAMUS_M8", "THALAMUS_M10"
        }
        self._topic_sequences: Dict[str, int] = {}
        self._replay_cache: "OrderedDict[Tuple[str, str, int], dict]" = OrderedDict()
        self._replay_cache_maxlen = 1000
        self._replay_lock = threading.Lock()
        self._replayed_queue = deque(maxlen=256)

        # --- 2. CONTROL PLANE (ROUTER/DEALER Direct) ---
        if self.is_master:
            self.ctrl_socket = self.context.socket(zmq.ROUTER)
            self.ctrl_socket.setsockopt(zmq.LINGER, 0)
            self.ctrl_socket.bind(f"tcp://127.0.0.1:{CONTROL_ROUTER_PORT}")
            logger.info(f"[{self.node_id}] Control Plane: ROUTER bound on port {CONTROL_ROUTER_PORT}")
            self._pending_acks: Dict[str, Dict] = {}
            self._lock = threading.Lock()
        else:
            self.ctrl_socket = self.context.socket(zmq.DEALER)
            self.ctrl_socket.setsockopt(zmq.LINGER, 0)
            self.ctrl_socket.setsockopt_string(zmq.IDENTITY, node_id)
            self.ctrl_socket.connect(f"tcp://127.0.0.1:{CONTROL_ROUTER_PORT}")
            logger.info(f"[{self.node_id}] Control Plane: DEALER connected to Master port {CONTROL_ROUTER_PORT}")
            self._control_handlers: Dict[str, Callable[[dict], dict]] = {}
            self._processed_command_ids = set()
            self._idempotent_results: Dict[str, Tuple[str, dict]] = {}
            self._dedup_maxlen = 1000
            self._dedup_lock = threading.Lock()
            self._cmd_history_list = []
            self.register_control_handler("REPLAY_DATA", self._handle_replay_data)
            self.register_control_handler("FETCH_REPLAY_CACHE", self._handle_fetch_replay_cache)

        self.dealer_socket = self.ctrl_socket
        self.poller = zmq.Poller()
        self.poller.register(self.sub_socket, zmq.POLLIN)
        self.poller.register(self.ctrl_socket, zmq.POLLIN)
        atexit.register(self.shutdown)
        logger.info(f"[{self.node_id}] CoreBus initialized (Master={is_master}).")

    # =========================================================================
    # DATA PLANE API (Fire-and-Forget)
    # =========================================================================
    def publish(self, topic: str, payload: dict, modality: str = None,
                episode_id: str = None, task_id: str = "NO_TASK",
                parent_event_id: str = None, tick_id: int = None):
        
        self._sequence_id += 1
        topic_seq = self._topic_sequences.get(topic, 0) + 1
        self._topic_sequences[topic] = topic_seq
        
        actual_episode = episode_id if episode_id is not None else getattr(self, 'current_episode', "UNKNOWN")
        actual_tick = tick_id if tick_id is not None else getattr(self, 'current_tick', 0)
        if (not task_id or task_id == "NO_TASK") and isinstance(payload, dict) and payload.get("task_id"):
            task_id = str(payload["task_id"])

        if topic == "GLOBAL_PLASTICITY" and isinstance(payload, dict):
            payload["revision"] = self._sequence_id
            payload["generation"] = payload.get("generation", 0)

        packet = {
            "schema_version": 1,
            "topic": topic,
            "episode_id": actual_episode,
            "tick_id": actual_tick,
            "sequence_id": self._sequence_id,
            "topic_seq": topic_seq,
            "task_id": task_id,
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "parent_event_id": parent_event_id,
            "source": self.node_id,
            "modality": modality,
            "timestamp": time.time(),
            "payload": payload
        }
        msg = f"{topic} {json.dumps(packet, ensure_ascii=False)}"
        
        if topic == "SYSTEM_CONTROL" and self.is_master:
            control_state = payload.get("state", "")
            if control_state in ["EMERGENCY_STOP", "AWAKE_AND_SAFE", "START_TRAINING", "ASLEEP", "AWAKE", "RESUME", "UPDATE_CREDIT", "RESET"]:
                from shared_utils import MODEL_SPECS
                target_modules = list(dict.fromkeys(list(MODEL_SPECS.keys()) + ["Sandbox_Core"]))
                self.send_control_fire_forget(command=control_state, payload=payload, target_modules=target_modules)

        if topic in self.CRITICAL_TOPICS:
            self._store_in_replay_cache(self.node_id, topic, topic_seq, packet)

        if not hasattr(self, '_dropped_packets'):
            self._dropped_packets = 0
            
        if topic in self.CRITICAL_TOPICS:
            deadline = time.monotonic() + 2.0
        else:
            deadline = time.monotonic() + 0.1
            
        while True:
            try:
                self.pub_socket.send_string(msg, flags=zmq.NOBLOCK)
                break
            except zmq.Again:
                if time.monotonic() >= deadline:
                    self._dropped_packets += 1
                    if topic in self.CRITICAL_TOPICS:
                        logger.critical(f"[{self.node_id}] 💀 DROP CRITICAL PACKET '{topic}' (Buffer đầy > 2s)! Total dropped: {self._dropped_packets}")
                    else:
                        logger.warning(f"[{self.node_id}] ⚠️ DROP NEURAL PACKET '{topic}' (Sequence: {self._sequence_id})! Tín hiệu giác quan bị đứt đoạn. Total dropped: {self._dropped_packets}")
                    break
                time.sleep(0.005)
            except Exception as e:
                logger.error(f"[{self.node_id}] Publish Error ({topic}): {e}")
                break

    def subscribe(self, topics: List[str]):
        for topic in topics:
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    def receive_data(self, timeout: int = 0) -> Optional[dict]:
        if len(self._replayed_queue) > 0:
            return self._replayed_queue.popleft()
        if self.sub_socket.poll(timeout) == 0: return None
        try:
            msg = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
            return self._parse_data_msg(msg)
        except zmq.Again: return None
        except Exception as e: logger.error(f"[{self.node_id}] Data Receive Error: {e}"); return None

    def sync_metadata(self, packet: dict):
        """[BẢN VÁ P0]: Đồng bộ Metadata trung tâm cho tất cả các gói tin đến."""
        if not isinstance(packet, dict): return
        tick = packet.get("tick_id")
        if isinstance(tick, int):
            self.current_tick = max(self.current_tick, tick)
        ep = packet.get("episode_id")
        if ep and ep not in [None, "UNKNOWN"]:
            self.current_episode = ep

    def _store_in_replay_cache(self, source: str, topic: str, seq_id: int, packet: dict):
        if not source or not topic or not isinstance(seq_id, int):
            return
        with self._replay_lock:
            key = (source, topic, seq_id)
            self._replay_cache[key] = packet
            self._replay_cache.move_to_end(key)
            while len(self._replay_cache) > self._replay_cache_maxlen:
                self._replay_cache.popitem(last=False)

    def query_replay_cache(self, source: str, from_seq: int, to_seq: int, topic: str = None) -> List[dict]:
        if not source or not isinstance(from_seq, int) or not isinstance(to_seq, int) or from_seq > to_seq:
            return []
        found = []
        with self._replay_lock:
            if topic:
                for s_id in range(from_seq, min(to_seq + 1, from_seq + 100)):
                    pkt = self._replay_cache.get((source, topic, s_id))
                    if pkt is not None:
                        found.append(pkt)
            else:
                for (src_k, top_k, s_id), pkt in self._replay_cache.items():
                    if src_k == source and from_seq <= s_id <= min(to_seq, from_seq + 100):
                        found.append(pkt)
        return found

    def _handle_fetch_replay_cache(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {"status": "ERROR", "packets": []}
        pkts = self.query_replay_cache(
            source=payload.get("source", self.node_id),
            from_seq=payload.get("from_sequence", 0),
            to_seq=payload.get("to_sequence", -1),
            topic=payload.get("topic")
        )
        return {
            "status": "REPLAY_CACHE_RESULT",
            "requester": payload.get("requester"),
            "req_cmd_id": payload.get("req_cmd_id"),
            "req_tx_id": payload.get("req_tx_id"),
            "packets": pkts
        }

    def _handle_replay_data(self, payload: dict) -> dict:
        packets = payload.get("packets", []) if isinstance(payload, dict) else []
        applied = 0
        for pkt in packets:
            if isinstance(pkt, dict) and pkt.get("source"):
                top = pkt.get("topic", "UNKNOWN")
                t_seq = pkt.get("topic_seq", pkt.get("sequence_id"))
                if isinstance(t_seq, int):
                    with self._replay_lock:
                        already_seen = (pkt["source"], top, t_seq) in self._replay_cache
                    if already_seen:
                        continue
                    pkt_copy = dict(pkt)
                    pkt_copy["is_replay"] = True
                    self._store_in_replay_cache(pkt["source"], top, t_seq, pkt_copy)
                    self._replayed_queue.append(pkt_copy)
                    self.sync_metadata(pkt_copy)
                    applied += 1
        return {"status": "REPLAY_ACK", "replayed_count": applied}

    def _parse_data_msg(self, msg: str) -> Optional[dict]:
        if not msg: return None
        parts = msg.split(" ", 1)
        if len(parts) < 2: return {"topic": parts[0], "payload": {}}
        topic, json_str = parts[0], parts[1]
        try:
            packet = json.loads(json_str)
            if not isinstance(packet, dict):
                packet = {"payload": packet}
            packet["topic"] = topic
            
            self.sync_metadata(packet)
                
            source = packet.get("source")
            t_seq = packet.get("topic_seq")
            if source and topic in self.CRITICAL_TOPICS and isinstance(t_seq, int):
                self._store_in_replay_cache(source, topic, t_seq, packet)
                if not packet.get("is_replay", False):
                    if not hasattr(self, '_last_seen_seq'):
                        self._last_seen_seq = {}
                    seq_key = (source, topic)
                    expected_seq = self._last_seen_seq.get(seq_key, t_seq - 1) + 1
                    if t_seq > expected_seq and not self.is_master:
                        gap = t_seq - expected_seq
                        logger.warning(f"[{self.node_id}] 📉 CRITICAL GAP từ {source}/{topic}: Mất {gap} packets (Kỳ vọng: {expected_seq}, Nhận: {t_seq}). Gửi REPLAY_REQUEST...")
                        self.request_master_control("REPLAY_REQUEST", {
                            "requester": self.node_id,
                            "source": source,
                            "topic": topic,
                            "from_sequence": expected_seq,
                            "to_sequence": min(t_seq - 1, expected_seq + 63)
                        })
                    self._last_seen_seq[seq_key] = max(self._last_seen_seq.get(seq_key, -1), t_seq)
                
            return packet
        except json.JSONDecodeError:
            return {"topic": topic, "payload": json_str}

    # =========================================================================
    # CONTROL PLANE API
    # =========================================================================
    def register_control_handler(self, command: str, handler: Callable[[dict], dict]):
        self._control_handlers[command] = handler

    def _send_ctrl_msg(self, frames: List[bytes], timeout_sec: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                self.ctrl_socket.send_multipart(frames, flags=zmq.NOBLOCK)
                return True
            except zmq.Again:
                if time.monotonic() >= deadline:
                    logger.warning(f"[{self.node_id}] LỖI MẠNG: Control Socket Buffer đầy, Drop sau {timeout_sec}s timeout!")
                    return False
                time.sleep(0.005)
            except Exception as e:
                logger.error(f"[{self.node_id}] Control Send Error: {e}")
                return False

    def send_ack(self, command_id: str, transaction_id: str, status: str = "OK", details: dict = None):
        """Hệ thống ACK trung tâm, thay thế việc truy cập _send_ctrl_msg riêng lẻ"""
        if not hasattr(self, 'ctrl_socket'):
            logger.error(f"[{self.node_id}] Không thể gửi ACK: Chưa khởi tạo ctrl_socket")
            return
            
        actual_details = details.copy() if details else {}
        if "candidate_hash" not in actual_details:
            actual_details["candidate_hash"] = ""
        if "revision" not in actual_details:
            actual_details["revision"] = getattr(self, '_sequence_id', 0)

        ack_packet = create_ack_packet(
            command_id=command_id,
            transaction_id=transaction_id,
            source=self.node_id,
            status=status,
            details=actual_details
        )
        self._send_ctrl_msg([b'', json.dumps(ack_packet).encode('utf-8')])

    @staticmethod
    def _get_valid_ack_statuses(command: str) -> set:
        if command == "EVOLUTION_PREPARE":
            return {"PREPARED", "PREPARED_NOOP", "PREPARED_INFRA"}
        if command == "GLOBAL_COMMIT":
            return {"COMMIT_APPLIED", "COMMITTED_INFRA"}
        if command == "GLOBAL_ROLLBACK":
            return {"ROLLED_BACK", "ROLLED_BACK_INFRA"}
        return {"OK", "PREPARED", "PREPARED_NOOP", "PREPARED_INFRA", "COMMIT_APPLIED", "COMMITTED_INFRA", "ROLLED_BACK", "ROLLED_BACK_INFRA", "REPLAY_ACK"}

    def send_control_command(self,
                             command: str,
                             payload: dict,
                             expected_modules: List[str],
                             transaction_id: str = None,
                             timeout_sec: float = 10.0,
                             tick_id: int = 0) -> Tuple[bool, Dict[str, dict]]:
        if not self.is_master:
            raise RuntimeError("Only Master can call send_control_command")

        unique_modules = list(dict.fromkeys(expected_modules))
        command_id = f"cmd_{uuid.uuid4().hex[:12]}"
        if transaction_id is None:
            transaction_id = f"tx_{uuid.uuid4().hex[:12]}"

        if isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("transaction_id", transaction_id)
            payload.setdefault("command_id", command_id)

        packet = create_control_packet(
            command=command, payload=payload, command_id=command_id,
            transaction_id=transaction_id, source=self.node_id,
            requires_ack=True, tick_id=tick_id
        )
        msg_frames = [json.dumps(packet).encode('utf-8')]

        ack_event = threading.Event()
        with self._lock:
            self._pending_acks[command_id] = {
                "expected": set(unique_modules),
                "received": {},
                "event": ack_event,
                "transaction_id": transaction_id,
                "command": command
            }

        for target_id in unique_modules:
            self._send_ctrl_msg([target_id.encode('utf-8'), b'', msg_frames[0]])

        start_wait = time.time()
        success = False
        while True:
            remaining = timeout_sec - (time.time() - start_wait)
            if remaining <= 0:
                break
            
            self.process_control_acks(timeout=0)
            
            if ack_event.wait(timeout=0.01):
                success = True
                break

        with self._lock:
            result = self._pending_acks.pop(command_id, {"received": {}})
            received = result.get("received", {})

        success_statuses = self._get_valid_ack_statuses(command)

        all_ok = (
            success 
            and len(received) == len(unique_modules)
            and all(ack.get("status") in success_statuses for ack in received.values())
        )

        if not all_ok:
            missing = set(unique_modules) - set(received.keys())
            error_nodes = {node: ack.get("status") for node, ack in received.items() if ack.get("status") not in success_statuses}
            if missing:
                logger.warning(f"[{self.node_id}] Control Command Timeout. Missing: {missing}")
            if error_nodes:
                logger.error(f"[{self.node_id}] Control Command Failed. Nodes returned ERROR: {error_nodes}")

        return all_ok, received

    def send_control_fire_forget(self, command: str, payload: dict, target_modules: List[str], transaction_id: str = None):
        if not self.is_master: return
        command_id = f"cmd_{uuid.uuid4().hex[:12]}"
        if transaction_id is None: transaction_id = f"tx_{uuid.uuid4().hex[:12]}"
        packet = create_control_packet(command, payload, command_id, transaction_id, self.node_id, requires_ack=False)
        msg_data = json.dumps(packet).encode('utf-8')
        for target_id in list(dict.fromkeys(target_modules)):
            self._send_ctrl_msg([target_id.encode('utf-8'), b'', msg_data])
            
    def request_master_control(self, command: str, payload: dict):
        """Worker phát tín hiệu điều khiển trực tiếp lên Master thông qua Control Plane (DEALER socket)"""
        if self.is_master: return
        
        command_id = f"cmd_{uuid.uuid4().hex[:12]}"
        transaction_id = f"tx_{uuid.uuid4().hex[:12]}"
        
        packet = create_control_packet(
            command=command, 
            payload=payload, 
            command_id=command_id, 
            transaction_id=transaction_id, 
            source=self.node_id, 
            requires_ack=True, 
            tick_id=self.current_tick
        )
        msg_data = json.dumps(packet).encode('utf-8')
        
        self._send_ctrl_msg([b'', msg_data])

    def _get_dedup_keys(self, packet: dict) -> List[str]:
        cmd = packet.get("command", "")
        cmd_id = packet.get("command_id", "")
        tx_id = packet.get("transaction_id", "")
        source = packet.get("source", "unknown")
        keys = [f"cmd:{cmd_id}:{tx_id}:{source}"]
        if cmd in ("EVOLUTION_PREPARE", "GLOBAL_COMMIT", "GLOBAL_ROLLBACK") and tx_id:
            keys.append(f"tx:{cmd}:{tx_id}")
        return keys

    def check_duplicate_command(self, packet: dict) -> bool:
        """[BẢN VÁ B]: Kiểm tra lệnh đã có kết quả thực thi Idempotent được lưu hay chưa"""
        keys = self._get_dedup_keys(packet)
        with self._dedup_lock:
            return any(k in self._processed_command_ids for k in keys)

    def _record_idempotent_result(self, packet: dict, status: str, details: dict):
        if status in ("ERROR", "NO_HANDLER"):
            return
        keys = self._get_dedup_keys(packet)
        with self._dedup_lock:
            for k in keys:
                self._processed_command_ids.add(k)
                self._cmd_history_list.append(k)
                self._idempotent_results[k] = (status, copy.deepcopy(details) if isinstance(details, dict) else {})
            while len(self._cmd_history_list) > getattr(self, '_dedup_maxlen', 1000):
                oldest_cmd = self._cmd_history_list.pop(0)
                self._processed_command_ids.discard(oldest_cmd)
                self._idempotent_results.pop(oldest_cmd, None)

    def _get_idempotent_result(self, packet: dict) -> Optional[Tuple[str, dict]]:
        keys = self._get_dedup_keys(packet)
        with self._dedup_lock:
            for k in keys:
                if k in self._idempotent_results:
                    st, det = self._idempotent_results[k]
                    return st, dict(det)
        return None

    def process_control_commands(self, timeout: int = 0) -> int:
        if self.is_master: return 0
        count = 0
        while self.ctrl_socket.poll(timeout) != 0:
            try:
                frames = self.ctrl_socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) < 2: continue
                msg_bytes = frames[-1]
                packet = json.loads(msg_bytes.decode('utf-8'))
                if packet.get("type") != "CONTROL": continue

                cmd_id = packet.get("command_id")
                tx_id = packet.get("transaction_id")
                cmd = packet.get("command")
                payload = packet.get("payload", {})
                requires_ack = packet.get("requires_ack", True)

                if cmd in ("EVOLUTION_PREPARE", "GLOBAL_COMMIT", "GLOBAL_ROLLBACK"):
                    if not cmd_id or not tx_id:
                        logger.error(f"[{self.node_id}] Từ chối lệnh {cmd} thiếu command_id ({cmd_id}) hoặc transaction_id ({tx_id})")
                        if requires_ack:
                            self.send_ack(cmd_id or "MISSING_CMD_ID", tx_id or "MISSING_TX_ID", "ERROR", {"error": "Missing command_id or transaction_id"})
                        continue

                self.sync_metadata(packet)

                if self.check_duplicate_command(packet):
                    cached = self._get_idempotent_result(packet)
                    if requires_ack:
                        if cached is not None:
                            c_status, c_details = cached
                            c_details["idempotent_replay"] = True
                            self.send_ack(cmd_id, tx_id, c_status, c_details)
                        else:
                            self.send_ack(cmd_id, tx_id, "ERROR", {"error": "Duplicate command without recorded state"})
                    continue

                if isinstance(payload, dict):
                    payload = dict(payload)
                    payload.setdefault("transaction_id", tx_id)
                    payload.setdefault("command_id", cmd_id)

                handler = self._control_handlers.get(cmd)
                status = "OK"
                details = {}
                try:
                    if handler:
                        result = handler(payload)
                        if isinstance(result, dict): 
                            details = result.copy()
                            if "status" in details:
                                status = details.pop("status")
                        elif isinstance(result, str):
                            status = result
                    else:
                        status = "NO_HANDLER"
                except Exception as e:
                    logger.error(f"[{self.node_id}] Handler Error for {cmd}: {e}")
                    status = "ERROR"
                    details = {"error": str(e)}

                self._record_idempotent_result(packet, status, details)
                if requires_ack:
                    self.send_ack(cmd_id, tx_id, status, details)
                count += 1
            except zmq.Again: break
            except Exception as e:
                logger.error(f"[{self.node_id}] Control Process Error: {e}")
        return count

    def _handle_master_replay_request(self, requester: str, packet: dict):
        cmd_id = packet.get("command_id", f"cmd_{uuid.uuid4().hex[:12]}")
        tx_id = packet.get("transaction_id", f"tx_{uuid.uuid4().hex[:12]}")
        payload = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
        src = payload.get("source")
        top = payload.get("topic")
        from_seq = payload.get("from_sequence")
        to_seq = payload.get("to_sequence")

        if not requester or not src or not isinstance(from_seq, int) or not isinstance(to_seq, int) or from_seq <= 0 or from_seq > to_seq or (to_seq - from_seq) > 100:
            ack = create_ack_packet(cmd_id, tx_id, self.node_id, status="ERROR", details={"error": "Invalid REPLAY_REQUEST parameters"})
            self._send_ctrl_msg([requester.encode('utf-8'), b'', json.dumps(ack).encode('utf-8')])
            return

        cached_pkts = self.query_replay_cache(src, from_seq, to_seq, topic=top)
        if not cached_pkts and src != self.node_id:
            fetch_cmd = create_control_packet(
                command="FETCH_REPLAY_CACHE",
                payload={"requester": requester, "source": src, "topic": top, "from_sequence": from_seq, "to_sequence": to_seq, "req_cmd_id": cmd_id, "req_tx_id": tx_id},
                command_id=f"fetch_{cmd_id}",
                transaction_id=tx_id,
                source=self.node_id,
                requires_ack=True,
                tick_id=self.current_tick
            )
            self._send_ctrl_msg([src.encode('utf-8'), b'', json.dumps(fetch_cmd).encode('utf-8')])
            return

        if cached_pkts:
            replay_cmd = create_control_packet(
                command="REPLAY_DATA",
                payload={"source": src, "topic": top, "from_sequence": from_seq, "to_sequence": to_seq, "packets": cached_pkts},
                command_id=f"rep_{cmd_id}",
                transaction_id=tx_id,
                source=self.node_id,
                requires_ack=False,
                tick_id=self.current_tick
            )
            self._send_ctrl_msg([requester.encode('utf-8'), b'', json.dumps(replay_cmd).encode('utf-8')])
        ack = create_ack_packet(cmd_id, tx_id, self.node_id, status="REPLAY_ACK", details={"replayed_count": len(cached_pkts)})
        self._send_ctrl_msg([requester.encode('utf-8'), b'', json.dumps(ack).encode('utf-8')])

    def process_control_acks(self, timeout: int = 0) -> int:
        if not self.is_master: return 0
        count = 0
        while self.ctrl_socket.poll(timeout) != 0:
            try:
                frames = self.ctrl_socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) < 2: continue
                identity = frames[0].decode('utf-8')
                msg_bytes = frames[-1]
                packet = json.loads(msg_bytes.decode('utf-8'))
                
                if packet.get("type") == "CONTROL" and packet.get("command") == "REPLAY_REQUEST":
                    self._handle_master_replay_request(identity, packet)
                    count += 1
                    continue

                if packet.get("type") != "CONTROL_ACK": continue
                
                cmd_id = packet.get("command_id")
                tx_id = packet.get("transaction_id")
                source = packet.get("source")
                status = packet.get("status")

                if status == "REPLAY_CACHE_RESULT":
                    det = packet.get("details", {}) if isinstance(packet.get("details"), dict) else {}
                    req_node = det.get("requester")
                    pkts = det.get("packets", [])
                    r_cmd_id = det.get("req_cmd_id", cmd_id)
                    r_tx_id = det.get("req_tx_id", tx_id)
                    if req_node:
                        if pkts:
                            for p_item in pkts:
                                if isinstance(p_item, dict) and p_item.get("source") and isinstance(p_item.get("topic_seq"), int):
                                    self._store_in_replay_cache(p_item["source"], p_item.get("topic", "UNKNOWN"), p_item["topic_seq"], p_item)
                            rep_pkt = create_control_packet(
                                command="REPLAY_DATA",
                                payload={"source": source, "packets": pkts},
                                command_id=f"rep_{r_cmd_id}",
                                transaction_id=r_tx_id,
                                source=self.node_id,
                                requires_ack=False,
                                tick_id=self.current_tick
                            )
                            self._send_ctrl_msg([req_node.encode('utf-8'), b'', json.dumps(rep_pkt).encode('utf-8')])
                        ack_pkt = create_ack_packet(r_cmd_id, r_tx_id, self.node_id, status="REPLAY_ACK", details={"replayed_count": len(pkts)})
                        self._send_ctrl_msg([req_node.encode('utf-8'), b'', json.dumps(ack_pkt).encode('utf-8')])
                    count += 1
                    continue

                if status == "DUPLICATE":
                    actual_st = packet.get("details", {}).get("actual_status")
                    if actual_st:
                        packet["status"] = actual_st
                    else:
                        logger.warning(f"[{self.node_id}] Bỏ qua DUPLICATE ACK không rõ trạng thái thực từ {identity} (tx={tx_id})")
                        continue

                with self._lock:
                    pending = self._pending_acks.get(cmd_id)
                    if pending is None and tx_id is not None:
                        for p_entry in self._pending_acks.values():
                            if (
                                p_entry.get("transaction_id") == tx_id
                                and identity in p_entry.get("expected", set())
                                and status in self._get_valid_ack_statuses(p_entry.get("command", ""))
                            ):
                                pending = p_entry
                                break
                    if pending:
                        valid_for_cmd = self._get_valid_ack_statuses(pending.get("command", ""))
                        if (
                            identity in pending["expected"]
                            and tx_id == pending["transaction_id"]
                            and source == identity
                            and (cmd_id in self._pending_acks or status in valid_for_cmd)
                        ):
                            pending["received"][identity] = packet
                            if len(pending["received"]) >= len(pending["expected"]):
                                pending["event"].set()
                count += 1
            except zmq.Again: break
            except Exception as e:
                logger.error(f"[{self.node_id}] Ack Process Error: {e}")
        return count

    def poll(self, timeout_ms: int = 100) -> bool:
        if len(self._replayed_queue) > 0:
            return True
        return self.sub_socket.poll(timeout_ms) != 0

    def receive(self, timeout_ms: int = 0) -> Optional[str]:
        if len(self._replayed_queue) > 0:
            pkt = self._replayed_queue.popleft()
            return f"{pkt.get('topic', 'UNKNOWN')} {json.dumps(pkt, ensure_ascii=False)}"
        if self.sub_socket.poll(timeout_ms) == 0: return None
        try:
            raw_msg = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
            self._parse_data_msg(raw_msg)
            return raw_msg
        except zmq.Again: return None
        except Exception as e:
            logger.error(f"[{self.node_id}] CoreBus.receive error: {e}")
            return None

    def receive_all(self, timeout_ms: int = 0) -> List[str]:
        msgs = []
        while True:
            m = self.receive(timeout_ms=0)
            if m is None: break
            msgs.append(m)
        return msgs

    def shutdown(self):
        if self._shutdown: return
        self._shutdown = True
        try:
            self.poller.unregister(self.sub_socket)
            self.poller.unregister(self.ctrl_socket)
        except: pass
        try: self.pub_socket.close(linger=0)
        except: pass
        try: self.sub_socket.close(linger=0)
        except: pass
        try: self.ctrl_socket.close(linger=0)
        except: pass
        try: self.context.term()
        except: pass
        logger.info(f"[{self.node_id}] CoreBus Shutdown.")

# =========================================================================
# CENTRAL BROKER (Own Context) — FIX: Steerable Proxy + Graceful Shutdown
# =========================================================================
def run_central_broker(stop_event):
    context = zmq.Context()
    ctrl_main = context.socket(zmq.PAIR)
    ctrl_main.bind("inproc://broker_ctrl")
    proxy_stopped = threading.Event()
    
    def proxy_runner():
        try:
            xpub = context.socket(zmq.XPUB)
            xsub = context.socket(zmq.XSUB)
            ctrl_proxy = context.socket(zmq.PAIR)
            
            xpub.setsockopt(zmq.LINGER, 0)
            xsub.setsockopt(zmq.LINGER, 0)
            ctrl_proxy.setsockopt(zmq.LINGER, 0)
            
            xpub.bind(f"tcp://127.0.0.1:{BROKER_XPUB_PORT}")
            xsub.bind(f"tcp://127.0.0.1:{BROKER_XSUB_PORT}")
            ctrl_proxy.connect("inproc://broker_ctrl")

            logging.info(f"[ZMQ_Broker] Data Plane Proxy Started: XPUB:{BROKER_XPUB_PORT} XSUB:{BROKER_XSUB_PORT}")
            zmq.proxy_steerable(xpub, xsub, ctrl_proxy)
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                logging.info("[ZMQ_Broker] Proxy stopped: Context terminated.")
            else:
                logging.error(f"[ZMQ_Broker] Proxy ZMQError: {e}")
        except Exception as e:
            logging.error(f"[ZMQ_Broker] Proxy thread error: {e}")
        finally:
            try: xpub.close(linger=0)
            except: pass
            try: xsub.close(linger=0)
            except: pass
            try: ctrl_proxy.close(linger=0)
            except: pass
            proxy_stopped.set()
            logging.info("[ZMQ_Broker] Proxy thread exited.")
    
    proxy_thread = threading.Thread(target=proxy_runner, daemon=True, name="ZMQ_Proxy")
    proxy_thread.start()
    
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("[ZMQ_Broker] Shutdown signal received, stopping proxy...")
        try: ctrl_main.send(b"TERMINATE", flags=zmq.NOBLOCK)
        except: pass
        proxy_stopped.wait(timeout=5.0)
        try: ctrl_main.close(linger=0)
        except: pass
        try: context.term()
        except: pass
        logging.info("[ZMQ_Broker] Shutdown complete.")