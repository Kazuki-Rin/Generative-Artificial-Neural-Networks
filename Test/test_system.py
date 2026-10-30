#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import time
import subprocess
import threading
import unittest
import zmq
import json
import random
import uuid
from collections import defaultdict, deque
from loguru import logger

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

REQUIRED_MODELS = [
    "M0_Cerebrospinal", "M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial",
    "M4_Motor", "M5_Amygdala", "M6_Language", "M7_Logic", "M8_Memory",
    "M9_Thalamus", "M10_Vocal"
]
ALL_REQUIRED = REQUIRED_MODELS

HANDSHAKE_TIMEOUT = 600.0
TEST_DURATION = 130.0  

class SystemIntegrationHarness:
    def __init__(self):
        self.master_proc = None
        self.stop_event = threading.Event()
        
        self.ready_modules = set()
        self.event_graph = {}
        self.system_started = False
        self.evolution_success_count = 0 
        
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        self.sub_socket.connect("tcp://127.0.0.1:5555") 
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.sniff_thread = None
        self.injector_thread = None

    def start_master(self):
        main_script = os.path.join(BASE_DIR, "main_life_loop.py")
        self.master_proc = subprocess.Popen(
            [sys.executable, "-u", main_script],
            cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        threading.Thread(target=self._log_master, daemon=True).start()
        self.sniff_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.sniff_thread.start()
        
        self.injector_thread = threading.Thread(target=self._stimulus_injector_loop, daemon=True)
        self.injector_thread.start()

    def _stimulus_injector_loop(self):
        import math, uuid, random, json
        logger.info("💉 [INJECTOR] Bật hệ thống giả lập Môi trường và Bơm Kích thích...")
        try:
            pub = self.context.socket(zmq.PUB)
            pub.setsockopt(zmq.LINGER, 0)
            pub.connect("tcp://127.0.0.1:5556")
            seq = 0
            
            time.sleep(15.0) 
            
            while not self.stop_event.is_set():
                seq += 1
                def inject(topic, modality, payload, source="Sandbox_Core", override_parent=None):
                    evt_id = f"evt_{uuid.uuid4().hex[:12]}"
                    pkt = {
                        "schema_version": 1, "episode_id": "TEST_EP", "tick_id": seq,
                        "sequence_id": seq,
                        "task_id": "TEST", "event_id": evt_id,
                        "parent_event_id": override_parent,
                        "source": source, "modality": modality,
                        "timestamp": time.time(), "payload": payload
                    }
                    pub.send_string(f"{topic} {json.dumps(pkt)}")
                    return evt_id
                
                if seq % 10 == 0:
                    inject("SYSTEM_CONTROL", "control", {"state": "AWAKE_AND_SAFE"})
                else:
                    inject("SYSTEM_CONTROL", "control", {"state": "START_TRAINING"})
                inject("SYSTEM_CONTROL", "control", {"state": "RESUME"})
                
                vision_pattern = [math.sin(i * 0.1) * 0.5 + 0.5 for i in range(1600)]
                audio_pattern = [math.cos(i * 0.05) * 0.5 + 0.5 for i in range(1600)]
                
                inject("ENV_VISION", "vision", {"data": vision_pattern, "target_labels": [0.5, 0.5], "type": "array"})
                
                inject("ENV_AUDIO", "audio", {"data": audio_pattern, "label": "Vowel"})
                inject("ENV_SPATIAL", "spatial", {"x": 0.5, "y": 0.5})
                
                inject("ENV_REWARD", "reward", {"reward": 0.8, "rpe": 0.3, "task_id": "TEST", "tick_id": seq, "terminated": False, "truncated": False})
                inject("ENV_HAZARD", "threat", {"is_danger": 1 if seq % 3 == 0 else 0})

                time.sleep(1.5)
        except Exception as e:
            logger.error(f"INJECTOR CRASHED: {e}")

    def _sniff_loop(self):
        poller = zmq.Poller()
        poller.register(self.sub_socket, zmq.POLLIN)
        while not self.stop_event.is_set():
            try:
                socks = dict(poller.poll(100))
                if self.sub_socket in socks:
                    while True:
                        try:
                            parts = self.sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                            if not parts: break
                            
                            if len(parts) == 1:
                                msg_str = parts[0].decode('utf-8', errors='ignore')
                                if " " in msg_str:
                                    topic, payload_str = msg_str.split(" ", 1)
                                else:
                                    topic, payload_str = msg_str, "{}"
                            else:
                                topic = parts[0].decode('ascii', errors='ignore')
                                payload_str = parts[1].decode('utf-8', errors='ignore')
                                
                            topic = topic.strip()
                            try:
                                payload = json.loads(payload_str)
                                evt_id = payload.get("event_id")
                                if evt_id:
                                    self.event_graph[evt_id] = {
                                        "topic": topic,
                                        "parent": payload.get("parent_event_id"),
                                        "episode": payload.get("episode_id"),
                                        "source": payload.get("source"),
                                        "timestamp": payload.get("timestamp", 0.0)
                                    }
                            except json.JSONDecodeError: pass
                        except zmq.Again: break 
            except Exception: pass

    def _log_master(self):
        try:
            for line in self.master_proc.stdout:
                line_str = line.strip()
                if line_str: logger.info(f"[MASTER] {line_str}")
                if any(k in line_str for k in ["READY!", "READY (PubSub)", "READY (epoch="]):
                    for mod in ALL_REQUIRED:
                        if mod in line_str: self.ready_modules.add(mod)
                if "START_TRAINING" in line_str:
                    self.system_started = True
                if any(k in line_str for k in ["EVOLUTION", "PREPARE", "COMMIT", "Ghi Manifest", "ROLLBACK"]):
                    line_lower = line_str.lower()
                    if "thành công" in line_lower or "commit_applied" in line_lower or "hoàn tất" in line_lower or "rollback" in line_lower:
                        self.evolution_success_count += 1
        except ValueError: pass 

    def wait_handshake(self):
        logger.info(f"⏳ Đang chờ 13/13 Thùy não báo cáo READY (Timeout: {HANDSHAKE_TIMEOUT}s)...")
        start = time.time()
        while time.time() - start < HANDSHAKE_TIMEOUT:
            if self.master_proc.poll() is not None:
                raise RuntimeError("Master Process DIED during handshake")
            if self.system_started and len(self.ready_modules) >= len(ALL_REQUIRED):
                return True
            time.sleep(0.5)
        raise RuntimeError(f"Handshake Timeout. Chỉ nhận được {len(self.ready_modules)}/{len(ALL_REQUIRED)}")

    def run_scenario(self):
        logger.info(f"🧪 Đang dùng Sniffer hút mạch Causal Chain trong {TEST_DURATION}s...")
        start = time.time()
        while time.time() - start < TEST_DURATION:
            if self.master_proc.poll() is not None: break
            time.sleep(1.0)
            
        logger.info("✅ Hoàn tất nghe lén. Đang đối chiếu mạch máu...")

    def shutdown(self):
        self.stop_event.set()
        for t_attr in ("sniffer_thread", "injector_thread"):
            t_obj = getattr(self, t_attr, None)
            if t_obj is not None and t_obj.is_alive():
                t_obj.join(timeout=2.0)
        if self.master_proc:
            logger.info("🛑 Đang tiêu diệt toàn bộ Cây Tiến Trình (Process Tree) của hệ thống Test...")
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.master_proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.master_proc.terminate()
            try: self.master_proc.wait(timeout=5)
            except Exception: self.master_proc.kill()
            try:
                if self.master_proc.stdout:
                    self.master_proc.stdout.close()
            except Exception: pass
        if hasattr(self, "log_thread") and self.log_thread.is_alive():
            self.log_thread.join(timeout=2.0)

class TestSystemIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")
        cls.harness = SystemIntegrationHarness()
        cls.harness.start_master()
        cls.harness.wait_handshake()

    @classmethod
    def tearDownClass(cls):
        cls.harness.shutdown()

    def test_01_all_modules_ready(self):
        missing = set(ALL_REQUIRED) - self.harness.ready_modules
        self.assertEqual(len(missing), 0, f"Các thùy não bị liệt (Chưa Ready): {missing}")

    def test_02_runtime_liveness_all_models(self):
        self.harness.run_scenario() 
        self.assertIsNone(self.harness.master_proc.poll(), "Master Process đã sập trong lúc chạy!")

    def _get_causal_paths(self, start_eid, target_source="Sandbox_Core", max_depth=15):
        """Truy vết ngược và trả về toàn bộ lộ trình Topic từ Ngọn về Rễ (Có giới hạn độ sâu chống bão tín hiệu)"""
        paths = []
        queue = deque([([start_eid], [self.harness.event_graph[start_eid]["topic"]])])
        
        while queue:
            path_eids, path_topics = queue.popleft()
            
            if len(path_eids) > max_depth:
                continue
                
            current_eid = path_eids[-1]
            node = self.harness.event_graph[current_eid]
            
            if node.get("source") == target_source:
                paths.append(path_topics)
                continue
                
            parents = str(node.get("parent") or "").split(",")
            for p in parents:
                p = p.strip()
                if p and p != "None" and p in self.harness.event_graph and p not in path_eids:
                    p_topic = self.harness.event_graph[p]["topic"]
                    queue.append((path_eids + [p], path_topics + [p_topic]))
        return paths

    def test_03_causal_chain_vision_to_motor(self):
        motor_exec_events = {eid: data for eid, data in self.harness.event_graph.items() if data["topic"] == "MOTOR_EXECUTE"}
        valid_chains = 0
        logger.info(f"DEBUG: Found {len(motor_exec_events)} MOTOR_EXECUTE events")
        for m_eid in motor_exec_events:
            paths = self._get_causal_paths(m_eid, target_source="Sandbox_Core")
            if not paths:
                curr = m_eid; trace = []
                while curr:
                    node = self.harness.event_graph.get(curr)
                    if not node: break
                    trace.append(node["topic"])
                    p = str(node.get("parent") or "").split(",")
                    curr = p[0].strip() if p and p[0] != "None" else None
                logger.info(f"DEBUG M_EID {m_eid}: {trace}")
            for path in paths:
                logger.info(f"MOTOR PATH: {path}")
                # Path là mảng đi từ Ngọn về Rễ, ví dụ: ['MOTOR_EXECUTE', 'CMD_MOTOR', 'THALAMUS_M7', 'RAW_VISION_M1A', 'ENV_VISION']
                if any(t in ["RAW_VISION_M1A", "RAW_VISION_M1B"] for t in path) and ("CMD_MOTOR" in path or "THALAMUS_M4" in path):
                    valid_chains += 1
                    break
                    
        self.assertGreater(valid_chains, 0, "LỖI CHUỖI NHÂN QUẢ: Sandbox -> Vision -> Thalamus -> Motor bị đứt gãy hoặc đi sai đường.")
        logger.info(f"🟢 Causal Chain: Khám phá thành công {valid_chains} luồng Vision -> Motor nguyên vẹn.")

    def test_04_causal_chain_language_to_vocal(self):
        vocal_fb_events = {eid: data for eid, data in self.harness.event_graph.items() if data["topic"] == "VOCAL_FEEDBACK"}
        valid_chains = 0
        logger.info(f"DEBUG: Found {len(vocal_fb_events)} VOCAL_FEEDBACK events")
        for v_eid in vocal_fb_events:
            paths = self._get_causal_paths(v_eid, target_source="Sandbox_Core")
            if not paths:
                curr = v_eid; trace = []
                while curr:
                    node = self.harness.event_graph.get(curr)
                    if not node: break
                    trace.append(node["topic"])
                    p = str(node.get("parent") or "").split(",")
                    curr = p[0].strip() if p and p[0] != "None" else None
                logger.info(f"DEBUG V_EID {v_eid}: {trace}")
            for path in paths:
                # Path ví dụ: ['VOCAL_FEEDBACK', 'VOCAL_INTENT', 'SEMANTICS', 'THALAMUS_M6', 'RAW_AUDIO', 'ENV_AUDIO']
                if "RAW_AUDIO" in path and any(t in ["THALAMUS_M10", "VOCAL_INTENT"] for t in path):
                    valid_chains += 1
                    break
                    
        self.assertGreater(valid_chains, 0, "LỖI CHUỖI NHÂN QUẢ: Sandbox -> Audio -> Thalamus -> Vocal bị đứt gãy hoặc đi sai đường.")
        logger.info(f"🟢 Causal Chain: Khám phá thành công {valid_chains} luồng Audio -> Vocal nguyên vẹn.")

    def test_05_neuromodulation_flow(self):
        topics = {data["topic"] for data in self.harness.event_graph.values()}
        self.assertIn("CEREBROSPINAL_FLUID", topics, "LỖI ĐIỀU HÒA: M0 không bơm CEREBROSPINAL_FLUID.")
        self.assertIn("GLOBAL_PLASTICITY", topics, "LỖI ĐIỀU HÒA: M0 không phát GLOBAL_PLASTICITY.")
        logger.info("🟢 Neuromodulation: M0 Broadcast Flow [CHÍNH XÁC]")

    def test_06_evolution_cycle_completed(self):
        self.assertTrue(
            self.harness.evolution_success_count > 0, 
            "LỖI 2PC: Không ghi nhận được toàn trình giao dịch PREPARE -> COMMIT/ROLLBACK trong Log của Master."
        )
        logger.info("🟢 Evolution: Chu kỳ 2PC nguyên tử đã hoàn tất [CHÍNH XÁC]")

if __name__ == '__main__':
    unittest.main(verbosity=2)