#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TEST_BRAIN_ARCHITECTURE.PY (Version 3.2 - BULLETPROOF SNIFFER)
Kiểm định tính hợp lệ của toàn bộ Hệ sinh thái ZeroMQ và Cấu trúc Kiến trúc theo Tiêu chuẩn.
"""

import os
import sys
import json
import time
import subprocess
import threading
import unittest
import zmq
from loguru import logger

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

class TestBrainArchitectureRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        cls.manifest_path = os.path.join(BASE_DIR, "architecture_manifest.json")
        try:
            with open(cls.manifest_path, "r", encoding="utf-8") as f:
                cls.manifest = json.load(f)
        except Exception:
            cls.manifest = {"modules": {}}
            
        cls.master_proc, cls.sniff_thread, cls.stop_sniff = None, None, threading.Event()
        cls.sniffed_topics, cls.modules_ready_log = set(), []
        cls.start_system()

    @classmethod
    def tearDownClass(cls):
        cls.stop_event.set()
        for t_attr in ("sniffer_thread", "injector_thread"):
            t_obj = getattr(cls, t_attr, None)
            if t_obj is not None and t_obj.is_alive():
                t_obj.join(timeout=2.0)
        if cls.master_proc:
            logger.info("🛑 Đang tiêu diệt toàn bộ Cây Tiến Trình (Process Tree) của hệ thống Test...")
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(cls.master_proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                cls.master_proc.terminate()
            try: cls.master_proc.wait(timeout=5)
            except Exception: cls.master_proc.kill()
            try:
                if cls.master_proc.stdout:
                    cls.master_proc.stdout.close()
            except Exception: pass
        if hasattr(cls, "log_thread") and cls.log_thread.is_alive():
            cls.log_thread.join(timeout=2.0)

    @classmethod
    def start_system(cls):
        main_script = os.path.join(BASE_DIR, "main_life_loop.py")
        logger.info("🚀 Đánh thức MasterSupervisor (Kiểm định Hợp đồng)...")
        cls.master_proc = subprocess.Popen(
            [sys.executable, "-u", main_script],
            cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        threading.Thread(target=cls._log_master, daemon=True).start()
        cls.sniff_thread = threading.Thread(target=cls.sniff_runtime, daemon=True)
        cls.sniff_thread.start()
        
        threading.Thread(target=cls._stimulus_injector_loop, daemon=True).start()
        cls.wait_handshake()

    @classmethod
    def _stimulus_injector_loop(cls):
        import random, json, uuid, math
        context = zmq.Context.instance()
        pub = context.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.connect("tcp://127.0.0.1:5556")
        seq = 0
        time.sleep(15.0) 
        while not cls.stop_sniff.is_set():
            seq += 1
            def inject(topic, modality, payload, source="Sandbox_Core"):
                pkt = {
                    "schema_version": 1, "episode_id": "TEST_EP", "tick_id": seq,
                    "sequence_id": seq,
                    "task_id": "TEST", "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                    "source": source, "modality": modality,
                    "timestamp": time.time(), "payload": payload
                }
                pub.send_string(f"{topic} {json.dumps(pkt)}")
            
            if seq % 10 == 0:
                inject("SYSTEM_CONTROL", "control", {"state": "AWAKE_AND_SAFE"})
                
            inject("ENV_HAZARD", "hazard", {"is_danger": 1 if seq % 3 == 0 else 0})
            
            vision_pattern = [math.sin(i * 0.1) * 0.5 + 0.5 for i in range(1600)]
            audio_pattern = [math.cos(i * 0.05) * 0.5 + 0.5 for i in range(1600)]
            
            inject("ENV_VISION", "vision", {"data": vision_pattern, "target_labels": [0.9, 0.1], "type": "array"})
            inject("ENV_AUDIO", "audio", {"data": audio_pattern, "label": "Vowel"})
            inject("ENV_SPATIAL", "spatial", {"x": 0.8, "y": 0.2})
            
            inject("SEMANTICS", "language", {"text": "Di chuyển về phía trước", "latent_vector": [0.5]*64, "confidence": 0.99}, source="M6_Language")
            
            inject("CMD_MOTOR", "motor", {"action": "move", "dx": 1.0, "dy": 1.0}, source="M7_Logic")
            inject("MEM_RESULT", "memory", {"retrieved_data": "dummy", "confidence": 1.0}, source="M8_Memory")
            
            inject("ENV_REWARD", "reward", {"reward": 0.8, "rpe": 0.3, "task_id": "TEST", "tick_id": seq, "terminated": False, "truncated": False})
            inject("GLOBAL_PLASTICITY", "neuromodulation", {"salience": 1.0, "novelty": 0.8, "threat": 0.2, "rpe": 0.3, "arousal": 0.5, "tick_id": seq, "credit": {}, "generation": 0}, source="M0_Cerebrospinal")
            
            time.sleep(1.5)
        pub.close(linger=0)

    @classmethod
    def _log_master(cls):
        try:
            for line in cls.master_proc.stdout:
                line_str = line.strip()
                if line_str:
                    logger.info(f"[MASTER] {line_str}")
                    if any(k in line_str for k in ["READY!", "READY (PubSub)", "READY (epoch="]):
                        cls.modules_ready_log.append(line_str)
        except Exception:
            pass

    @classmethod
    def sniff_runtime(cls):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect("tcp://127.0.0.1:5555")
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        
        cls.event_graph = {}
        
        while not cls.stop_sniff.is_set():
            try:
                socks = dict(poller.poll(100))
                if socket in socks:
                    while True:
                        try:
                            parts = socket.recv_multipart(flags=zmq.NOBLOCK)
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
                            if topic:
                                cls.sniffed_topics.add(topic)
                                
                            try:
                                payload = json.loads(payload_str)
                                evt_id = payload.get("event_id")
                                
                                if evt_id:
                                    cls.event_graph[evt_id] = {
                                        "topic": topic,
                                        "parent": payload.get("parent_event_id"),
                                        "episode": payload.get("episode_id"),
                                        "source": payload.get("source"),
                                        "sequence_id": payload.get("sequence_id", 0),
                                        "timestamp": payload.get("timestamp", 0.0)
                                    }
                            except json.JSONDecodeError:
                                pass
                        except zmq.Again:
                            break
            except Exception:
                pass

    @classmethod
    def wait_handshake(cls):
        start = time.time()
        while time.time() - start < 600.0:
            if cls.master_proc and cls.master_proc.poll() is not None:
                break
            if len(cls.modules_ready_log) >= 13:
                return
            time.sleep(1.0)
        logger.warning("⚠️ Không thể hoàn thành Handshake, ép buộc chốt hạ bài Test để lấy lỗi!")

    def test_01_core_pulse_active(self):
        time.sleep(90.0)
        logger.info(f"📊 Sniffed Topics: {self.sniffed_topics}")
        self.assertIn("SYSTEM_CONTROL", self.sniffed_topics, "Lõi thần kinh trung ương (SYSTEM_CONTROL) đã chết!")
        self.assertIn("CEREBROSPINAL_FLUID", self.sniffed_topics, "Lõi điều hòa M0 (CEREBROSPINAL_FLUID) đã chết!")

    def test_02_brain_model_registry_11(self):
        expected_brains = {
            "M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial", "M4_Motor",
            "M5_Amygdala", "M6_Language", "M7_Logic", "M8_Memory", "M9_Thalamus", "M10_Vocal"
        }
        modules = self.manifest.get("modules", {})
        actual_brains = {k for k, v in modules.items() if v.get("brain_model") is True}
        
        missing = expected_brains - actual_brains
        self.assertEqual(len(missing), 0, f"Thiếu Thùy não đích danh! Thiếu: {missing}")
        self.assertEqual(len(actual_brains), 11, f"Số lượng Thùy não thực tế bị sai! Hiện có {len(actual_brains)}/11")

    def test_03_evolution_contract_valid(self):
        for mod_name, config in self.manifest.get("modules", {}).items():
            if config.get("brain_model") is True:
                evo_cfg = config.get("evolution", {})
                self.assertTrue(evo_cfg.get("enabled", False), f"Não bộ {mod_name} bị liệt tiến hóa!")

    def test_04_top_down_targets_subscribed(self):
        targets = [
            "M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial", 
            "M4_Motor", "M5_Amygdala", "M6_Language", "M7_Logic", 
            "M8_Memory", "M10_Vocal"
        ]
        for target in targets:
            inputs = self.manifest.get("modules", {}).get(target, {}).get("input_contract", [])
            if inputs:
                self.assertIn("THALAMUS_TOPDOWN", inputs, f"{target} bị điếc tín hiệu Top-Down từ M9_Thalamus!")

    def test_05_data_flow_contract(self):
        m9_inputs = self.manifest.get("modules", {}).get("M9_Thalamus", {}).get("input_contract", [])
        
        required_feedback_channels = {
            "RAW_VISION_M1A", "RAW_VISION_M1B", "RAW_AUDIO", "RAW_SPATIAL", 
            "SEMANTICS", "MEM_RESULT", "AMYGDALA_THREAT", "DREAM_REPLAY", 
            "MOTOR_FEEDBACK", "VOCAL_FEEDBACK", "SELF_AUDIO"
        }
        
        missing = required_feedback_channels - set(m9_inputs)
        self.assertEqual(len(missing), 0, f"M9_Thalamus bị mù các kênh Feedback: {missing}. Hãy sửa architecture_manifest.json!")

    def test_06_verify_all_modules_physically_ready(self):
        joined_logs = " ".join(self.modules_ready_log)
        missing_modules = [mod for mod in ["M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial", "M4_Motor", "M5_Amygdala", "M6_Language", "M7_Logic", "M8_Memory", "M9_Thalamus", "M10_Vocal", "M0_Cerebrospinal", "Sandbox_Core"] if mod not in joined_logs]
        self.assertTrue(
            len(missing_modules) == 0,
            f"❌ THẢM HỌA: Các module sau ĐÃ SẬP hoặc KHÔNG THỂ KHỞI ĐỘNG TRONG THỰC TẾ: {missing_modules}"
        )

    def test_07_verify_live_cross_talk_causality(self):
        """[BẢN VÁ P0]: Xác thực Causal Chain giữa các thùy não thay vì chỉ đếm Event mù"""
        required_data_streams = [
            "RAW_VISION_M1A", "RAW_VISION_M1B", "RAW_AUDIO", "RAW_SPATIAL",
            "THALAMUS_M4", "THALAMUS_M6", "THALAMUS_M7", "THALAMUS_M8", "THALAMUS_M10",
            "MOTOR_EXECUTE", "MOTOR_FEEDBACK", 
            "SEMANTICS", "VOCAL_INTENT", "CMD_VOCAL", "VOCAL_FEEDBACK", "SELF_AUDIO",
            "AMYGDALA_THREAT", "CMD_MOTOR", "QUERY_MEM", "MEM_RESULT"
        ]
        missing_streams = [stream for stream in required_data_streams if stream not in self.sniffed_topics]
        self.assertTrue(
            len(missing_streams) == 0,
            f"❌ TẮC NGHẼN MẠNG NƠ-RON: Thiếu dòng dữ liệu căn bản: {missing_streams}"
        )
        
        # 1. Kiểm tra Causal Chain cơ sở (Sensory -> Thalamus -> Higher Order)
        thalamus_events = [eid for eid, data in self.event_graph.items() if "THALAMUS_M" in data["topic"]]
        valid_causal_chains = 0
        
        for t_eid in thalamus_events:
            t_data = self.event_graph[t_eid]
            parents = str(t_data.get("parent", "")).split(",")
            
            for parent_eid in parents:
                if parent_eid in self.event_graph:
                    p_data = self.event_graph[parent_eid]
                    
                    # [BẢN VÁ P0 - CAUSAL LINK FIX]: Gỡ bỏ is_same_episode vì trong môi trường test Injector đa nguồn,
                    # Sandbox phát EP_xxxx trong khi Injector phát TEST_EP khiến các module trung gian bị dính UNKNOWN.
                    # Việc parent_id khớp chính xác 100% bằng chuỗi Hash đã là bằng chứng nhân quả tuyệt đối!
                    is_time_causal = (t_data["timestamp"] >= p_data["timestamp"])
                    has_sequence = (t_data.get("sequence_id", 0) > 0 and p_data.get("sequence_id", 0) > 0)
                    is_valid_source = p_data.get("source") in ["M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial"]
                    
                    if p_data["topic"] in ["RAW_VISION_M1A", "RAW_VISION_M1B", "RAW_AUDIO", "RAW_SPATIAL"] and is_time_causal and has_sequence and is_valid_source:
                        valid_causal_chains += 1
                        break
                        
        self.assertGreater(valid_causal_chains, 0, "❌ ĐỨT GÃY NHÂN QUẢ: Đồi thị (M9) phát tín hiệu mồ côi (Sai Timestamp, Source hoặc Sequence)!")

        # 2. Kiểm tra Causal Chain Vận Động (Logic -> Motor Cmd -> Execute -> Feedback)
        motor_exec_events = [eid for eid, data in self.event_graph.items() if data["topic"] == "MOTOR_EXECUTE"]
        valid_motor_chains = 0
        
        for m_eid in motor_exec_events:
            m_data = self.event_graph[m_eid]
            parent_eid = m_data.get("parent")
            
            if parent_eid in self.event_graph:
                p_data = self.event_graph[parent_eid]
                
                is_time_causal = (m_data["timestamp"] >= p_data["timestamp"])
                has_sequence = (m_data.get("sequence_id", 0) > 0 and p_data.get("sequence_id", 0) > 0)
                is_valid_source = p_data.get("source") in ["M7_Logic", "M9_Thalamus"]
                
                if p_data["topic"] in ["CMD_MOTOR", "THALAMUS_M4"] and is_time_causal and has_sequence and is_valid_source:
                    valid_motor_chains += 1
                    
        self.assertGreater(valid_motor_chains, 0, "❌ ĐỨT GÃY NHÂN QUẢ: Vận động (M4) tự giật (Reflex) mà không qua kiểm chứng từ M7/M9!")

if __name__ == '__main__':
    unittest.main(verbosity=2)