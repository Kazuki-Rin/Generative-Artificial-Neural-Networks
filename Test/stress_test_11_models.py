#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import time
import json
import random
import subprocess
import threading
import zmq
from collections import defaultdict
from loguru import logger

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

HANDSHAKE_TIMEOUT = 600.0
TEST_DURATION = 1800.0  

EXPECTED_11 = [
    "M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial",
    "M4_Motor", "M5_Amygdala", "M6_Language", "M7_Logic", "M8_Memory",
    "M9_Thalamus", "M10_Vocal"
]

class RealStressTest:
    def __init__(self):
        self.master_proc = None
        self.system_started = False
        self.evolution_count = 0
        self.last_evo_time = 0
        self.stop_event = threading.Event()
        self.received_model_ids = set() 
        self.active_transactions = {}
        self.completed_generations = []
        self.sniff_thread = None
        self.injector_thread = None
        self.traffic_stats = defaultdict(int)

    def start_master(self):
        main_script = os.path.join(BASE_DIR, "main_life_loop.py")
        logger.info("🚀 Đang đánh thức MasterSupervisor... (Hãy kiên nhẫn)")
        self.master_proc = subprocess.Popen(
            [sys.executable, "-u", main_script],
            cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        threading.Thread(target=self._monitor_log, daemon=True).start()
        
        # [BẢN VÁ TỐI THƯỢNG]: Bật Máy quét và Ống tiêm NGAY LẬP TỨC để tránh lỗi Late Joiner
        self.sniff_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.sniff_thread.start()
        
        self.injector_thread = threading.Thread(target=self._stimulus_injector_loop, daemon=True)
        self.injector_thread.start()

    def _stimulus_injector_loop(self):
        """Khởi động sớm, dùng Raw Socket chuẩn JSON để trị triệt để ZeroMQ"""
        logger.info("💉 [INJECTOR] Chờ 5s cho Proxy Master mở cửa...")
        time.sleep(5.0) 
        
        context = zmq.Context.instance()
        pub = context.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.connect("tcp://127.0.0.1:5556")
        
        logger.info("💉 [INJECTOR] Đã hòa mạng thành công! Đang vào trạng thái bơm tĩnh (10Hz)...")
        seq = 0
        
        while not self.stop_event.is_set():
            seq += 1
            import math, uuid
            def inject(topic, modality, payload, source="Sandbox_Core", override_parent=None):
                evt_id = f"evt_{uuid.uuid4().hex[:12]}"
                packet = {
                    "schema_version": 1,
                    "episode_id": "STRESS_EP",
                    "tick_id": seq,
                    "sequence_id": seq,
                    "task_id": "DUMMY_TASK",
                    "event_id": evt_id,
                    "parent_event_id": override_parent,
                    "source": source,
                    "modality": modality,
                    "timestamp": time.time(),
                    "payload": payload
                }
                msg = f"{topic} {json.dumps(packet, ensure_ascii=False)}"
                pub.send_string(msg)
                return evt_id

            inject("SYSTEM_CONTROL", "control", {"state": "START_TRAINING"})
            
            # 1. Nâng cấp mảng thành 1600 điểm dữ liệu để ép SNN hoạt động tối đa
            vision_pattern = [math.sin(i * 0.1) * 0.5 + 0.5 for i in range(1600)]
            audio_pattern = [math.cos(i * 0.05) * 0.5 + 0.5 for i in range(1600)]
            
            inject("ENV_VISION", "vision", {"data": vision_pattern, "target_labels": [0.8, 0.8], "type": "array"})
            audio_evt_id = inject("ENV_AUDIO", "audio", {"data": audio_pattern, "label": "Vowel"})
            inject("ENV_SPATIAL", "spatial", {"x": 0.5, "y": 0.5})
            
            # 2. XÓA BỎ BYPASS CAUSAL TREE - ĐỂ NÃO BỘ TỰ CHẠY THEO CHUỖI NHÂN QUẢ (CAUSAL CHAIN)
            inject("ENV_REWARD", "reward", {"reward": 0.8, "rpe": 0.3, "task_id": "DUMMY_TASK", "tick_id": seq, "terminated": False, "truncated": False})
            inject("ENV_HAZARD", "threat", {"is_danger": 1 if seq % 20 == 0 else 0})
            
            time.sleep(0.1)
            
        pub.close(linger=0)

    def _sniff_loop(self):
        context = zmq.Context.instance()
        sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.connect("tcp://127.0.0.1:5555")
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        
        while not self.stop_event.is_set():
            if poller.poll(100):
                try:
                    parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                    if parts:
                        topic_bytes = parts[0].split(b" ", 1)[0]
                        topic = topic_bytes.decode('ascii', errors='ignore').strip()
                        if topic: 
                            self.traffic_stats[topic] += 1
                except Exception: pass
        sub.close(linger=0)

    def _monitor_log(self):
        try:
            for line in self.master_proc.stdout:
                line_str = line.strip()
                if not line_str: continue
                
                if any(k in line_str for k in ["READY!", "READY (PubSub)", "READY (epoch="]):
                    for mod in EXPECTED_11:
                        if mod in line_str:
                            self.received_model_ids.add(mod)
                if "START_TRAINING" in line_str:
                    self.system_started = True
                
                if "[PHASE 1] KHỞI TẠO 2-PHASE COMMIT. Transaction:" in line_str:
                    tx_id = line_str.split("Transaction:")[-1].strip()
                    self.active_transactions[tx_id] = "PREPARING"
                    
                elif "[PHASE 2 OK] 11/11 Module đã COMMIT_APPLIED" in line_str:
                    logger.info("✔️ [2PC] Master đã xác nhận gom đủ 11/11 tín hiệu CONTROL_ACK nội bộ.")
                    
                elif "🌟 [GLOBAL COMMIT] Thế hệ" in line_str and "thành công" in line_str:
                    try:
                        gen_str = line_str.split("Thế hệ")[1].split("thành công")[0].strip()
                        gen = int(gen_str)
                        if gen not in self.completed_generations:
                            self.completed_generations.append(gen)
                            self.evolution_count += 1
                            logger.success(f"🧬 [2PC VERIFIED] Giao dịch HOÀN TẤT. 11/11 Nodes đồng thuận nâng lên Generation {gen}.")
                    except (IndexError, ValueError):
                        pass
        except ValueError: pass

    def wait_handshake(self):
        logger.info(f"⏳ Đang chờ nạp 11 mô hình vào RAM (Timeout: {HANDSHAKE_TIMEOUT}s)...")
        start = time.time()
        while time.time() - start < HANDSHAKE_TIMEOUT:
            if self.master_proc.poll() is not None:
                logger.critical("💀 Master Process đã sập ngầm!")
                return False
            if self.system_started and set(self.received_model_ids) == set(EXPECTED_11):
                logger.success("🎉 Asuna đã thức tỉnh trọn vẹn 11/11 thùy não tiến hóa!")
                return True
            time.sleep(1.0)
        return False

    def verify_causal_chains(self):
        stats = self.traffic_stats
        vital_chains = {
            "Sensory Input": ["ENV_VISION", "ENV_AUDIO", "ENV_SPATIAL"],
            "Dorsal/Ventral Stream": ["RAW_VISION_M1A", "RAW_VISION_M1B"],
            "Auditory/Spatial": ["RAW_AUDIO", "RAW_SPATIAL"], 
            "Thalamic Routing": ["THALAMUS_M4", "THALAMUS_M6", "THALAMUS_M7", "THALAMUS_M8", "THALAMUS_M10"],
            "Cognitive/Logic": ["CMD_MOTOR", "CMD_VOCAL", "SEMANTICS"],
            "Motor/Vocal Execute": ["MOTOR_EXECUTE", "VOCAL_FEEDBACK"],
            "Feedback Loops": ["MOTOR_FEEDBACK", "SELF_AUDIO"],
            "Neuromodulation": ["GLOBAL_DOPAMINE", "GLOBAL_PLASTICITY", "CEREBROSPINAL_FLUID"]
        }
        all_passed = True
        logger.info("==================================================")
        logger.info("🔍 KẾT QUẢ QUÉT BĂNG THÔNG DATA PLANE THỰC TẾ:")
        logger.info(f"⚡ Tổng quan danh sách các gói tin đã bắt được: {list(stats.keys())}")
        logger.info("==================================================")
        
        for chain_name, topics in vital_chains.items():
            missing_topics = [t for t in topics if stats.get(t, 0) == 0]
            if len(missing_topics) == 0:
                logger.success(f"✔️ {chain_name}: HOẠT ĐỘNG TỐT")
                for t in topics:
                    logger.info(f"     └─ {t}: {stats[t]} gói tin")
            else:
                logger.warning(f"❌ {chain_name}: THIẾU TÍN HIỆU {missing_topics}")
                all_passed = False
                
        logger.info("==================================================")
        return all_passed

    def run_stress_test(self):
        logger.info(f"🔥 BẮT ĐẦU ÉP XUNG TRONG {TEST_DURATION} GIÂY (30 PHÚT)...")
        start = time.time()
        last_log_min = 0
        try:
            while time.time() - start < TEST_DURATION:
                if self.master_proc.poll() is not None: return False
                
                elapsed_min = int(time.time() - start) // 60
                if elapsed_min > last_log_min:
                    logger.info(f"⏳ Đã trụ vững {elapsed_min} phút... (Số lần tiến hóa: {self.evolution_count})")
                    self.verify_causal_chains()
                    last_log_min = elapsed_min
                
                time.sleep(1.0)
                    
        except KeyboardInterrupt:
            logger.warning("🛑 Người dùng chủ động ngắt Stress Test.")
            self.verify_causal_chains()
        return True

    def shutdown(self):
        self.stop_event.set()
        if self.master_proc:
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.master_proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.master_proc.terminate()
                try: self.master_proc.wait(timeout=5)
                except Exception: self.master_proc.kill()

if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    test = RealStressTest()
    test.start_master()
    if test.wait_handshake(): test.run_stress_test()
    test.shutdown()