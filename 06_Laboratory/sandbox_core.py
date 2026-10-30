#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import signal
import threading
import multiprocessing
from multiprocessing.synchronize import Event as MpEvent
import uuid
import numpy as np
import zmq
import cv2
import base64
import hashlib
from loguru import logger
from typing import Dict, Any, Optional, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core_bus import CoreBus
from sandbox_env import VPetEnvironment

# [BẢN VÁ TÍCH HỢP]: Import Reward Engine CHUẨN từ file bên ngoài
from reward_engine import RewardEngine

# =========================================================================
# SANDBOX CORE PROCESS
# =========================================================================
class SandboxCore:
    def __init__(self, stop_event: MpEvent):
        self.stop_event = stop_event
        
        # 1. CoreBus with Control Plane (DEALER)
        self.bus = CoreBus(node_id="Sandbox_Core", is_master=False)
        
        # 2. Upstream Socket (PUSH to Master Supervisor port 5554) for Heartbeat/Ready
        self.upstream_sock = zmq.Context().socket(zmq.PUSH)
        self.upstream_sock.connect("tcp://127.0.0.1:5554")
        
        # 3. Environment
        self.env = VPetEnvironment(screen_width=1920, screen_height=1080, render_width=320, render_height=180)
        self.reward_engine = RewardEngine() 
        
        # [BẢN VÁ AUDIT #15 & #31]: Khởi tạo Episode Manager để khép kín vòng lặp Ký ức
        from episode_manager import create_episode_manager_for_sandbox
        self.episode_manager = create_episode_manager_for_sandbox(self.bus, self.reward_engine)
        
        # 4. State (Khởi tạo ở trạng thái ngủ / chưa kích hoạt cho đến khi qua Barrier Readiness)
        self.episode_id = f"EP_{uuid.uuid4().hex[:12]}"
        self.running = True
        self.is_sleeping = True
        self._barrier_ready = False
        self.motor_inhibited = False 
        self._needs_reset = False
        self._last_sensor_pub = 0.0
        self._sensor_interval = 0.05
        self._last_heartbeat = 0.0
        self._heartbeat_interval = 5.0
        self.worker_epoch = int(os.environ.get("ASUNA_WORKER_EPOCH", "1"))
        self._known_worker_epochs: Dict[str, int] = {}
        
        # 5. Register Control Plane Handlers (Reliable Commands từ Master Router)
        self.bus.register_control_handler("EVOLUTION_PREPARE", self._handle_evolution_prepare)
        self.bus.register_control_handler("GLOBAL_COMMIT", self._handle_global_commit)
        self.bus.register_control_handler("GLOBAL_ROLLBACK", self._handle_global_rollback)
        self.bus.register_control_handler("ASLEEP", lambda p: self._set_sleep(True))
        self.bus.register_control_handler("AWAKE", lambda p: self._activate_barrier_and_wake("AWAKE"))
        self.bus.register_control_handler("AWAKE_AND_SAFE", lambda p: self._set_resume(p))
        self.bus.register_control_handler("START_TRAINING", lambda p: self._activate_barrier_and_wake("START_TRAINING"))
        self.bus.register_control_handler("RESUME", lambda p: self._set_resume(p))
        self.bus.register_control_handler("RESET", lambda p: self._reset_episode(publish_now=self._barrier_ready))
        self.bus.register_control_handler("EMERGENCY_STOP", lambda p: self._set_emergency())
        
        # 6. Data Plane Subscriptions
        self.bus.subscribe([
            "MOTOR_EXECUTE", 
            "VOCAL_FEEDBACK", 
            "CMD_VOCAL", 
            "SELF_AUDIO",
            "CEREBROSPINAL_FLUID",
            "GLOBAL_PLASTICITY",
            "SYSTEM_CONTROL" 
        ])
        
        logger.info("🧪 [SANDBOX_CORE] Initialized. Control Plane + Data Plane ready.")

    # -------------------------------------------------------------------------
    # CONTROL PLANE HANDLERS (P0-4 Reliable)
    # -------------------------------------------------------------------------
    def _handle_evolution_prepare(self, payload: Dict) -> Dict:
        tx_id = payload.get("transaction_id")
        ax = getattr(self.env, 'asuna_x', 0.0)
        ay = getattr(self.env, 'asuna_y', 0.0)
        state_hash = hashlib.sha256(f"{ax}{ay}{self.episode_id}".encode()).hexdigest()
        return {"candidate_hash": state_hash, "status": "PREPARED_INFRA"}

    def _handle_global_commit(self, payload: Dict) -> Dict:
        logger.debug(f"[SANDBOX] GLOBAL_COMMIT received. Episode: {self.episode_id}")
        return {"status": "COMMITTED_INFRA"}

    def _handle_global_rollback(self, payload: Dict) -> Dict:
        logger.warning(f"[SANDBOX] GLOBAL_ROLLBACK received. Scheduling Reset.")
        self._needs_reset = True
        return {"status": "ROLLED_BACK_INFRA"}

    def _activate_barrier_and_wake(self, trigger_cmd: str = "START_TRAINING"):
        was_ready = self._barrier_ready
        self._barrier_ready = True
        self.is_sleeping = False
        self.motor_inhibited = False
        if hasattr(self, 'episode_manager'):
            self.episode_manager.handle_system_control({"state": trigger_cmd, "source": "MasterSupervisor"})
        if not was_ready:
            logger.info(f"🔓 [SANDBOX] Barrier Readiness hoàn tất qua '{trigger_cmd}'. Bắt đầu phát Observation!")
            obs = self.env._get_obs()
            self._publish_observations(obs)
            self._last_sensor_pub = time.time()
        return {"status": "OK", "sleeping": False, "barrier_ready": True}

    def _set_sleep(self, sleeping: bool):
        self.is_sleeping = sleeping
        if hasattr(self, 'episode_manager'):
            self.episode_manager.handle_system_control({"state": "ASLEEP" if sleeping else "AWAKE", "source": "MasterSupervisor"})
        logger.info(f"🌙 [SANDBOX] Sleep mode: {sleeping}")
        return {"status": "OK", "sleeping": sleeping}

    def _set_emergency(self):
        self.motor_inhibited = True
        if hasattr(self, 'episode_manager'):
            self.episode_manager.handle_system_control({"state": "EMERGENCY_STOP", "source": "MasterSupervisor"})
        logger.critical(f"🚨 [SANDBOX] EMERGENCY STOP ACTIVATED. All motor execution rejected.")
        return {"status": "OK", "emergency": True}

    def _set_resume(self, payload: Optional[Dict] = None):
        self._barrier_ready = True
        self.motor_inhibited = False
        self.is_sleeping = False
        if isinstance(payload, dict):
            sync_ep = payload.get("episode_id")
            sync_tick = payload.get("tick_id")
            if sync_ep and sync_ep not in ("UNKNOWN", self.episode_id):
                self.episode_id = sync_ep
                self.bus.current_episode = sync_ep
                if hasattr(self, 'episode_manager'):
                    self.episode_manager.episode_id = sync_ep
            if isinstance(sync_tick, int) and hasattr(self, 'episode_manager'):
                self.episode_manager.current_tick = max(getattr(self.episode_manager, 'current_tick', 0), sync_tick)
                self.bus.current_tick = self.episode_manager.current_tick
        if hasattr(self, 'episode_manager'):
            self.episode_manager.handle_system_control({"state": "RESUME", "source": "MasterSupervisor"})
        obs = self.env._get_obs()
        self._publish_observations(obs)
        self._last_sensor_pub = time.time()
        logger.info(f"🟢 [SANDBOX] RESUME & RE-SYNC ACTIVATED (episode={self.episode_id}, tick={getattr(self.episode_manager, 'current_tick', 0)}).")
        return {"status": "OK", "emergency": False, "episode_id": self.episode_id}

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    def run(self):
        self._reset_episode(publish_now=False)
        self._send_upstream({
            "state": "READY",
            "module": "Sandbox_Core",
            "schema_verified": True,
            "load_mode": "READY",
            "worker_epoch": self.worker_epoch
        })
        
        last_cleanup = time.time()
        
        try:
            while not self.stop_event.is_set() and self.running:
                current_time = time.time()
                
                if getattr(self, '_needs_reset', False):
                    self._reset_episode(publish_now=self._barrier_ready and not self.is_sleeping)
                    self._needs_reset = False
                
                self.bus.process_control_commands(timeout=0)
                self._process_data_plane()
                
                if self._barrier_ready and not self.is_sleeping and current_time - self._last_sensor_pub > self._sensor_interval:
                    obs = self.env._get_obs()
                    self._publish_observations(obs)
                    self._last_sensor_pub = current_time
                
                if current_time - self._last_heartbeat > self._heartbeat_interval:
                    self._send_upstream({"state": "HEARTBEAT", "module": "Sandbox_Core", "worker_epoch": self.worker_epoch})
                    self._last_heartbeat = current_time
                
                if self._barrier_ready and current_time - last_cleanup > 300.0:
                    if hasattr(self, 'episode_manager'):
                        logger.info("🧹 [SANDBOX] Kích hoạt cơ chế xả rác định kỳ (Timeout 5 phút)...")
                        self.episode_manager.end_episode(force=True, final_context="TIMEOUT_CLEANUP")
                        self._needs_reset = True
                    last_cleanup = current_time
                
                time.sleep(0.005)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.exception(f"💀 [SANDBOX_CORE] Fatal Error: {e}")
        finally:
            self.shutdown()

    def _extract_data(self, msg: Any) -> Any:
        if not msg: return None
        if isinstance(msg, dict):
            parsed = dict(msg)
            if "payload" in parsed and isinstance(parsed["payload"], dict):
                for k, v in parsed["payload"].items():
                    if k not in parsed or (k == "task_id" and parsed.get("task_id") in (None, "NO_TASK")):
                        parsed[k] = v
            return parsed
            
        if isinstance(msg, str):
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
                except json.JSONDecodeError: return parts[1]
        return None

    def _process_data_plane(self):
        if not self.bus.poll(timeout_ms=0):
            return 
        raw_msgs = self.bus.receive_all(timeout_ms=0)
        if not raw_msgs:
            return
        for raw in raw_msgs:
            try:
                topic = raw.get("topic", "") if isinstance(raw, dict) else raw.split(" ", 1)[0]
                payload = self._extract_data(raw)
                if isinstance(payload, dict):
                    self._handle_data_plane_msg(topic, payload)
            except Exception as e:
                logger.warning(f"[SandboxCore] Parse data plane msg fail: {e}")

    # -------------------------------------------------------------------------
    # MOTOR / VOCAL HANDLING
    # -------------------------------------------------------------------------
    def _handle_motor_execute(self, payload: Dict, event_id: str):
        is_reflex = isinstance(payload, dict) and payload.get("action_name") == "EMERGENCY_REFLEX"
        if not self._barrier_ready or self.is_sleeping or (getattr(self, 'motor_inhibited', False) and not is_reflex):
            logger.debug("[SANDBOX] Lệnh Motor bị từ chối do chưa qua Barrier hoặc đang ASLEEP/EMERGENCY_STOP.")
            return
        if is_reflex:
            self.motor_inhibited = False

        try:
            dx = float(payload.get("dx", 0.0))
            dy = float(payload.get("dy", 0.0))
            click = float(payload.get("click", 0.0))
            task_id = payload.get("task_id", "NO_TASK")
            
            action = [np.clip(dx, -1.0, 1.0), np.clip(dy, -1.0, 1.0), np.clip(click, 0.0, 1.0)]
            obs, raw_reward, terminated, truncated, info = self.env.step(action)
            
            ax = getattr(self.env, 'asuna_x', 0.0)
            ay = getattr(self.env, 'asuna_y', 0.0)
            tx = getattr(self.env, 'target_x', 0.0)
            ty = getattr(self.env, 'target_y', 0.0)
            
            # [BẢN VÁ P0 - XỬ LÝ PHẦN THƯỞNG ĐÍCH THỰC]: Tính toán RPE qua RewardEngine thay vì tin tưởng dummy info
            if hasattr(self.reward_engine, 'calculate_dopamine'):
                dopamine_packet = self.reward_engine.calculate_dopamine(
                    raw_reward=raw_reward,
                    state_vector=[ax, ay, tx, ty],
                    action_vector=action
                )
            else:
                dopamine_packet = {"reward": raw_reward, "rpe": raw_reward, "novelty": 0.1}
            
            if hasattr(self, 'episode_manager'):
                self.episode_manager.current_tick = getattr(self.episode_manager, 'current_tick', 0) + 1
                current_tick = self.episode_manager.current_tick
            else:
                current_tick = self.bus.current_tick + 1
            self.bus.current_tick = current_tick
            
            # 1. MOTOR_FEEDBACK 
            fb_payload = {
                "task_id": task_id,
                "status": "DONE" if not (terminated or truncated) else "TERMINATED",
                "actual_dx": info.get("actual_dx", 0.0),
                "actual_dy": info.get("actual_dy", 0.0),
                "position": {"x": ax, "y": ay},
                "reward": raw_reward,
                "rpe": dopamine_packet.get("rpe", 0.0),
                "tick_id": current_tick
            }
            self.bus.publish("MOTOR_FEEDBACK", fb_payload, parent_event_id=event_id, episode_id=self.episode_id, modality="motor_feedback", tick_id=current_tick)
            
            # 2. ENV_REWARD 
            rew_payload = {
                "task_id": task_id,
                "reward": raw_reward,
                "rpe": dopamine_packet.get("rpe", 0.0),
                "terminated": terminated,
                "truncated": truncated,
                "distance_to_target": info.get("distance", -1),
                "tick_id": current_tick
            }
            self.bus.publish("ENV_REWARD", rew_payload, parent_event_id=event_id, episode_id=self.episode_id, modality="reward", tick_id=current_tick)
            
            # 3. Cập nhật tick_id cho gói dopamine nội bộ (Chỉ M7_Logic mới phát GLOBAL_DOPAMINE lên Bus)
            dopamine_packet["tick_id"] = current_tick
            
            # 4. ENV_HAZARD
            hazards = info.get("hazards", {"is_danger": 0, "hazard_type": "None"})
            hazard_payload = {"is_danger": hazards.get("is_danger", 0), "hazards": hazards, "position": {"x": ax, "y": ay}, "tick_id": current_tick}
            self.bus.publish("ENV_HAZARD", hazard_payload, episode_id=self.episode_id, modality="hazard", tick_id=current_tick)
            
            # 5. Lưu vào Lõi Ký ức
            state_summary = {"asuna_pos": (ax, ay), "target_pos": (tx, ty)}
            action_taken = {"dx": dx, "dy": dy, "click": click, "task_id": task_id, "action_name": "MOTOR_EXECUTE"}
            self.episode_manager.log_event(state_summary, action_taken, raw_reward, info, dopamine_packet=dopamine_packet)
            
            # 6. Episode Termination
            if terminated or truncated:
                logger.info(f"🏁 [SANDBOX] Episode End (Term: {terminated}, Trunc: {truncated}). Reset.")
                self.episode_manager.end_episode(force=True, final_context="TERMINATED" if terminated else "TRUNCATED")
                self._needs_reset = True
                
        except Exception as e:
            logger.error(f"❌ [SANDBOX] Motor Execute Error: {e}")

    def _handle_vocal_feedback(self, payload: Dict, event_id: str):
        try:
            acoustic = payload.get("acoustic_output", [0.5, 0.5])
            pitch_norm, volume_norm = acoustic[0], acoustic[1]
            task_id = payload.get("task_id", "NO_TASK")
            
            # 1. Đánh giá hành động thanh quản (Reward thô)
            # Thưởng nhẹ nếu phát âm có âm lượng đủ nghe (volume > 0.1) để khuyến khích phát ra tiếng
            raw_reward = 0.1 if volume_norm > 0.1 else 0.0
            
            freq = 200 + pitch_norm * 1800
            amp = volume_norm * 0.5
            
            sr = 16000
            duration = 0.5
            t = np.linspace(0, duration, int(sr * duration))
            wave = np.sin(2 * np.pi * freq * t) * amp
            wave += np.random.randn(len(wave)) * 0.01
            wave = wave.astype(np.float32)
            
            current_tick = self.episode_manager.current_tick if hasattr(self, 'episode_manager') else 0
            
            # 2. Tính Dopamine qua RewardEngine
            if hasattr(self.reward_engine, 'calculate_dopamine'):
                dopamine_packet = self.reward_engine.calculate_dopamine(
                    raw_reward=raw_reward,
                    state_vector=[pitch_norm, volume_norm, 0.0, 0.0],
                    action_vector=acoustic
                )
            else:
                dopamine_packet = {"reward": raw_reward, "rpe": raw_reward, "novelty": 0.1}
                
            dopamine_packet["tick_id"] = current_tick
            
            # 3. Phát xạ SELF_AUDIO
            audio_payload = {
                "label": "SELF_VOCAL_PHYSICAL",
                "data": wave.tolist(),
                "type": "self_voice_physical",
                "task_id": task_id,
                "source": "Sandbox_Core",
                "tick_id": current_tick
            }
            self.bus.publish("SELF_AUDIO", audio_payload, parent_event_id=event_id, episode_id=self.episode_id, modality="audio", tick_id=current_tick)
            
            # 4. Phát xạ ENV_REWARD cho chuỗi hành động thanh quản
            rew_payload = {
                "task_id": task_id,
                "reward": raw_reward,
                "rpe": dopamine_packet.get("rpe", 0.0),
                "terminated": False,
                "truncated": False,
                "distance_to_target": -1,
                "tick_id": current_tick
            }
            self.bus.publish("ENV_REWARD", rew_payload, parent_event_id=event_id, episode_id=self.episode_id, modality="reward", tick_id=current_tick)
            
            # 5. Chỉ M7_Logic phát GLOBAL_DOPAMINE sau khi nhận ENV_REWARD
            
            # 6. Ghi log Ký ức vào Episode Manager
            state_summary = {"vocal_pitch": pitch_norm, "vocal_volume": volume_norm}
            action_taken = {"task_id": task_id, "action_name": "VOCAL_EXECUTE", "acoustic_output": acoustic}
            if hasattr(self, 'episode_manager'):
                self.episode_manager.log_event(state_summary, action_taken, raw_reward, {}, dopamine_packet=dopamine_packet)
            
        except Exception as e:
            logger.error(f"❌ [SANDBOX] Vocal Feedback Error: {e}")
            
    # -------------------------------------------------------------------------
    # DATA PLANE ROUTER
    # -------------------------------------------------------------------------
    def _handle_data_plane_msg(self, topic: str, payload: Dict):
        event_id = payload.get("event_id", "") if isinstance(payload, dict) else ""
        
        if topic == "MOTOR_EXECUTE":
            self._handle_motor_execute(payload, event_id)
        elif topic == "VOCAL_FEEDBACK":
            if self._barrier_ready and not self.is_sleeping:
                self._handle_vocal_feedback(payload, event_id)
        elif topic == "CEREBROSPINAL_FLUID":
            incoming_tick = payload.get("tick", payload.get("tick_id"))
            incoming_ep = payload.get("episode_id")
            if isinstance(incoming_tick, int) and hasattr(self, 'episode_manager'):
                curr_t = getattr(self.episode_manager, 'current_tick', 0)
                self.episode_manager.current_tick = incoming_tick
                self.bus.current_tick = incoming_tick
                if incoming_tick < curr_t and (curr_t - incoming_tick) > 5:
                    logger.warning(f"🔄 [SANDBOX] M0_Cerebrospinal khởi động lại (tick {curr_t} -> {incoming_tick}). Đã đồng bộ lại nhịp thời gian!")
                    if self._barrier_ready and not self.is_sleeping:
                        obs = self.env._get_obs()
                        self._publish_observations(obs)
            if incoming_ep and incoming_ep not in ("UNKNOWN", self.episode_id) and not self._barrier_ready:
                self.episode_id = incoming_ep
                self.bus.current_episode = incoming_ep
                if hasattr(self, 'episode_manager'):
                    self.episode_manager.episode_id = incoming_ep
        elif topic == "SYSTEM_CONTROL":
            if isinstance(payload, dict):
                state = payload.get("state", "")
                src = payload.get("module", payload.get("source", ""))
                w_epoch = payload.get("worker_epoch")
                if state == "EMERGENCY_STOP":
                    self._set_emergency()
                elif state in ("HEARTBEAT", "READY", "MODULE_READY") and src and src not in ("Sandbox_Core", "MasterSupervisor"):
                    prev_ep = self._known_worker_epochs.get(src)
                    if isinstance(w_epoch, int):
                        self._known_worker_epochs[src] = w_epoch
                    is_restarted = (state in ("READY", "MODULE_READY")) or (prev_ep is not None and isinstance(w_epoch, int) and w_epoch > prev_ep)
                    if is_restarted and self._barrier_ready and not self.is_sleeping:
                        logger.info(f"🔄 [SANDBOX] Module '{src}' khởi động lại giữa phiên (epoch={w_epoch}). Phát đồng bộ lại Phiên={self.episode_id}, Tick={getattr(self.episode_manager, 'current_tick', 0)}.")
                        obs = self.env._get_obs()
                        self._publish_observations(obs)
                elif payload.get("source") in ("MasterSupervisor", "Sandbox_Core"):
                    if state == "ASLEEP":
                        self._set_sleep(True)
                    elif state in ("AWAKE", "START_TRAINING", "AWAKE_AND_SAFE"):
                        self._activate_barrier_and_wake(state)
                    elif state == "RESUME":
                        self._set_resume(payload)
                    elif state == "GLOBAL_COMMIT":
                        logger.debug("[SANDBOX] GLOBAL_COMMIT received (no-op).")
                    elif state in ("GLOBAL_ROLLBACK", "RESET"):
                        self._needs_reset = True
                        if hasattr(self, 'episode_manager'):
                            self.episode_manager.handle_system_control(payload)

    # -------------------------------------------------------------------------
    # OBSERVATION PUBLISHING
    # -------------------------------------------------------------------------
    def _publish_observations(self, obs: Dict):
        current_tick = self.episode_manager.current_tick if hasattr(self, 'episode_manager') else self.bus.current_tick
        self.bus.current_tick = current_tick
        
        if "visual" in obs:
            frame = obs["visual"]
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            
            proprio = obs.get("proprioception", [0,0])
            latent_vec = proprio.tolist() if hasattr(proprio, 'tolist') else list(proprio)
            
            sw = getattr(self.env, 'width', getattr(self.env, 'screen_width', 1920))
            sh = getattr(self.env, 'height', getattr(self.env, 'screen_height', 1080))
            tx = getattr(self.env, 'target_x', sw / 2)
            ty = getattr(self.env, 'target_y', sh / 2)
            target_x_norm = float(tx / sw) if sw > 0 else 0.5
            target_y_norm = float(ty / sh) if sh > 0 else 0.5
            
            self.bus.publish("ENV_VISION", {
                "type": "jpg_base64",
                "data": b64_frame,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "latent_vector": latent_vec,
                "target_labels": [target_x_norm, target_y_norm],
                "tick_id": current_tick
            }, episode_id=self.episode_id, modality="vision", tick_id=current_tick)
        
        if "audio" in obs:
            audio_data = obs["audio"]
            audio_list = audio_data.tolist() if hasattr(audio_data, 'tolist') else list(audio_data)
            self.bus.publish("ENV_AUDIO", {
                "label": "ambient",
                "data": audio_list,
                "type": "environment",
                "tick_id": current_tick
            }, episode_id=self.episode_id, modality="audio", tick_id=current_tick)
        
        if "spatial" in obs:
            vec = obs["spatial"]
            self.bus.publish("ENV_SPATIAL", {
                "x": float((vec[0] + 1.0) / 2.0),
                "y": float((vec[1] + 1.0) / 2.0),
                "tick_id": current_tick
            }, episode_id=self.episode_id, modality="spatial", tick_id=current_tick)

        hazards = self.env._detect_hazards(0.0, 0.0, 0.0)
        self.bus.publish("ENV_HAZARD", {
            "is_danger": hazards.get("is_danger", 0),
            "hazards": hazards,
            "position": {"x": getattr(self.env, "asuna_x", 0.0), "y": getattr(self.env, "asuna_y", 0.0)},
            "tick_id": current_tick
        }, episode_id=self.episode_id, modality="hazard", tick_id=current_tick)

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    def _reset_episode(self, publish_now: bool = False):
        self.episode_id = f"EP_{uuid.uuid4().hex[:12]}"
        self.bus.current_episode = self.episode_id
        self.reward_engine.reset_episode() 
        obs, info = self.env.reset()
        
        if hasattr(self, 'episode_manager'):
            self.episode_manager.start_episode(context="EXPLORATION", episode_id=self.episode_id)
            
        if publish_now and self._barrier_ready and not self.is_sleeping:
            self._publish_observations(obs)
            
        logger.info(f"🆕 [SANDBOX] New Episode prepared: {self.episode_id} (published={publish_now and self._barrier_ready})")

    def _send_upstream(self, payload: Dict):
        try:
            self.upstream_sock.send_string(f"UPSTREAM {json.dumps(payload)}", flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as e:
            logger.error(f"[SANDBOX] Upstream send error: {e}")

    # =========================================================================
    # SHUTDOWN (PHẢI NẰM TRONG CLASS)
    # =========================================================================
    def shutdown(self):
        self.running = False
        try: self.upstream_sock.close(linger=0)
        except: pass
        self.bus.shutdown()
        logger.info("🛑 [SANDBOX_CORE] Shutdown complete.")


def start_sandbox_core(stop_event: MpEvent):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    core = SandboxCore(stop_event)
    core.run()


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 SANDBOX CORE STANDALONE TEST")
    print("=" * 60)
    
    stop_evt = multiprocessing.Event()
    core = SandboxCore(stop_evt)
    
    def test_commands():
        time.sleep(1)
        core._activate_barrier_and_wake("STANDALONE_TEST")
        time.sleep(1)
        print("\n🧪 [TEST] Sending MOTOR_EXECUTE via Data Plane...")
        core.bus.publish("MOTOR_EXECUTE", {
            "dx": 0.5, "dy": 0.0, "click": 0.0, 
            "task_id": "TEST_MOVE", "action_name": "MOVE_RIGHT"
        }, episode_id=core.episode_id)
        time.sleep(1)
        print("🧪 [TEST] Sending VOCAL_FEEDBACK (Simulate M10)...")
        core.bus.publish("VOCAL_FEEDBACK", {
            "acoustic_output": [0.8, 0.6], "task_id": "TEST_VOC", "status": "DONE"
        }, episode_id=core.episode_id)
        time.sleep(2)
        stop_evt.set()
    
    t = threading.Thread(target=test_commands, daemon=True)
    t.start()
    
    try:
        core.run()
    except KeyboardInterrupt:
        pass
    finally:
        core.shutdown()