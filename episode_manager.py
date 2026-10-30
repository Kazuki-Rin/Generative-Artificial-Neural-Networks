#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
EPISODE MANAGER - Runtime Integrated (v2.3 Pylance Proof)
=========================================================
- Fix Pylance reportInvalidTypeForm: Future annotations + String Literals
- Clean imports, no namespace pollution
"""

import time
import uuid
import logging
from typing import Dict, Optional, Any

from core_bus import CoreBus

logger = logging.getLogger(__name__)

class EpisodeManager:
    def __init__(self, 
                 bus: CoreBus, 
                 reward_engine: "Any",
                 source_node: str = "Sandbox_Core", 
                 max_events_per_episode: int = 500):
        
        self.bus = bus
        self.reward_engine = reward_engine
        self.source_node = source_node
        self.max_events = max_events_per_episode
        
        self.current_episode: Optional[Dict] = None
        self.episode_counter = 0
        self.episode_id: Optional[str] = None
        self.is_sleeping = False
        self.current_tick = 0
        self.current_generation = 0

    def handle_system_control(self, payload: Dict):
        state = payload.get("state", "")
        
        if "brain_generation" in payload:
            self.current_generation = payload.get("brain_generation", self.current_generation)
        elif "payload" in payload and isinstance(payload["payload"], dict):
            self.current_generation = payload["payload"].get("brain_generation", self.current_generation)

        if state == "RESET" or state == "GLOBAL_ROLLBACK":
            logger.warning(f"🔄 [EpisodeManager] Received {state}. Forcing Episode Reset.")
            self.force_reset()
            self.start_episode(context="POST_ROLLBACK")
        elif state == "ASLEEP":
            self.is_sleeping = True
            logger.info("🌙 [EpisodeManager] Sleep mode ON. Pausing Episode logging.")
            if self.current_episode:
                self.end_episode(force=True, final_context="SLEEP_TRANSITION")
        elif state in ["AWAKE", "START_TRAINING", "RESUME"]:
            self.is_sleeping = False
            logger.info("☀️ [EpisodeManager] Wake mode ON. Ready for new Episode.")
            if not self.current_episode:
                self.start_episode(context="WAKE_UP")

    def force_reset(self):
        if self.current_episode:
            # [BẢN VÁ P0]: Không vứt bỏ (discard) dữ liệu! Lưu trữ an toàn toàn bộ Ký ức vào Thùy Nhớ trước khi xóa sạch.
            logger.warning(f"💾 [EpisodeManager] Force Reset: Saving {len(self.current_episode['events'])} events before reset.")
            self.end_episode(force=True, final_context="FORCE_RESET")
        self.current_episode = None
        self.episode_id = None

    def start_episode(self, context: str = "Exploration", episode_id: str = None) -> str:
        if self.is_sleeping:
            return "SLEEPING"
        self.episode_counter += 1
        self.episode_id = episode_id or f"EP_{self.episode_counter}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.current_episode = {
            "episode_id": self.episode_id, "context": context, "start_time": time.time(),
            "events": [], "cumulative_reward": 0.0, "cumulative_rpe": 0.0, "step_count": 0
        }
        return self.episode_id

    def log_event(self, state_summary: Dict, action_taken: Dict, raw_reward: float, env_info: Dict = None, dopamine_packet: Dict = None) -> str:
        if self.is_sleeping: return "IGNORED_SLEEPING"
        
        if not self.current_episode:
            self.start_episode(context="AUTO_RECOVERY")
            
        if len(self.current_episode["events"]) >= self.max_events:
            self.end_episode(force=True, final_context="BUFFER_FLUSH")
            self.start_episode(self.current_episode.get("context", "Continued") if self.current_episode else "Continued")

        dopamine_packet = dopamine_packet or {"rpe": 0.0, "reward": raw_reward}
        rpe = dopamine_packet.get("rpe", 0.0)
        effective_reward = dopamine_packet.get("reward", raw_reward)
        
        task_success = float(env_info.get("task_success", 0.0)) if env_info else 0.0
        dopamine_packet["task_success"] = task_success

        event = {
            "step": self.current_episode["step_count"], "timestamp": time.time(), "time_offset": round(time.time() - self.current_episode["start_time"], 3),
            "state": state_summary, "action": action_taken, "raw_reward": float(raw_reward),  
            "effective_reward": float(effective_reward), "rpe": float(rpe),                
            "dopamine_meta": dopamine_packet, "env_info": env_info or {}, "task_success": task_success
        }
        self.current_episode["events"].append(event)
        self.current_episode["cumulative_reward"] += raw_reward
        self.current_episode["cumulative_rpe"] += rpe
        self.current_episode["step_count"] += 1
        return "LOGGED"

    def end_episode(self, force: bool = False, final_context: str = None) -> Optional[Dict]:
        if not self.current_episode: return None
        if not force and len(self.current_episode["events"]) == 0:
            self.current_episode = None; return None

        duration = time.time() - self.current_episode["start_time"]
        events = self.current_episode["events"]
        peak_positive = max(events, key=lambda e: e["rpe"]) if events else None
        peak_negative = min(events, key=lambda e: e["rpe"]) if events else None
        
        action_stats = {}
        for e in events:
            act_name = e["action"].get("action_name", "UNKNOWN")
            action_stats[act_name] = action_stats.get(act_name, 0) + 1
        
        first_state = events[0]["state"] if events else {}
        last_state = events[-1]["state"] if events else {}
        
        summary_packet = {
            "episode_id": self.current_episode["episode_id"], "context": final_context or self.current_episode["context"],
            "duration_sec": round(duration, 2), "total_steps": self.current_episode["step_count"], "total_events": len(events),
            "cumulative_reward": round(self.current_episode["cumulative_reward"], 4), "cumulative_rpe": round(self.current_episode["cumulative_rpe"], 4),
            "avg_reward_per_step": round(self.current_episode["cumulative_reward"] / max(1, self.current_episode["step_count"]), 4),
            "peak_positive_event": {"step": peak_positive["step"], "rpe": peak_positive["rpe"], "action": peak_positive["action"].get("action_name"), "state": peak_positive["state"]} if peak_positive else None,
            "peak_negative_event": {"step": peak_negative["step"], "rpe": peak_negative["rpe"], "action": peak_negative["action"].get("action_name"), "state": peak_negative["state"]} if peak_negative else None,
            "action_distribution": action_stats, "start_state_summary": {"pos": first_state.get("asuna_pos"), "target": first_state.get("target_pos")},
            "end_state_summary": {"pos": last_state.get("asuna_pos"), "target": last_state.get("target_pos")},
            "source_node": self.source_node, "timestamp": time.time()
        }
        importance = self._calculate_importance(summary_packet)
        save_payload = {"type": "episodic_summary", "memory_content": summary_packet, "episode_id": self.episode_id, "generation": getattr(self, 'current_generation', 0), "importance": importance}
        try:
            self.bus.publish("SAVE_MEM", save_payload, modality="memory")
        except Exception as e:
            logger.error(f"❌ [EpisodeManager] Failed to Publish SAVE_MEM: {e}")
        
        self.current_episode = None; self.episode_id = None
        return summary_packet

    def _calculate_importance(self, summary: Dict) -> float:
        peak_pos = summary.get("peak_positive_event") or {}
        peak_neg = summary.get("peak_negative_event") or {}
        peak_rpe = max(abs(peak_pos.get("rpe", 0)), abs(peak_neg.get("rpe", 0)))
        
        reward_mag = abs(summary.get("cumulative_reward", 0))
        steps = summary.get("total_steps", 1)
        imp = (min(peak_rpe, 5.0) / 5.0 * 0.5) + (min(reward_mag, 10.0) / 10.0 * 0.3) + (min(steps, 1000) / 1000.0 * 0.2)
        return max(0.1, min(1.0, imp))

def create_episode_manager_for_sandbox(bus: CoreBus, reward_engine: "Any") -> "EpisodeManager":
    return EpisodeManager(bus=bus, reward_engine=reward_engine, source_node="Sandbox_Core", max_events_per_episode=500)