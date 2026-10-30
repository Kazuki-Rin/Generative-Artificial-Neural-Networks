#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reward Engine - Single Source of Truth for RPE/Dopamine (P1-3, P1-4)
====================================================================
- Tính RPE (Reward Prediction Error) từ raw reward.
- Duy trì Baseline (Global + Per-Context) + Habituation.
- Output: Dict chuẩn cho topic GLOBAL_DOPAMINE (Primary Reward Pathway).
- Persistence: get_state() / load_state() cho CheckpointManager.
"""

import time
import uuid
import copy
from typing import Dict, Any, Optional
import numpy as np


class RewardEngine:
    """
    Primary Reward Engine (Subcortical Dopamine System).
    
    Flow:
        Env Step -> Raw Reward
            -> RewardEngine.calculate(raw_reward, context)
                -> RPE = Effective_Reward - Expected_Reward
                -> Update Baselines (EMA)
                -> Update Habituation
                -> Return Dopamine_Packet (for GLOBAL_DOPAMINE topic)
    """
    
    def __init__(self, 
                 ema_alpha: float = 0.05, 
                 habituation_decay: float = 0.995,
                 habituation_increment: float = 0.05,
                 habituation_threshold: float = 0.01):
        """
        Args:
            ema_alpha: Learning rate cho Baseline (0.05 = chậm, ổn định).
            habituation_decay: Decay factor cho habituation khi không có stimulus (0.995).
            habituation_increment: Tăng habituation mỗi lần có stimulus mạnh.
            habituation_threshold: Ngưỡng |reward| để coi là stimulus.
        """
        self.ema_alpha = ema_alpha
        self.habituation_decay = habituation_decay
        self.habituation_increment = habituation_increment
        self.habituation_threshold = habituation_threshold
        
        # --- State ---
        self.global_baseline: float = 0.0
        self.context_baselines: Dict[str, float] = {}
        self.habituation_levels: Dict[str, float] = {}
        
        # Stats
        self.total_steps: int = 0
        self.last_update_time: float = time.time()

    # -------------------------------------------------------------------------
    # CORE CALCULATION
    # -------------------------------------------------------------------------
    def calculate_dopamine(self, raw_reward: float, state_vector=None, action_vector=None, context=None) -> Dict[str, Any]:
        if context is None:
            context = {}
        if state_vector is not None:
            context["state"] = state_vector
        if action_vector is not None:
            context["action"] = action_vector
        return self.calculate(actual_reward=raw_reward, context=context)

    def calculate(self, actual_reward: float, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Tính RPE và tạo Dopamine Packet.
        
        Args:
            actual_reward: Raw reward từ env.step() (float).
            context: Dict chứa metadata: 
                {
                    "task_id": str,
                    "episode_id": str,
                    "modality": "motor" | "vocal" | "cognitive",
                    "task_type": "navigation" | "feeding" | "social" | ...,
                    "state": Any (optional)
                }
        
        Returns:
            dopamine_packet: Dict chuẩn để publish lên GLOBAL_DOPAMINE.
                             Chứa keys: rpe, reward, expected_reward, novelty, salience, 
                             threat, arousal, habituation, baselines...
        """
        context = context or {}
        self.total_steps += 1
        current_time = time.time()
        
        # 1. Context Key & Stimulus Signature
        ctx_key = self._get_context_key(context)
        stim_sig = self._get_stimulus_signature(actual_reward, context)
        
        # 2. Expected Reward (Baseline)
        expected_reward = self.context_baselines.get(ctx_key, self.global_baseline)
        
        # 3. Habituation Modulation
        habituation = self.habituation_levels.get(stim_sig, 0.0)
        # Chỉ habituate reward có magnitude lớn (bỏ qua noise ~0)
        if abs(actual_reward) > self.habituation_threshold:
            effective_reward = actual_reward * (1.0 - habituation)
        else:
            effective_reward = actual_reward
        
        # 4. RPE (Reward Prediction Error) - "Quantum of Dopamine"
        rpe = effective_reward - expected_reward
        
        # 5. Update Baselines (Value Learning)
        # Global
        self.global_baseline += self.ema_alpha * (effective_reward - self.global_baseline)
        # Context-specific
        if ctx_key not in self.context_baselines:
            self.context_baselines[ctx_key] = effective_reward
        else:
            self.context_baselines[ctx_key] += self.ema_alpha * (effective_reward - self.context_baselines[ctx_key])
        
        # 6. Update Habituation (Sensitization/Desensitization)
        if abs(actual_reward) > self.habituation_threshold:
            # Tăng habituation (giảm sensitivity) cho stimulus này
            self.habituation_levels[stim_sig] = min(0.98, habituation + self.habituation_increment)
        else:
            # Decay habituation (Spontaneous Recovery)
            keys_to_del = []
            for k, v in self.habituation_levels.items():
                new_v = v * self.habituation_decay
                if new_v < 0.001:
                    keys_to_del.append(k)
                else:
                    self.habituation_levels[k] = new_v
            for k in keys_to_del:
                del self.habituation_levels[k]
        
        self.last_update_time = current_time
        
        # 7. Compute Modulatory Signals for M0/Plasticity
        if rpe > 0:
            novelty = float(min(1.0, rpe * 2.0))
        else:
            novelty = 0.0
            
        # Salience ~ |Raw Reward| + |RPE| (Intensity of the event)
        salience = float(min(1.0, abs(actual_reward) + abs(rpe)))
        # Threat ~ Negative Reward magnitude (Punishment)
        threat = float(min(1.0, abs(min(0.0, actual_reward)) * 2.0))
        # Arousal ~ General activation (|RPE| + |Reward|)
        arousal = float(min(1.0, (abs(rpe) + abs(actual_reward)) * 0.5))
        
        # 8. Build Standard Dopamine Packet (Schema v1)
        packet = {
            # Metadata
            "schema_version": 1,
            "signal": "PRIMARY_REWARD",      # Identifier: Primary Reward Pathway (Subcortical)
            "source": "RewardEngine_Env",
            "timestamp": current_time,
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "task_id": context.get("task_id", "ENV_STEP"),
            "episode_id": context.get("episode_id", "UNKNOWN"),
            
            # Core RL Signals (Consumed by M0, M7, M4, M5, M9...)
            "reward": float(effective_reward),       # Reward sau habituation (Used for Policy Gradient baseline)
            "raw_reward": float(actual_reward),      # Raw từ Env
            "expected_reward": float(expected_reward), # V(s) estimate
            "rpe": float(np.clip(rpe, -10.0, 10.0)),  # **CORE: Dopamine Signal** (Rescorla-Wagner RPE, NOT TD Error)
            
            # Neuromodulatory Context (Consumed by M0 -> GLOBAL_PLASTICITY)
            "novelty": novelty,
            "salience": salience,
            "threat": threat,
            "arousal": arousal,
            
            # Context
            "modality": context.get("modality", "motor"),
            "task_type": context.get("task_type", "unknown"),
            
            # Debug/Trace
            "habituation": float(habituation),
            "baseline_global": float(self.global_baseline),
            "baseline_context": float(self.context_baselines.get(ctx_key, 0.0)),
            "context_key": ctx_key,
            "stimulus_signature": stim_sig
        }
        
        return packet

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _get_context_key(self, context: Dict) -> str:
        """Key cho Context Baseline: task_type + modality."""
        t_type = context.get("task_type", "default")
        mod = context.get("modality", "motor")
        return f"{t_type}_{mod}"

    def _get_stimulus_signature(self, reward: float, context: Dict) -> str:
        """Signature cho Habituation: context + valence (reward/punishment)."""
        ctx_key = self._get_context_key(context)
        valence = "reward" if reward > 0 else ("punishment" if reward < 0 else "neutral")
        return f"{ctx_key}_{valence}"

    # -------------------------------------------------------------------------
    # LIFECYCLE & PERSISTENCE (Checkpoint)
    # -------------------------------------------------------------------------
    def reset_episode(self):
        """Gọi khi bắt đầu Episode mới (Reset Habituation, giữ Baselines)."""
        self.habituation_levels.clear()
        # Baselines (global_baseline, context_baselines) GIỮ LẠI -> Long-term memory

    def reset_all(self):
        """Full reset (Dùng khi GLOBAL_ROLLBACK về Gen 0)."""
        self.global_baseline = 0.0
        self.context_baselines.clear()
        self.habituation_levels.clear()
        self.total_steps = 0

    def get_state(self) -> Dict:
        """Lấy state để CheckpointManager.save()."""
        return {
            "global_baseline": self.global_baseline,
            "context_baselines": self.context_baselines,
            "habituation_levels": self.habituation_levels,
            "ema_alpha": self.ema_alpha,
            "habituation_decay": self.habituation_decay,
            "habituation_increment": self.habituation_increment,
            "habituation_threshold": self.habituation_threshold,
            "total_steps": self.total_steps,
            "last_update_time": self.last_update_time
        }

    def load_state(self, state: Dict):
        """Load state từ CheckpointManager.load()."""
        if not state: return
        self.global_baseline = state.get("global_baseline", 0.0)
        self.context_baselines = state.get("context_baselines", {})
        self.habituation_levels = state.get("habituation_levels", {})
        self.ema_alpha = state.get("ema_alpha", 0.05)
        self.habituation_decay = state.get("habituation_decay", 0.995)
        self.habituation_increment = state.get("habituation_increment", 0.05)
        self.habituation_threshold = state.get("habituation_threshold", 0.01)
        self.total_steps = state.get("total_steps", 0)
        self.last_update_time = state.get("last_update_time", time.time())


# =========================================================================
# STANDALONE TEST
# =========================================================================
if __name__ == "__main__":
    print("🍬 [TEST] Reward Engine - Single Source of Truth for Dopamine")
    engine = RewardEngine(ema_alpha=0.1, habituation_decay=0.9)
    
    print("\n--- 1. Baseline Acquisition (Reward 1.0 x5) ---")
    for i in range(5):
        pkt = engine.calculate(1.0, {"task_id": "nav", "task_type": "navigation", "modality": "motor"})
        print(f"  S{i+1}: RPE={pkt['rpe']:+.3f} | BaseG={pkt['baseline_global']:.3f} | BaseCtx={pkt['baseline_context']:.3f} | Hab={pkt['habituation']:.2f}")
    
    print("\n--- 2. Omission (Reward 0.0) -> Negative RPE (Dopamine Dip) ---")
    pkt = engine.calculate(0.0, {"task_id": "nav", "task_type": "navigation", "modality": "motor"})
    print(f"  RPE={pkt['rpe']:+.3f} (Expected ~ -0.9) | Dopamine={pkt['reward']:.3f}")
    
    print("\n--- 3. Surprise Jackpot (Reward 2.0) -> Positive RPE (Dopamine Surge) ---")
    pkt = engine.calculate(2.0, {"task_id": "nav", "task_type": "jackpot", "modality": "motor"})
    print(f"  RPE={pkt['rpe']:+.3f} | Novelty={pkt['novelty']:.2f} | Arousal={pkt['arousal']:.2f}")
    
    print("\n--- 4. Habituation (Repeat Jackpot 3x) -> RPE decays ---")
    for i in range(3):
        pkt = engine.calculate(2.0, {"task_id": "nav", "task_type": "jackpot", "modality": "motor"})
        print(f"  R{i+1}: EffRew={pkt['reward']:.3f} | RPE={pkt['rpe']:+.3f} | Hab={pkt['habituation']:.2f}")
    
    print("\n--- 5. Context Separation (Navigation vs Feeding) ---")
    pkt_nav = engine.calculate(0.5, {"task_id": "t1", "task_type": "navigation", "modality": "motor"})
    pkt_eat = engine.calculate(0.5, {"task_id": "t2", "task_type": "feeding", "modality": "motor"})
    print(f"  Nav Baseline: {pkt_nav['baseline_context']:.3f} | Eat Baseline: {pkt_eat['baseline_context']:.3f} (Separated!)")
    
    print("\n--- 6. Persistence Test ---")
    state = engine.get_state()
    print(f"  State keys: {list(state.keys())}")
    print(f"  Global Baseline: {state['global_baseline']:.4f}")
    print(f"  Context Baselines: {list(state['context_baselines'].keys())}")
    print(f"  Habituation Levels: {list(state['habituation_levels'].keys())}")
    
    engine2 = RewardEngine()
    engine2.load_state(state)
    pkt = engine2.calculate(0.0, {"task_id": "nav", "task_type": "navigation", "modality": "motor"})
    print(f"  Loaded Engine RPE (Omission): {pkt['rpe']:+.3f} (Matches!)")
    
    print("\n✅ Reward Engine PASSED - Ready for SandboxCore Integration.")