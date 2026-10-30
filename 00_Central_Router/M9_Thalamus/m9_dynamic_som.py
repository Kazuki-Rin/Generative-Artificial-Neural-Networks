#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

# ThÃªm Ä‘Æ°á»ng dáº«n gá»‘c Ä‘á»ƒ import CheckpointManager náº¿u cháº¡y Ä‘á»™c láº­p
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

class DynamicSOM(nn.Module):
    """
    Thalamus - Dynamic Self-Organizing Map vá»›i Adaptive Routing Matrix.
    
    Features:
    - SOM topology growth (Structural Evolution).
    - Adaptive Routing Matrix (Functional Evolution) via Hebbian/Neuromodulatory plasticity.
    - Eligibility Traces for credit assignment.
    - Top-Down Attention Generation.
    - 2-Phase Evolution Transaction Support (Prepare/Commit/Rollback).
    """
    def __init__(self, input_dim: int = 128, initial_nodes: int = 4, **_ignored_kwargs):
        super(DynamicSOM, self).__init__()
        self.input_dim = input_dim
        self.dynamic_threshold = 0.5
        
        # --- 1. SOM Core (Structural) ---
        self.weights = nn.Parameter(torch.randn(initial_nodes, input_dim) * 0.1)
        self.num_nodes = initial_nodes
        self.last_input = None
        self.topology_version = 1
        
        self.model_generation = 1
        self.global_generation = 1
        self.step_count = 0 
        
        # --- 2. Routing Matrix (Functional) ---
        self.modality_names = ["vision_where", "vision_what", "audio", "spatial", "threat", "semantics", "memory", "motor_fb", "vocal_fb", "self_audio", "dream_replay"]
        self.dest_names = ["M6_Language", "M7_Logic", "M8_Memory", "M4_Motor", "M10_Vocal"]
        self.num_dest = len(self.dest_names)
        self.num_modalities = len(self.modality_names)
        
        init_matrix = torch.zeros(self.num_dest, self.num_modalities)
        init_matrix[0] = torch.tensor([0.5, 0.8, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5]) # M6
        init_matrix[1] = torch.tensor([1.0, 1.0, 0.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.8]) # M7
        init_matrix[2] = torch.tensor([0.2, 0.5, 0.2, 0.2, 0.0, 0.8, 1.0, 0.2, 0.2, 0.2, 0.2]) # M8
        init_matrix[3] = torch.tensor([0.8, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0]) # M4
        init_matrix[4] = torch.tensor([0.0, 0.0, 0.8, 0.0, 0.5, 0.8, 0.0, 0.0, 0.5, 0.8, 0.0]) # M10
        
        self.attention_to_route = nn.Parameter(init_matrix, requires_grad=True)
        
        # --- 3. Plasticity State (Eligibility Traces) ---
        self.register_buffer('routing_eligibility', torch.zeros(self.num_dest, self.num_modalities))
        self.register_buffer('som_eligibility', torch.zeros(initial_nodes, input_dim))
        
        self.routing_lr = 0.01       
        self.som_lr = 0.05           
        self.eligibility_decay = 0.9 
        self.weight_decay = 1e-4     
        
        self.pending_plasticity = []
        self._sync_working_from_committed()
        
        # --- 4. 2-Phase Evolution State ---
        self._evo_state = "LIVE" 
        self.candidate_state = None
        self._candidate_weights = None
        self._candidate_num_nodes = 0
        self._candidate_routing_matrix = None
        self._candidate_topology_version = 0
        self._current_transaction_id = None
        self._parent_weights = None
        self._parent_routing_matrix = None
        self._parent_num_nodes = 0
        self._parent_topology_version = 0
        self._pending_growth = None
        self._verify_structural_integrity()

    def _verify_structural_integrity(
        self,
        weights: Optional[torch.Tensor] = None,
        routing_matrix: Optional[torch.Tensor] = None,
        som_elig: Optional[torch.Tensor] = None,
        routing_elig: Optional[torch.Tensor] = None,
        expected_nodes: Optional[int] = None
    ) -> bool:
        w = weights if weights is not None else self.weights
        rm = routing_matrix if routing_matrix is not None else self.attention_to_route
        se = som_elig if som_elig is not None else self.som_eligibility
        re = routing_elig if routing_elig is not None else self.routing_eligibility
        n_nodes = expected_nodes if expected_nodes is not None else self.num_nodes

        if n_nodes <= 0 or w.dim() != 2 or w.shape != (n_nodes, self.input_dim):
            raise RuntimeError(f"[M9_Thalamus] Lỗi toàn vẹn weights: shape={tuple(w.shape)}, kỳ vọng=({n_nodes}, {self.input_dim})")
        if rm.dim() != 2 or rm.shape != (self.num_dest, self.num_modalities):
            raise RuntimeError(f"[M9_Thalamus] Lỗi toàn vẹn routing matrix: shape={tuple(rm.shape)}, kỳ vọng=({self.num_dest}, {self.num_modalities})")
        if se.dim() != 2 or se.shape != (n_nodes, self.input_dim):
            raise RuntimeError(f"[M9_Thalamus] Lỗi toàn vẹn som_eligibility: shape={tuple(se.shape)}, kỳ vọng=({n_nodes}, {self.input_dim})")
        if re.dim() != 2 or re.shape != (self.num_dest, self.num_modalities):
            raise RuntimeError(f"[M9_Thalamus] Lỗi toàn vẹn routing_eligibility: shape={tuple(re.shape)}, kỳ vọng=({self.num_dest}, {self.num_modalities})")
        if not (torch.all(torch.isfinite(w)) and torch.all(torch.isfinite(rm)) and torch.all(torch.isfinite(se)) and torch.all(torch.isfinite(re))):
            raise RuntimeError("[M9_Thalamus] Phát hiện NaN/Inf trong weights, routing matrix hoặc eligibility traces!")
        return True

    def _can_grow_topology(self, target_nodes: int) -> Tuple[bool, str]:
        import json
        max_nodes = 500
        max_model_params = 40000
        global_max_params = 500000
        max_rss_mb = 2048.0
        arch_path = os.path.join(BASE_DIR, "architecture_manifest.json")
        if os.path.exists(arch_path):
            try:
                with open(arch_path, "r", encoding="utf-8") as f:
                    arch = json.load(f)
                gov = arch.get("system_topology", {}).get("resource_governor", arch.get("governor", {}))
                m9_cfg = arch.get("modules", {}).get("M9_Thalamus", {})
                evo_cfg = m9_cfg.get("evolution", {})
                if not evo_cfg.get("structural", True):
                    return False, "evolution.structural=false trong manifest"
                node_caps = gov.get("max_nodes_per_module", {})
                param_caps = gov.get("max_parameters_per_module", {})
                max_nodes = min(
                    int(node_caps.get("M9_Thalamus", max_nodes)),
                    int(evo_cfg.get("max_nodes", m9_cfg.get("max_nodes", max_nodes)))
                )
                max_model_params = min(
                    int(param_caps.get("M9_Thalamus", gov.get("max_params_per_model", max_model_params))),
                    int(evo_cfg.get("max_parameters", gov.get("max_params_per_model", max_model_params)))
                )
                global_max_params = int(gov.get("global_max_parameters", global_max_params))
                max_rss_mb = float(gov.get("max_process_rss_mb", max_rss_mb))
            except Exception:
                pass
        if target_nodes > max_nodes:
            return False, f"Vượt giới hạn node của Governor ({target_nodes} > {max_nodes})"
        projected_params = target_nodes * self.input_dim + (self.num_dest * self.num_modalities)
        if projected_params > max_model_params or projected_params > global_max_params:
            return False, f"Vượt ngân sách tham số Governor ({projected_params} > {min(max_model_params, global_max_params)})"
        try:
            import psutil
            vm = psutil.virtual_memory()
            proc_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
            if vm.percent >= 90.0 or proc_rss_mb >= max_rss_mb:
                return False, f"Tài nguyên hệ thống chạm ngưỡng bảo vệ (SysRAM={vm.percent:.1f}%, ProcRSS={proc_rss_mb:.1f}MB)"
        except Exception:
            pass
        return True, "OK"

    def _sync_optimizer_after_mutation(self, optimizer: Optional[torch.optim.Optimizer], old_weights_param: Optional[nn.Parameter] = None):
        if optimizer is None:
            return
        if old_weights_param is not None and old_weights_param in optimizer.state:
            old_st = optimizer.state.pop(old_weights_param)
            new_st = {}
            for k, v in old_st.items():
                if torch.is_tensor(v) and v.shape == old_weights_param.shape:
                    nv = torch.zeros_like(self.weights, memory_format=torch.preserve_format)
                    rows = min(v.shape[0], nv.shape[0])
                    nv[:rows, :].copy_(v[:rows, :])
                    new_st[k] = nv
                else:
                    new_st[k] = copy.deepcopy(v)
            optimizer.state[self.weights] = new_st
        if len(optimizer.param_groups) > 0:
            optimizer.param_groups[0]["params"] = [p for p in self.parameters() if p.requires_grad]

    # =========================================================================
    # FORWARD & SOM CORE
    # =========================================================================
    def _sync_working_from_committed(self):
        self._working_weights = self.weights.detach().clone()
        self._working_routing = self.attention_to_route.detach().clone()
        self._working_routing_elig = self.routing_eligibility.detach().clone()
        self._working_som_elig = self.som_eligibility.detach().clone()

    def _ensure_working_copies(self):
        if getattr(self, '_working_weights', None) is None or self._working_weights.shape != self.weights.shape:
            self._working_weights = self.weights.detach().clone()
        if getattr(self, '_working_routing', None) is None or self._working_routing.shape != self.attention_to_route.shape:
            self._working_routing = self.attention_to_route.detach().clone()
        if getattr(self, '_working_routing_elig', None) is None or self._working_routing_elig.shape != self.routing_eligibility.shape:
            self._working_routing_elig = self.routing_eligibility.detach().clone()
        if getattr(self, '_working_som_elig', None) is None or self._working_som_elig.shape != self.som_eligibility.shape:
            self._working_som_elig = self.som_eligibility.detach().clone()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_input = x.detach()
        self._ensure_working_copies()
        active_weights = self._working_weights
        
        diff = x.unsqueeze(1) - active_weights.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        bmu_idx = torch.argmin(dist, dim=-1)
        return bmu_idx, dist

    def update_and_grow(self, x: torch.Tensor, current_tick: int = 0, current_episode: str = "UNKNOWN") -> Tuple[int, bool]:
        bmu_idx, dist = self.forward(x)
        bmu = bmu_idx[0].item()
        min_dist = dist[0, bmu].item()
        
        self.step_count += 1
        raw_thresh = 0.99 * getattr(self, 'dynamic_threshold', 0.5) + 0.01 * (min_dist * 3.0)
        self.dynamic_threshold = max(0.1, min(50.0, float(raw_thresh)))

        changed = False
        if min_dist > self.dynamic_threshold:
            can_grow, _ = self._can_grow_topology(self.num_nodes + 1)
            if can_grow:
                self._pending_growth = {
                    "bmu": bmu,
                    "x": x.detach().clone(),
                    "tick_id": current_tick,
                    "episode_id": current_episode
                }
                changed = True
            
        return bmu, changed

    # =========================================================================
    # ROUTING & PLASTICITY
    # =========================================================================
    def get_dynamic_routing(self, 
                            bmu_idx: int, 
                            attention_routing: Dict[str, float], 
                            salience: float, 
                            inhibition_level: float,
                            reward_signal: float = 0.0,      
                            prediction_error: float = 0.0,   
                            novelty: float = 0.0,            
                            task_success: float = 0.0        
                           ) -> Tuple[Dict[str, bool], List[float], Dict[str, float]]:
                           
        device = self.weights.device
        
        att_vec = torch.tensor([
            attention_routing.get("vision_where", 0.0),
            attention_routing.get("vision_what", 0.0),
            attention_routing.get("audio", 0.0),
            attention_routing.get("spatial", 0.0),
            attention_routing.get("threat", 0.0),
            attention_routing.get("semantics", 0.0),
            attention_routing.get("memory", 0.0),
            attention_routing.get("motor_fb", 0.0),
            attention_routing.get("vocal_fb", 0.0),
            attention_routing.get("self_audio", 0.0),
            attention_routing.get("dream_replay", 0.0)
        ], dtype=torch.float32, device=device)
        
        att_clean = torch.clamp(att_vec, min=0.0)
        att_sum = float(torch.sum(att_clean).item())
        if att_sum > 1e-6:
            att_vec_norm = att_clean / att_sum
        else:
            att_vec_norm = torch.zeros_like(att_clean)
        
        # [STRUCTURAL INVARIANT]: Prevent infinite echo-loop by masking out the M6 <- semantic edge
        with torch.no_grad():
            self._ensure_working_copies()
            self._working_routing[0, 5] = 0.0
            
        active_matrix = self._working_routing
        routing_logits = torch.matmul(active_matrix, att_vec_norm)
        eff_inhib = max(0.0, float(inhibition_level) - 0.3)
        
        panic_bias = torch.tensor([
            -2.0 * eff_inhib,  
            -2.0 * eff_inhib,  
            -1.0 * eff_inhib,  
            +3.0 * eff_inhib,  
            +1.0 * eff_inhib   
        ], dtype=torch.float32, device=device)
        
        dopamine_bias = torch.tensor([
            0.2 * abs(prediction_error) + 0.2 * novelty,                  
            0.5 * abs(prediction_error) + 0.5 * novelty,  
            1.0 * novelty + 0.5 * abs(prediction_error),  
            0.3 * abs(prediction_error) + 0.2 * max(0.0, reward_signal),                    
            0.5 * max(0.0, reward_signal) + 0.2 * novelty                      
        ], dtype=torch.float32, device=device)
        
        final_logits = (routing_logits * max(float(salience), 0.2)) + panic_bias + dopamine_bias
        probs = torch.sigmoid(final_logits) 
        
        destination_mask = {}
        for i, dest in enumerate(self.dest_names):
            destination_mask[dest] = bool(probs[i] > 0.5)
        if float(routing_logits[0].item()) <= 1e-6:
            destination_mask["M6_Language"] = False
        
        if not any(destination_mask.values()):
            if inhibition_level > 0.5: destination_mask["M4_Motor"] = True
            else: destination_mask["M7_Logic"] = True
            
        active_dest = probs.detach() 
        topdown_weights = torch.matmul(active_matrix.t(), active_dest) 
        topdown_weights = F.softmax(topdown_weights, dim=0) 
        
        bmu_weight = self._working_weights[bmu_idx].detach()
        gaze_x = float(torch.sigmoid(bmu_weight[0]).item()) 
        gaze_y = float(torch.sigmoid(bmu_weight[1]).item())
        
        topdown_attention = {
            "m1a_focus": topdown_weights[0].item(), 
            "m1b_focus": topdown_weights[1].item(), 
            "m2_focus":   topdown_weights[2].item(), 
            "m3_focus":   topdown_weights[3].item(), 
            "gaze": { "x": gaze_x, "y": gaze_y }    
        }
        
        with torch.no_grad():
            dest_act = probs.unsqueeze(1) 
            mod_act = att_vec_norm.unsqueeze(0) 
            
            if getattr(self, '_evo_state', 'LIVE') == 'LIVE':
                self._working_routing_elig.mul_(self.eligibility_decay).add_(dest_act @ mod_act)
                if self.last_input is not None and 0 <= bmu_idx < self._working_weights.shape[0]:
                    delta = self.last_input[0] - self._working_weights[bmu_idx]
                    self._working_som_elig.mul_(self.eligibility_decay)
                    self._working_som_elig[bmu_idx].add_(delta)
        
        mod_signal = reward_signal + 0.5 * novelty + 0.5 * task_success - 0.5 * abs(prediction_error)
        self.apply_plasticity(mod_signal, bmu_idx)
            
        return destination_mask, probs.tolist(), topdown_attention

    def apply_plasticity(self, mod_signal: float, bmu_idx: int = 0):
        if abs(mod_signal) <= 1e-4:
            return
            
        if getattr(self, '_evo_state', 'LIVE') != 'LIVE':
            if not hasattr(self, 'pending_plasticity'):
                self.pending_plasticity = []
            self.pending_plasticity.append((mod_signal, bmu_idx))
            return

        self._apply_routing_plasticity(mod_signal)
        self._apply_som_plasticity(mod_signal, bmu_idx)

    def _apply_routing_plasticity(self, mod_signal: float):
        with torch.no_grad():
            self._ensure_working_copies()
            delta = self.routing_lr * mod_signal * self._working_routing_elig
            decay = self.weight_decay * self._working_routing
            self._working_routing.add_(delta - decay)
            min_floor = torch.clamp(self.attention_to_route.detach(), min=0.0) * 0.25
            self._working_routing.copy_(torch.maximum(self._working_routing.clamp(-2.0, 2.0), min_floor))
            self._working_routing[0, 5] = 0.0

    def _apply_routing_constraints(self):
        with torch.no_grad():
            self.attention_to_route.data[0, 5] = 0.0
            if getattr(self, '_working_routing', None) is not None:
                self._working_routing[0, 5] = 0.0

    def _apply_som_plasticity(self, mod_signal: float, bmu_idx: int):
        if self.last_input is not None:
            with torch.no_grad():
                self._ensure_working_copies()
                if 0 <= bmu_idx < self._working_weights.shape[0]:
                    delta = self.som_lr * mod_signal * self._working_som_elig[bmu_idx]
                    self._working_weights[bmu_idx].add_(delta)

    # =========================================================================
    # 2-PHASE EVOLUTION (P0-3)
    # =========================================================================
    def prepare_candidate(self, transaction_id: str, target_tick: int = 0, target_episode: str = "UNKNOWN") -> Dict:
        if not transaction_id or not isinstance(transaction_id, str):
            raise ValueError("[M9_Thalamus] prepare_candidate yêu cầu transaction_id hợp lệ!")

        if self._evo_state == "CANDIDATE_STAGED" and self._current_transaction_id == transaction_id and self.candidate_state is not None:
            topo_changed = (self._candidate_num_nodes != self._parent_num_nodes)
            weights_or_routing_changed = (
                topo_changed
                or not torch.equal(self._candidate_weights, self._parent_weights)
                or not torch.equal(self._candidate_routing_matrix, self._parent_routing_matrix)
            )
            c_hash = self._compute_candidate_hash()
            return {
                "candidate_hash": c_hash if weights_or_routing_changed else f"UNCHANGED_{c_hash}",
                "topology_changed": topo_changed,
                "new_num_nodes": self._candidate_num_nodes,
                "transaction_id": transaction_id
            }

        if self._evo_state == "CANDIDATE_STAGED" and self._current_transaction_id not in (None, transaction_id):
            logger.warning(f"[M9_Thalamus] Hủy giao dịch treo cũ ({self._current_transaction_id}) để chuẩn bị giao dịch mới ({transaction_id}).")
            self.rollback_candidate(self._current_transaction_id)

        self._current_transaction_id = transaction_id
        self._evo_state = "CANDIDATE_STAGED"
        
        self._parent_weights = self.weights.detach().clone()
        self._parent_routing_matrix = self.attention_to_route.detach().clone()
        self._parent_num_nodes = self.num_nodes
        self._parent_topology_version = self.topology_version
        self._parent_step_count = self.step_count
        self._parent_dynamic_threshold = getattr(self, 'dynamic_threshold', 0.5)
        self._parent_last_input = self.last_input.clone() if self.last_input is not None else None
        self._parent_pending_growth = copy.deepcopy(self._pending_growth) if hasattr(self, '_pending_growth') else None
        self._parent_routing_eligibility = self.routing_eligibility.clone()
        self._parent_som_eligibility = self.som_eligibility.clone()
        self._parent_model_generation = getattr(self, 'model_generation', 1)
        self._parent_global_generation = getattr(self, 'global_generation', 1)
        
        self._ensure_working_copies()
        self._candidate_routing_matrix = self._working_routing.detach().clone()
        self._candidate_weights = self._working_weights.detach().clone()
        self._candidate_num_nodes = self.num_nodes
        self._candidate_topology_version = self.topology_version
        
        self._candidate_routing_eligibility = self._working_routing_elig.clone()
        self._candidate_som_eligibility = self._working_som_elig.clone()
        
        topology_changed = False
        
        pending = getattr(self, '_pending_growth', None)
        if pending is not None:
            can_grow, reason = self._can_grow_topology(self.num_nodes + 1)
            p_tick = int(pending.get("tick_id", 0))
            if target_tick > 0 and p_tick > target_tick:
                logger.warning(f"⚠️ [M9_Thalamus] Từ chối mọc Nơ-ron do tick tương lai ({p_tick} > {target_tick}).")
            elif not can_grow:
                logger.warning(f"⚠️ [M9_Thalamus] Từ chối mọc Nơ-ron: {reason}")
            else:
                new_weight = pending["x"][0].detach().clone() + torch.randn(self.input_dim, device=self.weights.device) * 0.1
                new_cand_weights = torch.cat([self._candidate_weights, new_weight.unsqueeze(0)], dim=0).detach().clone()
                self._candidate_weights = new_cand_weights
                self._candidate_num_nodes += 1
                self._candidate_topology_version = self.topology_version + 1
                topology_changed = True
                    
                new_elig = torch.zeros(self._candidate_num_nodes, self.input_dim, device=self.weights.device)
                new_elig[:self.som_eligibility.shape[0]] = self.som_eligibility.clone()
                self._candidate_som_eligibility = new_elig
                
            self._pending_growth = None

        self._verify_structural_integrity(
            weights=self._candidate_weights,
            routing_matrix=self._candidate_routing_matrix,
            som_elig=self._candidate_som_eligibility,
            routing_elig=self._candidate_routing_eligibility,
            expected_nodes=self._candidate_num_nodes
        )
        self.candidate_state = {
            "weights": self._candidate_weights.clone(),
            "attention_to_route": self._candidate_routing_matrix.clone(),
            "routing_eligibility": self._candidate_routing_eligibility.clone(),
            "som_eligibility": self._candidate_som_eligibility.clone()
        }
                
        cand_hash = self._compute_candidate_hash()
        weights_or_routing_changed = (
            topology_changed
            or not torch.equal(self._candidate_weights, self._parent_weights)
            or not torch.equal(self._candidate_routing_matrix, self._parent_routing_matrix)
        )
        if not weights_or_routing_changed:
            cand_hash = f"UNCHANGED_{cand_hash}"
        return {
            "candidate_hash": cand_hash,
            "topology_changed": topology_changed,
            "new_num_nodes": self._candidate_num_nodes,
            "transaction_id": transaction_id
        }

    def commit_candidate(
        self,
        transaction_id: str = None,
        target_generation: int = None,
        target_tick: int = 0,
        target_episode: str = "UNKNOWN",
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> bool:
        if not transaction_id:
            logger.error("[M9_Thalamus] Commit Failed: Thiếu transaction_id!")
            return False
        if getattr(self, 'last_committed_tx_id', None) == transaction_id:
            return True
        if self._evo_state != "CANDIDATE_STAGED" or self._current_transaction_id != transaction_id:
            logger.error(f"[M9_Thalamus] Commit Failed: Lệch transaction_id (Kỳ vọng={self._current_transaction_id}, Nhận={transaction_id})")
            return False

        self._verify_structural_integrity(
            weights=self._candidate_weights,
            routing_matrix=self._candidate_routing_matrix,
            som_elig=self._candidate_som_eligibility,
            routing_elig=self._candidate_routing_eligibility,
            expected_nodes=self._candidate_num_nodes
        )

        old_w_param = self.weights
        if self._candidate_weights is not None:
            if self._candidate_num_nodes != self.num_nodes:
                self.topology_version = max(self.topology_version + 1, self._candidate_topology_version)
            self.weights = nn.Parameter(self._candidate_weights.detach().clone())
            self.num_nodes = self._candidate_num_nodes
            self.register_buffer('som_eligibility', self._candidate_som_eligibility.detach().clone())

        if self._candidate_routing_matrix is not None:
            self.attention_to_route.data.copy_(self._candidate_routing_matrix)
            self._apply_routing_constraints()
            
        if self._candidate_routing_eligibility is not None:
            self.routing_eligibility.copy_(self._candidate_routing_eligibility)

        self._verify_structural_integrity()
        self._sync_working_from_committed()
        self._sync_optimizer_after_mutation(optimizer, old_weights_param=old_w_param)
        self.last_committed_tx_id = transaction_id
        self._clear_candidate()
        self._evo_state = "LIVE"
        
        if target_generation is not None:
            self.model_generation = target_generation
            self.global_generation = target_generation
        else:
            self.model_generation = getattr(self, 'model_generation', 1) + 1
            self.global_generation = getattr(self, 'global_generation', 1) + 1
        
        if hasattr(self, 'pending_plasticity') and self.pending_plasticity:
            for sig, b_idx in self.pending_plasticity:
                if 0 <= b_idx < self.num_nodes:
                    self.apply_plasticity(sig, b_idx)
            self.pending_plasticity.clear()
            
        return True

    def rollback_candidate(self, transaction_id: str = None, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        if not transaction_id:
            logger.error("[M9_Thalamus] Rollback Failed: Thiếu transaction_id!")
            return False
        if self._current_transaction_id is None and getattr(self, 'last_committed_tx_id', None) != transaction_id:
            logger.error(f"[M9_Thalamus] Rollback Failed: Không có giao dịch đang chờ khớp với {transaction_id}")
            return False
        if self._current_transaction_id is not None and self._current_transaction_id != transaction_id:
            logger.error(f"[M9_Thalamus] Rollback Failed: Lệch transaction_id (Kỳ vọng={self._current_transaction_id}, Nhận={transaction_id})")
            return False

        if getattr(self, '_parent_weights', None) is not None:
            old_w_param = self.weights
            self.weights = nn.Parameter(self._parent_weights.detach().clone())
            self.attention_to_route.data.copy_(self._parent_routing_matrix)
            self._apply_routing_constraints()
            self.num_nodes = self._parent_num_nodes
            self.topology_version = self._parent_topology_version
            
            self.step_count = self._parent_step_count
            self.dynamic_threshold = self._parent_dynamic_threshold
            self.last_input = self._parent_last_input
            self._pending_growth = self._parent_pending_growth
            self.routing_eligibility.copy_(self._parent_routing_eligibility)
            self.model_generation = self._parent_model_generation
            self.global_generation = self._parent_global_generation
            
            if self._parent_som_eligibility.shape == (self.num_nodes, self.input_dim):
                self.register_buffer('som_eligibility', self._parent_som_eligibility.clone())
            else:
                new_elig = torch.zeros(self.num_nodes, self.input_dim, device=self.weights.device)
                min_n = min(self._parent_som_eligibility.shape[0], self.num_nodes)
                new_elig[:min_n] = self._parent_som_eligibility[:min_n]
                self.register_buffer('som_eligibility', new_elig)

            self._verify_structural_integrity()
            self._sync_working_from_committed()
            self._sync_optimizer_after_mutation(optimizer, old_weights_param=old_w_param)

        if getattr(self, 'last_committed_tx_id', None) == transaction_id:
            self.last_committed_tx_id = None
        if hasattr(self, 'pending_plasticity'):
            self.pending_plasticity.clear()
            
        self._clear_candidate()
        self._evo_state = "LIVE"
        return True

    def _clear_candidate(self):
        self.candidate_state = None
        self._candidate_weights = None
        self._candidate_num_nodes = 0
        self._candidate_routing_matrix = None
        self._candidate_topology_version = 0
        self._candidate_routing_eligibility = None
        self._candidate_som_eligibility = None
        self._current_transaction_id = None

    def _compute_candidate_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        
        if self._candidate_weights is not None:
            h.update(self._candidate_weights.cpu().numpy().tobytes())
        else:
            h.update(self.weights.detach().cpu().numpy().tobytes())
            
        if self._candidate_routing_matrix is not None:
            h.update(self._candidate_routing_matrix.cpu().numpy().tobytes())
        else:
            h.update(self.attention_to_route.detach().cpu().numpy().tobytes())
            
        if getattr(self, '_candidate_routing_eligibility', None) is not None:
            h.update(self._candidate_routing_eligibility.cpu().numpy().tobytes())
            
        nodes = self._candidate_num_nodes if self._candidate_num_nodes > 0 else self.num_nodes
        topo_ver = self._candidate_topology_version if self._candidate_topology_version > 0 else self.topology_version
        h.update(str(nodes).encode())
        h.update(str(topo_ver).encode())
        return h.hexdigest()

    # =========================================================================
    # CHECKPOINT & VALIDATION
    # =========================================================================
    def get_topology_info(self) -> Dict:
        return {
            "name": "DynamicSOM",
            "input_dim": self.input_dim,
            "num_nodes": self.num_nodes,
            "topology_version": self.topology_version,
            "routing_matrix_shape": list(self.attention_to_route.shape),
            "dynamic_threshold": getattr(self, 'dynamic_threshold', 0.5),
            "model_generation": getattr(self, 'model_generation', 1),
            "step_count": getattr(self, 'step_count', 0)
        }

    @staticmethod
    def validate_topology_compatibility(current_model: 'DynamicSOM', loaded_state_dict: Dict) -> Tuple[bool, str]:
        loaded_weights = loaded_state_dict.get("weights")
        if loaded_weights is None: return False, "Missing 'weights'"
        if loaded_weights.dim() != 2 or loaded_weights.shape[1] != current_model.input_dim:
            return False, f"Weights Dimension Mismatch: {tuple(loaded_weights.shape)}"
        loaded_nodes = loaded_weights.shape[0]
        if loaded_nodes != current_model.num_nodes:
            return False, f"Node Count Mismatch: Current={current_model.num_nodes}, Loaded={loaded_nodes}"
        
        loaded_route = loaded_state_dict.get("attention_to_route")
        if loaded_route is not None and loaded_route.shape != current_model.attention_to_route.shape:
            return False, "Routing Matrix Shape Mismatch"
        loaded_som_elig = loaded_state_dict.get("som_eligibility")
        if loaded_som_elig is not None and loaded_som_elig.shape != (loaded_nodes, current_model.input_dim):
            return False, "SOM Eligibility Shape Mismatch"
            
        return True, "OK"

    def enforce_topology_from_checkpoint(self, loaded_state_dict: Dict, topology_meta: Optional[Dict] = None):
        loaded_weights = loaded_state_dict.get("weights")
        if loaded_weights is None or loaded_weights.dim() != 2 or loaded_weights.shape[1] != self.input_dim:
            raise ValueError("[M9_Thalamus] Checkpoint thiếu hoặc sai kích thước 'weights'!")
        target_nodes = int(loaded_weights.shape[0])
        if isinstance(topology_meta, dict) and "num_nodes" in topology_meta:
            if int(topology_meta["num_nodes"]) != target_nodes:
                raise ValueError(f"[M9_Thalamus] Lệch num_nodes trong metadata ({topology_meta['num_nodes']}) và weights ({target_nodes})!")

        backup_w = self.weights.detach().clone()
        backup_rm = self.attention_to_route.detach().clone()
        backup_se = self.som_eligibility.clone()
        backup_re = self.routing_eligibility.clone()
        backup_nodes = self.num_nodes
        backup_topo_ver = self.topology_version
        try:
            if target_nodes != self.num_nodes or self.weights.shape != loaded_weights.shape:
                self.weights = nn.Parameter(loaded_weights.clone()) 
                self.num_nodes = target_nodes
            if self.som_eligibility.shape != (target_nodes, self.input_dim):
                self.register_buffer('som_eligibility', torch.zeros(target_nodes, self.input_dim, device=self.weights.device))
            if "som_eligibility" in loaded_state_dict and loaded_state_dict["som_eligibility"].shape == (target_nodes, self.input_dim):
                self.som_eligibility.copy_(loaded_state_dict["som_eligibility"])
            
            loaded_route = loaded_state_dict.get("attention_to_route")
            if loaded_route is not None:
                self.attention_to_route = nn.Parameter(loaded_route.clone())
                self._apply_routing_constraints()
            
            if self.routing_eligibility.shape != (self.num_dest, self.num_modalities):
                self.register_buffer('routing_eligibility', torch.zeros(self.num_dest, self.num_modalities, device=self.weights.device))
            if "routing_eligibility" in loaded_state_dict and loaded_state_dict["routing_eligibility"].shape == (self.num_dest, self.num_modalities):
                self.routing_eligibility.copy_(loaded_state_dict["routing_eligibility"])
            if isinstance(topology_meta, dict) and "topology_version" in topology_meta:
                self.topology_version = int(topology_meta["topology_version"])
            self._verify_structural_integrity()
        except Exception as e:
            self.weights = nn.Parameter(backup_w)
            self.attention_to_route = nn.Parameter(backup_rm)
            self.register_buffer('som_eligibility', backup_se)
            self.register_buffer('routing_eligibility', backup_re)
            self.num_nodes = backup_nodes
            self.topology_version = backup_topo_ver
            raise RuntimeError(f"[M9_Thalamus] Khôi phục về trạng thái trước đó do lỗi dựng topology: {e}")

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, topology_meta: Optional[Dict] = None):
        backup_w = self.weights.detach().clone()
        backup_rm = self.attention_to_route.detach().clone()
        backup_se = self.som_eligibility.clone()
        backup_re = self.routing_eligibility.clone()
        backup_nodes = self.num_nodes
        backup_topo_ver = self.topology_version
        try:
            if "weights" in state_dict and isinstance(state_dict["weights"], torch.Tensor):
                self.enforce_topology_from_checkpoint(state_dict, topology_meta=topology_meta)
                if "som_eligibility" not in state_dict:
                    state_dict = dict(state_dict)
                    state_dict["som_eligibility"] = self.som_eligibility.clone()
                if "routing_eligibility" not in state_dict:
                    state_dict = dict(state_dict)
                    state_dict["routing_eligibility"] = self.routing_eligibility.clone()
            res = super().load_state_dict(state_dict, strict)
            if self.som_eligibility.shape != (self.num_nodes, self.input_dim):
                self.register_buffer('som_eligibility', torch.zeros(self.num_nodes, self.input_dim, device=self.weights.device))
            self._apply_routing_constraints()
            self._verify_structural_integrity()
            self._sync_working_from_committed()
            return res
        except Exception as e:
            self.weights = nn.Parameter(backup_w)
            self.attention_to_route = nn.Parameter(backup_rm)
            self.register_buffer('som_eligibility', backup_se)
            self.register_buffer('routing_eligibility', backup_re)
            self.num_nodes = backup_nodes
            self.topology_version = backup_topo_ver
            raise e

    def get_routing_payload(self, bmu_idx, attention_routing, salience, inhibition_level, destination_mask, raw_probs, topdown_attention):
        return {
            "neural_focus": [float(bmu_idx)],
            "destination_mask": destination_mask,
            "routing_probs": raw_probs,
            "salience": float(salience),
            "inhibition_level": float(inhibition_level),
            "attention_routing": attention_routing,
            "topdown_attention": topdown_attention,
            "modality": "routing",
            "num_nodes": self.num_nodes,
            "topology_version": self.topology_version
        }

if __name__ == "__main__":
    print("ðŸ§  [TEST] DynamicSOM - Adaptive Routing & 2-Phase Evolution")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicSOM(input_dim=128, initial_nodes=4).to(device)
    
    x = torch.randn(1, 128, device=device)
    bmu, dist = model.forward(x)
    print(f"BMU: {bmu.item()}, Dist: {dist[0, bmu].item():.4f}")
    
    attn = {k: 0.1 for k in model.modality_names}
    attn["vision_where"] = 0.8
    attn["semantics"] = 0.7
    
    mask, probs, topdown = model.get_dynamic_routing(
        bmu.item(), attn, salience=0.9, inhibition_level=0.1,
        reward_signal=0.5, novelty=0.2, task_success=1.0
    )
    print(f"Mask: {mask}")
    
    print("\n--- 2-Phase Evolution Test ---")
    far_input = torch.randn(1, 128, device=device) * 100 
    bmu_grow, changed = model.update_and_grow(far_input, current_tick=10, current_episode="TEST")
    
    prep = model.prepare_candidate("tx_test_123", target_tick=10, target_episode="TEST")
    assert model.commit_candidate("tx_test_123"), "Commit phải thành công khi đúng transaction_id"
    assert not model.rollback_candidate("tx_wrong_id"), "Rollback phải trả về False khi sai transaction_id"
    print(f"Committed. Live Nodes: {model.num_nodes}, Gen: {model.model_generation}")
    
    print("\nâœ… DynamicSOM Tests PASSED.")
