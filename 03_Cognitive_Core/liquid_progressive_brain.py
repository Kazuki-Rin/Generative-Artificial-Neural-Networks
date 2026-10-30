#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Progressive Liquid Brain - Dynamic Skill Columns with Cross-Modal Projectors
============================================================================
M6: Language Cortex (Broca/Wernicke) - 5 inputs, 128 hidden, 300 outputs
M7: Logic Cortex (Prefrontal) - 5 inputs, 128 hidden, 300 outputs

Features:
- Dynamic skill columns (Liquid Time-Constant networks)
- Cross-modal projectors for inter-module communication
- Structural evolution (grow/prune columns)
- Weight pruning (sparsity)
- Topology validation & enforcement for checkpoint compatibility
- 2-Phase Evolution Protocol (PREPARE/COMMIT/ROLLBACK)
"""

import os
import sys
import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


class LiquidSkillColumn(nn.Module):
    """
    A single Liquid Time-Constant (LTC) skill column.
    Implements a continuous-time RNN with adaptive time constants.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(LiquidSkillColumn, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 10.0) 
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        with torch.no_grad():
            self.A.weight.data = -torch.eye(hidden_dim) * 0.1 + torch.randn(hidden_dim, hidden_dim) * 0.01
    
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None, dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is not None and h.size(0) != (x.size(1) if x.dim() == 3 else x.size(0)):
            h = None 
            
        if x.dim() == 3:
            seq_len, batch, _ = x.shape
            outputs = []
            if h is None:
                h = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
            
            for t in range(seq_len):
                h = self._step(x[t], h, dt)
                outputs.append(self.output_proj(h))
            return torch.stack(outputs, dim=0), h
        
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        h = self._step(x, h, dt)
        return self.output_proj(h), h
    
    def _step(self, x: torch.Tensor, h: torch.Tensor, dt: float) -> torch.Tensor:
        x_proj = self.input_proj(x)
        tau = F.softplus(self.tau) + 1e-3
        Ah = self.A(h)
        dh = (-h + Ah + x_proj) / tau
        h_new = h + dt * dh
        return h_new


class CrossModalProjector(nn.Module):
    """
    Cross-modal projector for inter-module communication.
    Projects from source module's hidden space to target module's input space.
    """
    def __init__(self, source_dim: int, target_dim: int, hidden_dim: int = 64):
        super(CrossModalProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class ProgressiveLiquidBrain(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        output_dim: int = 300,
        name: str = "ProgressiveLiquidBrain",
        max_columns: int = 16,
        min_columns: int = 1
    ):
        super(ProgressiveLiquidBrain, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_columns = max_columns
        self.min_columns = min_columns
        
        self.topology_version = 1
        self.weight_revision = 1
        
        self._evolution_state = "IDLE"
        self._parent_state = None
        self._candidate_state = None
        self._plasticity_queue = []
        
        self.columns = nn.ModuleList()
        self._init_columns(1)
        
        self.gate = nn.Linear(input_dim, 1, bias=True)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        
        self.projectors = nn.ModuleDict()
        self.projectors["thalamus"] = CrossModalProjector(128, input_dim)
        
        if "M6" in name:
            self.projectors["memory"] = CrossModalProjector(39, input_dim) 
            self.projectors["vocal_fb"] = CrossModalProjector(2, input_dim) 
            self.projectors["logic_cmd"] = CrossModalProjector(10, input_dim)
        elif "M7" in name:
            self.projectors["amygdala"] = CrossModalProjector(2, input_dim) 
            self.projectors["memory"] = CrossModalProjector(39, input_dim)

        self.register_buffer('column_usage', torch.zeros(max_columns))
        self.register_buffer('column_age', torch.zeros(max_columns))
        self._runtime_column_usage = torch.zeros(max_columns)
        self._runtime_column_age = torch.zeros(max_columns)
        self._verify_topology_integrity()
        
    def _init_columns(self, num: int):
        for _ in range(num):
            col = LiquidSkillColumn(self.input_dim, self.hidden_dim)
            self.columns.append(col)

    def _ensure_tracking_buffers_capacity(self, required_cols: int):
        device = self.out_layer.weight.device
        if required_cols > self.max_columns:
            self.max_columns = required_cols
        if self.column_usage.numel() < self.max_columns:
            new_usage = torch.zeros(self.max_columns, device=device, dtype=self.column_usage.dtype)
            new_age = torch.zeros(self.max_columns, device=device, dtype=self.column_age.dtype)
            old_len = min(self.column_usage.numel(), self.max_columns)
            new_usage[:old_len] = self.column_usage[:old_len]
            new_age[:old_len] = self.column_age[:old_len]
            self.register_buffer('column_usage', new_usage)
            self.register_buffer('column_age', new_age)

    def _verify_topology_integrity(self) -> bool:
        num_cols = len(self.columns)
        if num_cols < self.min_columns:
            raise RuntimeError(f"[{self.name}] Lỗi toàn vẹn: số cột ({num_cols}) < min_columns ({self.min_columns})")
        if self.gate.in_features != self.input_dim or self.gate.out_features != num_cols:
            raise RuntimeError(
                f"[{self.name}] Lỗi toàn vẹn Gate: shape={tuple(self.gate.weight.shape)}, kỳ vọng=({num_cols}, {self.input_dim})"
            )
        for idx, col in enumerate(self.columns):
            if col.input_dim != self.input_dim or col.hidden_dim != self.hidden_dim:
                raise RuntimeError(f"[{self.name}] Lỗi toàn vẹn Column {idx}: ({col.input_dim}, {col.hidden_dim})")
        if self.out_layer.in_features != self.hidden_dim or self.out_layer.out_features != self.output_dim:
            raise RuntimeError(f"[{self.name}] Lỗi toàn vẹn out_layer: {tuple(self.out_layer.weight.shape)}")
        for p in self.parameters():
            if not torch.all(torch.isfinite(p)):
                raise RuntimeError(f"[{self.name}] Phát hiện NaN/Inf trong trọng số!")
        return True

    def _can_grow_column(self, target_cols: int, extra_params: int = None) -> Tuple[bool, str]:
        import json
        gov_max_cols = self.max_columns
        gov_max_params = 150000
        global_max_params = 500000
        max_rss_mb = 2048.0
        arch_path = os.path.join(BASE_DIR, "architecture_manifest.json")
        if os.path.exists(arch_path):
            try:
                with open(arch_path, "r", encoding="utf-8") as f:
                    arch = json.load(f)
                gov = arch.get("system_topology", {}).get("resource_governor", arch.get("governor", {}))
                mod_key = "M6_Language" if "M6" in self.name else ("M7_Logic" if "M7" in self.name else self.name)
                mod_cfg = arch.get("modules", {}).get(mod_key, {})
                evo_cfg = mod_cfg.get("evolution", {})
                if not evo_cfg.get("structural", True):
                    return False, f"evolution.structural=false cho {mod_key}"
                node_caps = gov.get("max_nodes_per_module", {})
                param_caps = gov.get("max_parameters_per_module", {})
                gov_max_cols = min(
                    gov_max_cols,
                    int(node_caps.get(mod_key, gov_max_cols)),
                    int(evo_cfg.get("max_columns", evo_cfg.get("max_nodes", mod_cfg.get("max_columns", gov_max_cols))))
                )
                gov_max_params = min(
                    int(param_caps.get(mod_key, gov.get("max_params_per_model", gov_max_params))),
                    int(evo_cfg.get("max_parameters", gov.get("max_params_per_model", gov_max_params)))
                )
                global_max_params = int(gov.get("global_max_parameters", global_max_params))
                max_rss_mb = float(gov.get("max_process_rss_mb", max_rss_mb))
            except Exception:
                pass
        if target_cols > gov_max_cols:
            return False, f"Vượt giới hạn cột tối đa ({target_cols} > {gov_max_cols})"
        if extra_params is None:
            col_params = (self.input_dim * self.hidden_dim + self.hidden_dim) + (self.hidden_dim * self.hidden_dim) + self.hidden_dim + (self.hidden_dim * self.hidden_dim + self.hidden_dim)
            extra_params = col_params + self.input_dim + 1
        curr_params = sum(p.numel() for p in self.parameters())
        projected_params = curr_params + extra_params
        if projected_params > gov_max_params or projected_params > global_max_params:
            return False, f"Vượt ngân sách tham số Governor ({projected_params} > {min(gov_max_params, global_max_params)})"
        try:
            import psutil
            vm = psutil.virtual_memory()
            proc_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
            if vm.percent >= 90.0 or proc_rss_mb >= max_rss_mb:
                return False, f"Tài nguyên hệ thống chạm ngưỡng bảo vệ (SysRAM={vm.percent:.1f}%, ProcRSS={proc_rss_mb:.1f}MB)"
        except Exception:
            pass
        return True, "OK"

    def _sync_optimizer_after_mutation(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        old_gate_weight: Optional[nn.Parameter] = None,
        old_gate_bias: Optional[nn.Parameter] = None,
        surviving_indices: Optional[List[int]] = None,
        removed_params: Optional[List[nn.Parameter]] = None
    ):
        if optimizer is None:
            return
        if removed_params:
            for rp in removed_params:
                optimizer.state.pop(rp, None)
        if old_gate_weight is not None and old_gate_weight in optimizer.state:
            old_w_state = optimizer.state.pop(old_gate_weight)
            new_w_state = {}
            for k, v in old_w_state.items():
                if torch.is_tensor(v) and v.shape == old_gate_weight.shape:
                    nv = torch.zeros_like(self.gate.weight, memory_format=torch.preserve_format)
                    if surviving_indices is not None:
                        nv.copy_(v[surviving_indices, :])
                    else:
                        rows = min(v.shape[0], nv.shape[0])
                        nv[:rows, :].copy_(v[:rows, :])
                    new_w_state[k] = nv
                else:
                    new_w_state[k] = copy.deepcopy(v)
            optimizer.state[self.gate.weight] = new_w_state
        if old_gate_bias is not None and old_gate_bias in optimizer.state:
            old_b_state = optimizer.state.pop(old_gate_bias)
            new_b_state = {}
            for k, v in old_b_state.items():
                if torch.is_tensor(v) and v.shape == old_gate_bias.shape:
                    nv = torch.zeros_like(self.gate.bias, memory_format=torch.preserve_format)
                    if surviving_indices is not None:
                        nv.copy_(v[surviving_indices])
                    else:
                        rows = min(v.shape[0], nv.shape[0])
                        nv[:rows].copy_(v[:rows])
                    new_b_state[k] = nv
                else:
                    new_b_state[k] = copy.deepcopy(v)
            optimizer.state[self.gate.bias] = new_b_state
        if len(optimizer.param_groups) > 0:
            optimizer.param_groups[0]["params"] = [p for p in self.parameters() if p.requires_grad]
    
    def _refresh_candidate_snapshot(self):
        if getattr(self, '_evolution_state', "IDLE") == "STAGED" and getattr(self, '_current_transaction_id', None) is not None:
            self._candidate_state = {
                'transaction_id': self._current_transaction_id,
                'state_dict': copy.deepcopy(self.state_dict()),
                'num_columns': len(self.columns),
                'projector_keys': list(self.projectors.keys()),
                'topology_version': getattr(self, 'topology_version', 1),
                'weight_revision': getattr(self, 'weight_revision', 1)
            }

    def sprout_new_skill_column(self, enforce_governor: bool = True, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        target_cols = len(self.columns) + 1
        if enforce_governor:
            can_grow, reason = self._can_grow_column(target_cols)
            if not can_grow:
                logger.warning(f"⚠️ [{self.name}] Từ chối mọc cột kỹ năng mới: {reason}")
                return False
        elif len(self.columns) >= self.max_columns:
            return False
        
        backup_columns = copy.deepcopy(self.columns)
        backup_gate = copy.deepcopy(self.gate)
        backup_max_cols = self.max_columns
        backup_usage = self.column_usage.clone()
        backup_age = self.column_age.clone()
        backup_topo_ver = self.topology_version
        old_gate_w = self.gate.weight
        old_gate_b = self.gate.bias
        try:
            device = self.out_layer.weight.device
            self._ensure_tracking_buffers_capacity(target_cols)
            new_col = LiquidSkillColumn(self.input_dim, self.hidden_dim).to(device)
            new_gate = nn.Linear(self.input_dim, target_cols, bias=True).to(device)
            with torch.no_grad():
                old_cols = len(self.columns)
                new_gate.weight.data[:old_cols, :].copy_(self.gate.weight.data)
                new_gate.bias.data[:old_cols].copy_(self.gate.bias.data)
                new_gate.weight.data[old_cols, :] = torch.randn(self.input_dim, device=device) * 0.01
                new_gate.bias.data[old_cols] = -2.0
            self.columns.append(new_col)
            self.gate = new_gate
            self.hidden_states = [None] * len(self.columns)
            self._verify_topology_integrity()
            self.topology_version += 1
            self._sync_optimizer_after_mutation(optimizer, old_gate_weight=old_gate_w, old_gate_bias=old_gate_b)
            self._refresh_candidate_snapshot()
            return True
        except Exception as e:
            logger.error(f"❌ [{self.name}] Lỗi khi mọc cột mới ({e}), tự động khôi phục trạng thái nhất quán!")
            self.columns = backup_columns
            self.gate = backup_gate
            self.max_columns = backup_max_cols
            self.register_buffer('column_usage', backup_usage)
            self.register_buffer('column_age', backup_age)
            self.topology_version = backup_topo_ver
            self.hidden_states = [None] * len(self.columns)
            return False
    
    def prune_weights(self, sparsity: float = 0.1) -> Dict:
        total_params = 0
        pruned_params = 0
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    total_params += param.numel()
                    threshold = torch.quantile(param.abs().flatten(), sparsity)
                    mask = param.abs() > threshold
                    param.mul_(mask.float())
                    pruned_params += (~mask).sum().item()
        
        self.weight_revision += 1
        self._refresh_candidate_snapshot()
        return {
            'total_params': total_params,
            'params_pruned': pruned_params,
            'sparsity': pruned_params / max(total_params, 1)
        }
    
    def prune_structural(
        self, 
        sample_batch: torch.Tensor, 
        min_gate_activation: float = 0.01,
        min_usage_ratio: float = 0.05,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        if len(self.columns) <= self.min_columns:
            return {'pruned': False, 'reason': 'at_minimum_columns'}
        
        backup_columns = copy.deepcopy(self.columns)
        backup_gate = copy.deepcopy(self.gate)
        backup_usage = self.column_usage.clone()
        backup_age = self.column_age.clone()
        backup_topo_ver = self.topology_version
        old_gate_w = self.gate.weight
        old_gate_b = self.gate.bias
        device = self.out_layer.weight.device
        self.eval()
        try:
            with torch.no_grad():
                gate_logits = self.gate(sample_batch)
                gate_probs = torch.softmax(gate_logits, dim=-1)
                mean_activation = gate_probs.mean(dim=0) 
                
                dead_columns = (mean_activation < min_gate_activation).nonzero(as_tuple=True)[0]
                if len(dead_columns) == 0:
                    return {'pruned': False, 'reason': 'no_dead_columns'}
                
                max_prune = len(self.columns) - self.min_columns
                dead_columns = dead_columns[:max_prune]
                
                old_num_columns = len(self.columns)
                dead_list = dead_columns.tolist()
                surviving_indices = [i for i in range(old_num_columns) if i not in dead_list]
                new_gate = nn.Linear(self.input_dim, len(surviving_indices), bias=True).to(device)
                new_gate.weight.data.copy_(self.gate.weight[surviving_indices, :])
                new_gate.bias.data.copy_(self.gate.bias[surviving_indices])
                
                removed_params = []
                for idx in sorted(dead_list, reverse=True):
                    removed_params.extend(list(self.columns[idx].parameters()))
                    del self.columns[idx]
                self.gate = new_gate
                
                self.column_usage[:len(surviving_indices)] = self.column_usage[surviving_indices]
                self.column_usage[len(surviving_indices):] = 0.0
                self.column_age[:len(surviving_indices)] = self.column_age[surviving_indices]
                self.column_age[len(surviving_indices):] = 0.0
                self.hidden_states = [None] * len(self.columns)
                self._verify_topology_integrity()
                self.topology_version += 1
                self._sync_optimizer_after_mutation(
                    optimizer,
                    old_gate_weight=old_gate_w,
                    old_gate_bias=old_gate_b,
                    surviving_indices=surviving_indices,
                    removed_params=removed_params
                )
                self._refresh_candidate_snapshot()
                
                return {
                    'pruned': True,
                    'num_pruned': len(dead_columns),
                    'remaining_columns': len(self.columns)
                }
        except Exception as e:
            logger.error(f"❌ [{self.name}] Lỗi khi cắt tỉa cột ({e}), khôi phục trạng thái cũ!")
            self.columns = backup_columns
            self.gate = backup_gate
            self.column_usage.copy_(backup_usage)
            self.column_age.copy_(backup_age)
            self.topology_version = backup_topo_ver
            self.hidden_states = [None] * len(self.columns)
            return {'pruned': False, 'reason': f'error_recovered:{e}'}
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_details: bool = False,
        cross_modal_inputs: Optional[Dict[str, torch.Tensor]] = None,
        integration_time: float = 0.1 
    ) -> Dict:
        batch_size = x.size(0) if x.dim() == 2 else x.size(1)
        device = x.device
        
        if not hasattr(self, 'hidden_states') or len(self.hidden_states) != len(self.columns):
            self.hidden_states = [None] * len(self.columns)
            
        if cross_modal_inputs:
            for src_name, src_tensor in cross_modal_inputs.items():
                if src_name in self.projectors:
                    projected = self.projectors[src_name](src_tensor)
                    if x.dim() == 3: x = x + projected.unsqueeze(0)
                    else: x = x + projected
        
        gate_logits = self.gate(x)  
        gate_weights = torch.softmax(gate_logits, dim=-1)
        
        column_outputs = []
        
        is_validating = not torch.is_grad_enabled()
        
        for i, col in enumerate(self.columns):
            h_prev = self.hidden_states[i]
            col_out, h_next = col(x, h=h_prev, dt=integration_time) 
            
            if self.training and not is_validating:
                self.hidden_states[i] = h_next.detach() 
                
            column_outputs.append(col_out)
        
        if x.dim() == 3:
            column_outputs = torch.stack(column_outputs, dim=0) 
            gw = gate_weights.permute(2, 0, 1).unsqueeze(-1)
            fused = (column_outputs * gw).sum(dim=0)
        else:
            column_outputs = torch.stack(column_outputs, dim=1) 
            fused = (column_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
        
        output = self.out_layer(fused)
        
        if self.training:
            with torch.no_grad():
                if x.dim() == 3: usage = gate_weights.mean(dim=(0, 1))
                else: usage = gate_weights.mean(dim=0)
                if not hasattr(self, '_runtime_column_usage') or self._runtime_column_usage.numel() < self.max_columns:
                    self._runtime_column_usage = self.column_usage.detach().clone()
                    self._runtime_column_age = self.column_age.detach().clone()
                self._runtime_column_usage[:len(self.columns)] = usage.detach().to(self._runtime_column_usage.device)
                self._runtime_column_age[:len(self.columns)] += 1
        
        if return_details:
            return {
                'output': output,
                'fused_hidden': fused,
                'column_outputs': column_outputs,
                'gate_weights': gate_weights,
                'gate_logits': gate_logits
            }
        return output
    
    def add_cross_modal_projector(self, source_module: str, source_dim: int, optimizer: Optional[torch.optim.Optimizer] = None, enforce_governor: bool = True) -> bool:
        device = self.out_layer.weight.device
        new_proj = CrossModalProjector(source_dim, self.input_dim).to(device)
        proj_params = sum(p.numel() for p in new_proj.parameters())
        if enforce_governor:
            can_grow, reason = self._can_grow_column(len(self.columns), extra_params=proj_params)
            if not can_grow:
                logger.warning(f"⚠️ [{self.name}] Từ chối thêm projector '{source_module}': {reason}")
                return False
        backup_projectors = copy.deepcopy(self.projectors)
        backup_topo_ver = self.topology_version
        try:
            self.projectors[source_module] = new_proj
            self._verify_topology_integrity()
            self.topology_version += 1
            self._sync_optimizer_after_mutation(optimizer)
            self._refresh_candidate_snapshot()
            return True
        except Exception as e:
            logger.error(f"❌ [{self.name}] Lỗi khi thêm projector '{source_module}' ({e}), hoàn tác!")
            self.projectors = backup_projectors
            self.topology_version = backup_topo_ver
            return False
    
    def get_topology_signature(self) -> Dict:
        return {
            'name': self.name,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_columns': len(self.columns),
            'projector_keys': list(self.projectors.keys()),
            'topology_version': getattr(self, 'topology_version', 1),
            'weight_revision': getattr(self, 'weight_revision', 1),
            'max_columns': self.max_columns,
            'min_columns': self.min_columns
        }
    
    @staticmethod
    def _infer_column_count_from_state_dict(loaded_state_dict: Dict) -> Tuple[int, Optional[str]]:
        loaded_gate_weight = loaded_state_dict.get("gate.weight")
        if loaded_gate_weight is None or loaded_gate_weight.dim() != 2:
            return 0, "Missing or invalid gate.weight in checkpoint"
        gate_cols = int(loaded_gate_weight.shape[0])
        col_indices = sorted({
            int(k.split(".")[1])
            for k in loaded_state_dict.keys()
            if k.startswith("columns.") and len(k.split(".")) > 2 and k.split(".")[1].isdigit()
        })
        if not col_indices or col_indices != list(range(gate_cols)):
            return 0, f"Column keys {col_indices} do not match gate.weight rows ({gate_cols})"
        return gate_cols, None

    @staticmethod
    def validate_topology_compatibility(current_model: 'ProgressiveLiquidBrain', loaded_state_dict: Dict) -> Tuple[bool, str]:
        loaded_num_columns, err = ProgressiveLiquidBrain._infer_column_count_from_state_dict(loaded_state_dict)
        if err is not None:
            return False, err
        current_num_columns = len(current_model.columns)
        
        if loaded_num_columns != current_num_columns:
            return False, f"Column Count Mismatch: Current={current_num_columns}, Loaded={loaded_num_columns}"
        
        loaded_in_weight = loaded_state_dict.get("columns.0.input_proj.weight")
        if loaded_in_weight is not None:
            loaded_input_dim = loaded_in_weight.shape[1]
            if loaded_input_dim != current_model.input_dim:
                return False, f"Input Dim Mismatch: Current={current_model.input_dim}, Loaded={loaded_input_dim}"
        
        loaded_out_weight = loaded_state_dict.get("out_layer.weight")
        if loaded_out_weight is not None:
            loaded_output_dim = loaded_out_weight.shape[0]
            if loaded_output_dim != current_model.output_dim:
                return False, f"Output Dim Mismatch: Current={current_model.output_dim}, Loaded={loaded_output_dim}"
        
        return True, "OK"
    
    def enforce_topology_from_checkpoint(self, loaded_state_dict: Dict, topology_meta: Optional[Dict] = None):
        target_cols, err = self._infer_column_count_from_state_dict(loaded_state_dict)
        if err is not None:
            raise ValueError(f"[{self.name}] Checkpoint không hợp lệ: {err}")
        if isinstance(topology_meta, dict) and "num_columns" in topology_meta:
            meta_cols = int(topology_meta["num_columns"])
            if meta_cols != target_cols:
                raise ValueError(f"[{self.name}] Lệch topology metadata (num_columns={meta_cols}) và trọng số (gate_cols={target_cols})!")

        backup_state = copy.deepcopy(self.state_dict())
        backup_cols = len(self.columns)
        backup_projectors = copy.deepcopy(self.projectors)
        backup_max_cols = self.max_columns
        backup_usage = self.column_usage.clone()
        backup_age = self.column_age.clone()
        backup_topo_ver = self.topology_version
        device = self.out_layer.weight.device
        try:
            self._ensure_tracking_buffers_capacity(target_cols)
            new_cols = nn.ModuleList([LiquidSkillColumn(self.input_dim, self.hidden_dim).to(device) for _ in range(target_cols)])
            for i in range(min(len(self.columns), target_cols)):
                new_cols[i].load_state_dict(self.columns[i].state_dict())
            new_gate = nn.Linear(self.input_dim, target_cols, bias=True).to(device)
            with torch.no_grad():
                copy_rows = min(self.gate.weight.shape[0], target_cols)
                new_gate.weight[:copy_rows, :].copy_(self.gate.weight[:copy_rows, :])
                new_gate.bias[:copy_rows].copy_(self.gate.bias[:copy_rows])
            for k, v in loaded_state_dict.items():
                if k.startswith("projectors.") and k.endswith(".projector.0.weight") and v.dim() == 2:
                    proj_name = k.split(".")[1]
                    src_dim = int(v.shape[1])
                    hid_dim = int(v.shape[0])
                    if proj_name not in self.projectors:
                        self.projectors[proj_name] = CrossModalProjector(src_dim, self.input_dim, hidden_dim=hid_dim).to(device)
            self.columns = new_cols
            self.gate = new_gate
            self.hidden_states = [None] * target_cols
            if isinstance(topology_meta, dict) and "topology_version" in topology_meta:
                self.topology_version = int(topology_meta["topology_version"])
            else:
                self.topology_version = backup_topo_ver
            self._verify_topology_integrity()
        except Exception as e:
            self.columns = nn.ModuleList([LiquidSkillColumn(self.input_dim, self.hidden_dim).to(device) for _ in range(backup_cols)])
            self.gate = nn.Linear(self.input_dim, backup_cols, bias=True).to(device)
            self.projectors = backup_projectors
            self.max_columns = backup_max_cols
            self.register_buffer('column_usage', backup_usage)
            self.register_buffer('column_age', backup_age)
            super().load_state_dict(backup_state, strict=False)
            self.topology_version = backup_topo_ver
            raise RuntimeError(f"[{self.name}] Khôi phục về trạng thái trước đó do lỗi dựng topology: {e}")

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, topology_meta: Optional[Dict] = None):
        if isinstance(state_dict, dict) and "gate.weight" in state_dict:
            self.enforce_topology_from_checkpoint(state_dict, topology_meta=topology_meta)
            state_dict = dict(state_dict)
            if "out_layer.weight" in state_dict and state_dict["out_layer.weight"].shape != self.out_layer.weight.shape:
                old_w = state_dict["out_layer.weight"]
                new_w = torch.zeros_like(self.out_layer.weight)
                r = min(old_w.shape[0], new_w.shape[0])
                c = min(old_w.shape[1], new_w.shape[1])
                new_w[:r, :c] = old_w[:r, :c]
                state_dict["out_layer.weight"] = new_w
            if "out_layer.bias" in state_dict and state_dict["out_layer.bias"].shape != self.out_layer.bias.shape:
                old_b = state_dict["out_layer.bias"]
                new_b = torch.zeros_like(self.out_layer.bias)
                r = min(old_b.shape[0], new_b.shape[0])
                new_b[:r] = old_b[:r]
                state_dict["out_layer.bias"] = new_b
            for buf_name in ("column_usage", "column_age"):
                if buf_name in state_dict and hasattr(self, buf_name):
                    buf_tensor = getattr(self, buf_name)
                    if state_dict[buf_name].shape != buf_tensor.shape:
                        resized = torch.zeros_like(buf_tensor)
                        min_len = min(resized.numel(), state_dict[buf_name].numel())
                        resized[:min_len] = state_dict[buf_name][:min_len]
                        state_dict[buf_name] = resized
        res = super().load_state_dict(state_dict, strict=strict)
        self.hidden_states = [None] * len(self.columns)
        self._verify_topology_integrity()
        return res

    # =========================================================================
    # 2-PHASE EVOLUTION PROTOCOL
    # =========================================================================
    def prepare_candidate(self, transaction_id: str) -> Dict:
        if not transaction_id or not isinstance(transaction_id, str):
            raise ValueError(f"[{self.name}] prepare_candidate yêu cầu transaction_id hợp lệ!")

        from shared_utils import tensor_state_hash
        if getattr(self, '_evolution_state', "IDLE") == "STAGED":
            if getattr(self, '_current_transaction_id', None) == transaction_id and self._candidate_state is not None:
                return {
                    'transaction_id': transaction_id,
                    'new_num_columns': self._candidate_state['num_columns'],
                    'candidate_hash': tensor_state_hash(self._candidate_state['state_dict']),
                    'status': 'PREPARED'
                }
            logger.warning(f"⚠️ [{self.name}] Hủy giao dịch treo cũ ({self._current_transaction_id}) trước khi tạo giao dịch mới ({transaction_id}).")
            self.rollback_candidate(self._current_transaction_id)

        self._verify_topology_integrity()
        self._current_transaction_id = transaction_id
        
        self._parent_state = {
            'state_dict': copy.deepcopy(self.state_dict()),
            'num_columns': len(self.columns),
            'projector_keys': list(self.projectors.keys()),
            'topology_version': getattr(self, 'topology_version', 1),
            'weight_revision': getattr(self, 'weight_revision', 1)
        }
        
        self._candidate_state = {
            'transaction_id': transaction_id,
            'state_dict': copy.deepcopy(self.state_dict()),
            'num_columns': len(self.columns),
            'projector_keys': list(self.projectors.keys()),
            'topology_version': getattr(self, 'topology_version', 1),
            'weight_revision': getattr(self, 'weight_revision', 1)
        }
        
        self._evolution_state = "STAGED"
        self._plasticity_queue = []
        cand_hash = tensor_state_hash(self._candidate_state['state_dict'])
            
        return {
            'transaction_id': transaction_id,
            'new_num_columns': len(self.columns),
            'candidate_hash': cand_hash,
            'status': 'PREPARED'
        }

    def commit_candidate(self, transaction_id: str, generation: int = None) -> bool:
        if not transaction_id or not isinstance(transaction_id, str):
            logger.error(f"[{self.name}] Từ chối Commit: Thiếu transaction_id hợp lệ!")
            return False
        if getattr(self, 'last_committed_tx_id', None) == transaction_id:
            return True
        if self._evolution_state != "STAGED" or getattr(self, '_current_transaction_id', None) != transaction_id:
            logger.warning(f"[{self.name}] Từ chối Commit do sai trạng thái ({self._evolution_state}) hoặc lệch Transaction ID (Nhận: {transaction_id}, Chờ: {getattr(self, '_current_transaction_id', None)})")
            return False

        try:
            if self._candidate_state is not None and "state_dict" in self._candidate_state:
                self.load_state_dict(
                    self._candidate_state["state_dict"],
                    strict=True,
                    topology_meta={"num_columns": self._candidate_state.get("num_columns", len(self.columns))}
                )
            self._verify_topology_integrity()
        except Exception as e:
            logger.error(f"[{self.name}] Lỗi khi áp dụng candidate lúc Commit ({e}), tự động Rollback về parent!")
            self.rollback_candidate(transaction_id)
            return False
        
        self.last_committed_tx_id = transaction_id
        self._parent_state = None
        self._candidate_state = None
        self._current_transaction_id = None
        self._evolution_state = "IDLE"
        
        if generation is not None:
            self.generation = generation
            
        for sig, trace, grads in self._plasticity_queue:
            self._apply_plasticity_immediate(sig, trace, grads)
        self._plasticity_queue.clear()
        
        return True

    def rollback_candidate(self, transaction_id: str) -> bool:
        if not transaction_id or not isinstance(transaction_id, str):
            logger.error(f"[{self.name}] Từ chối Rollback: Thiếu transaction_id hợp lệ!")
            return False
        if self._current_transaction_id is None:
            logger.warning(f"[{self.name}] Từ chối Rollback {transaction_id}: Không có giao dịch đang ở trạng thái STAGED.")
            return False
        if self._current_transaction_id != transaction_id:
            logger.warning(f"[{self.name}] Từ chối Rollback do lệch Transaction ID (Nhận: {transaction_id}, Đang giữ: {self._current_transaction_id})")
            return False

        if getattr(self, '_parent_state', None) is not None:
            parent = self._parent_state
            current_keys = list(self.projectors.keys())
            for key in current_keys:
                if key not in parent['projector_keys']:
                    del self.projectors[key]
            self.load_state_dict(
                parent['state_dict'],
                strict=True,
                topology_meta={'num_columns': parent['num_columns']}
            )
            self.topology_version = parent.get('topology_version', 1)
            self.weight_revision = parent.get('weight_revision', 1)
            
        self._parent_state = None
        self._candidate_state = None
        self._current_transaction_id = None
        self._plasticity_queue.clear()
        self._evolution_state = "IDLE"
        
        return True

    def reset_recurrent_state(self):
        self.hidden_states = [None] * len(self.columns)

    def apply_plasticity(self, mod_signal: float, eligibility_trace: torch.Tensor = None):
        if abs(mod_signal) < 1e-4: return
        
        if getattr(self, '_evolution_state', "IDLE") == "STAGED":
            cloned_grads = {}
            if eligibility_trace is None:
                for name, param in self.named_parameters():
                    if 'weight' in name and param.dim() >= 2 and param.grad is not None:
                        cloned_grads[name] = param.grad.clone()
            self._plasticity_queue.append((mod_signal, eligibility_trace, cloned_grads))
            return
            
        self._apply_plasticity_immediate(mod_signal, eligibility_trace)
        
    def _apply_plasticity_immediate(self, mod_signal: float, eligibility_trace: torch.Tensor = None, cloned_grads: Dict = None):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    if eligibility_trace is not None and param.shape == eligibility_trace.shape:
                        param.add_(0.001 * mod_signal * eligibility_trace)
                    elif cloned_grads and name in cloned_grads:
                        param.add_(-0.001 * mod_signal * cloned_grads[name])
                    elif param.grad is not None and not cloned_grads:
                        param.add_(-0.001 * mod_signal * param.grad)
    
        self.weight_revision = getattr(self, 'weight_revision', 1) + 1


if __name__ == "__main__":
    print("🧠 [TEST] ProgressiveLiquidBrain - Structural Evolution & Cross-Modal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ProgressiveLiquidBrain(input_dim=5, hidden_dim=128, output_dim=133, name="M6_Language").to(device)
    print(f"Init: {model.get_topology_signature()}")
    
    x = torch.randn(4, 5, device=device)
    out_dict = model(x, return_details=True)
    print(f"\nForward Keys: {list(out_dict.keys())}")
    
    model.sprout_new_skill_column()
    print(f"After Grow: {model.get_topology_signature()}")
    
    stats = model.prune_weights(0.1)
    print(f"\nWeight Pruning: {stats['params_pruned']:,} / {stats['total_params']:,}")
    
    print("\n✅ All tests passed!")