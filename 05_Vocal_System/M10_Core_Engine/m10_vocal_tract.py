#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
from typing import Dict
import torch
import torch.nn as nn
from torchdiffeq import odeint

# --- THÊM KHỐI NÀY ĐỂ ĐỊNH TUYẾN THƯ MỤC GỐC ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -----------------------------------------------

# Import shared utilities to break circular imports
from shared_utils import BASE_DIR, CheckpointManager

class VocalTractODE(nn.Module):
    """
    [BẢN VÁ LỖI VI.7]: Mạng Neural ODE mô phỏng Thanh quản.
    Đã bổ sung thuộc tính hidden_dim và khả năng tự bẻ khung xương khi Load Checkpoint.
    """
    def __init__(self, motor_dim=3, physical_dim=2, hidden_dim=16):
        super(VocalTractODE, self).__init__()
        self.motor_dim = motor_dim
        self.physical_dim = physical_dim
        self.hidden_dim = hidden_dim

        self.topology_version = 1
        self.weight_revision = 1
        self.generation = 1 

        self.motor_intent = torch.zeros(1, motor_dim)

        self.dynamics = nn.Sequential(
            nn.Linear(physical_dim + motor_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, physical_dim)
        )

        self._evo_state = "LIVE"
        self._staged_tx_id = None
        self._candidate_state = None
        self._parent_state = None

    def _ode_func(self, t, state):
        batch_size = state.shape[0]
        forcing = self.motor_intent.to(state.device).expand(batch_size, -1)
        fused_input = torch.cat([state, forcing], dim=-1)
        raw_output = self.dynamics(fused_input)
        return torch.clamp(raw_output, min=-2.0, max=2.0)

    def forward(self, t_span, state0):
        """
        [BẢN VÁ]: Sử dụng odeint chuẩn thay vì Euler tự code.
        t_span: tensor chứa các mốc thời gian, vd: torch.tensor([0.0, 1.0])
        state0: trạng thái ban đầu
        """
        trajectory = odeint(self._ode_func, state0, t_span, method='rk4')
        return trajectory
    
    def compute_loss(self, predicted_audio, vocal_feedback):
        """
        [BẢN VÁ]: Dùng VOCAL_FEEDBACK từ môi trường làm ground truth 
        thay vì nhãn chuỗi (string encoding) cũ.
        """
        if not isinstance(vocal_feedback, torch.Tensor):
            vocal_feedback = torch.tensor(vocal_feedback, dtype=torch.float32, device=predicted_audio.device)
        return torch.nn.functional.mse_loss(predicted_audio, vocal_feedback)

    def prepare_candidate(self, tx_id: str):
        self._staged_tx_id = tx_id
        self._evo_state = "CANDIDATE_STAGED"
        self._parent_state = copy.deepcopy(self.state_dict())
        self._candidate_state = copy.deepcopy(self.state_dict())
        return {"status": "prepared", "transaction_id": tx_id}

    def commit_candidate(self, tx_id: str) -> bool:
        """
        [BẢN VÁ]: Kiểm tra chặt chẽ staged_tx_id. Phải khớp mới cho commit.
        """
        if getattr(self, 'last_committed_tx_id', None) == tx_id:
            return True

        if self._evo_state != "CANDIDATE_STAGED":
            return False
        if self._staged_tx_id != tx_id:
            return False
            
        self.last_committed_tx_id = tx_id
        self._evo_state = "LIVE"
        self._parent_state = None
        self._candidate_state = None
        self._staged_tx_id = None
        self.generation += 1
        return True

    def rollback_candidate(self, tx_id: str) -> bool:
        """
        [BẢN VÁ]: Kiểm tra chặt chẽ staged_tx_id. Phải khớp mới cho rollback.
        """
        if self._evo_state != "CANDIDATE_STAGED":
            return False
        if self._staged_tx_id != tx_id:
            return False
            
        if self._parent_state is not None:
            self.load_state_dict(self._parent_state)
            
        self._evo_state = "LIVE"
        self._parent_state = None
        self._candidate_state = None
        self._staged_tx_id = None
        return True

    def _verify_topology_integrity(self) -> bool:
        in_dim = self.physical_dim + self.motor_dim
        if self.hidden_dim <= 0 or self.dynamics[0].in_features != in_dim or self.dynamics[0].out_features != self.hidden_dim:
            raise RuntimeError(f"[M10_Vocal] Lệch shape dynamics[0]: {tuple(self.dynamics[0].weight.shape)}")
        if self.dynamics[2].in_features != self.hidden_dim or self.dynamics[2].out_features != self.physical_dim:
            raise RuntimeError(f"[M10_Vocal] Lệch shape dynamics[2]: {tuple(self.dynamics[2].weight.shape)}")
        for p in self.parameters():
            if not torch.all(torch.isfinite(p)):
                raise RuntimeError("[M10_Vocal] Phát hiện NaN/Inf trong trọng số VocalTractODE!")
        return True

    def _can_grow_neuron(self, target_hidden: int):
        import json
        max_nodes = 1000
        max_model_params = 30000
        global_max_params = 500000
        max_rss_mb = 2048.0
        arch_path = os.path.join(BASE_DIR, "architecture_manifest.json")
        if os.path.exists(arch_path):
            try:
                with open(arch_path, "r", encoding="utf-8") as f:
                    arch = json.load(f)
                gov = arch.get("system_topology", {}).get("resource_governor", arch.get("governor", {}))
                m10_cfg = arch.get("modules", {}).get("M10_Vocal", {})
                evo_cfg = m10_cfg.get("evolution", {})
                if not evo_cfg.get("structural", False):
                    return False, "evolution.structural=false trong manifest"
                max_nodes = min(
                    int(gov.get("max_nodes_per_module", {}).get("M10_Vocal", max_nodes)),
                    int(evo_cfg.get("max_nodes", max_nodes))
                )
                max_model_params = min(
                    int(gov.get("max_parameters_per_module", {}).get("M10_Vocal", max_model_params)),
                    int(evo_cfg.get("max_parameters", gov.get("max_params_per_model", max_model_params)))
                )
                global_max_params = int(gov.get("global_max_parameters", global_max_params))
                max_rss_mb = float(gov.get("max_process_rss_mb", max_rss_mb))
            except Exception:
                pass
        if target_hidden > max_nodes:
            return False, f"Vượt giới hạn node của Governor ({target_hidden} > {max_nodes})"
        in_dim = self.physical_dim + self.motor_dim
        projected_params = (in_dim * target_hidden + target_hidden) + (target_hidden * self.physical_dim + self.physical_dim)
        if projected_params > max_model_params or projected_params > global_max_params:
            return False, f"Vượt ngân sách tham số Governor ({projected_params} > {min(max_model_params, global_max_params)})"
        try:
            import psutil
            vm = psutil.virtual_memory()
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
            if vm.percent >= 90.0 or rss_mb >= max_rss_mb:
                return False, f"Vượt ngưỡng RAM an toàn (SysRAM={vm.percent:.1f}%, ProcRSS={rss_mb:.1f}MB)"
        except Exception:
            pass
        return True, "OK"

    @staticmethod
    def _migrate_optimizer_tensors(old_opt: torch.optim.Optimizer, old_params: Dict[str, nn.Parameter], new_params: Dict[str, nn.Parameter], old_hidden: int):
        if old_opt is None:
            return
        new_group_params = []
        for name, new_p in new_params.items():
            old_p = old_params.get(name)
            if old_p is not None and old_p in old_opt.state:
                old_st = old_opt.state.pop(old_p)
                new_st = {}
                for k, v in old_st.items():
                    if torch.is_tensor(v) and v.shape == old_p.shape:
                        nv = torch.zeros_like(new_p, memory_format=torch.preserve_format)
                        if name == "dynamics.0.weight":
                            nv[:old_hidden, :].copy_(v[:old_hidden, :])
                        elif name == "dynamics.0.bias":
                            nv[:old_hidden].copy_(v[:old_hidden])
                        elif name == "dynamics.2.weight":
                            nv[:, :old_hidden].copy_(v[:, :old_hidden])
                        else:
                            nv.copy_(v)
                        new_st[k] = nv
                    else:
                        new_st[k] = copy.deepcopy(v)
                old_opt.state[new_p] = new_st
            new_group_params.append(new_p)
        if len(old_opt.param_groups) > 0:
            old_opt.param_groups[0]["params"] = new_group_params

    def sprout_new_neuron(self, optimizer: torch.optim.Optimizer = None, enforce_governor: bool = True):
        old_hidden = self.hidden_dim
        new_hidden = old_hidden + 1
        if enforce_governor:
            can_grow, _ = self._can_grow_neuron(new_hidden)
            if not can_grow:
                return False

        backup_fc1 = copy.deepcopy(self.dynamics[0])
        backup_fc2 = copy.deepcopy(self.dynamics[2])
        backup_topo_ver = self.topology_version
        old_named_params = dict(self.named_parameters())
        device = self.dynamics[0].weight.device
        try:
            new_fc1 = nn.Linear(self.physical_dim + self.motor_dim, new_hidden).to(device)
            new_fc2 = nn.Linear(new_hidden, self.physical_dim).to(device)
            with torch.no_grad():
                new_fc1.weight.data[:old_hidden, :].copy_(self.dynamics[0].weight.data)
                new_fc1.bias.data[:old_hidden].copy_(self.dynamics[0].bias.data)
                new_fc1.weight.data[old_hidden, :] = torch.randn(new_fc1.in_features, device=device) * 0.01
                new_fc1.bias.data[old_hidden] = 0.0

                new_fc2.weight.data[:, :old_hidden].copy_(self.dynamics[2].weight.data)
                new_fc2.bias.data.copy_(self.dynamics[2].bias.data)
                new_fc2.weight.data[:, old_hidden] = torch.randn(new_fc2.out_features, device=device) * 0.01

            self.dynamics[0] = new_fc1
            self.dynamics[2] = new_fc2
            self.hidden_dim = new_hidden
            self._verify_topology_integrity()
            self.topology_version += 1
            if optimizer is not None:
                self._migrate_optimizer_tensors(optimizer, old_named_params, dict(self.named_parameters()), old_hidden)
            return True
        except Exception:
            self.dynamics[0] = backup_fc1
            self.dynamics[2] = backup_fc2
            self.hidden_dim = old_hidden
            self.topology_version = backup_topo_ver
            return False

    def load_state_dict(self, state_dict, strict=True):
        backup_fc1 = copy.deepcopy(self.dynamics[0])
        backup_fc2 = copy.deepcopy(self.dynamics[2])
        backup_hidden = self.hidden_dim
        try:
            weight = state_dict.get("dynamics.0.weight")
            if weight is not None:
                actual_hidden = int(weight.shape[0])
                if actual_hidden != self.hidden_dim:
                    self.hidden_dim = actual_hidden
                    device = self.dynamics[0].weight.device
                    self.dynamics[0] = nn.Linear(self.physical_dim + self.motor_dim, self.hidden_dim).to(device)
                    self.dynamics[2] = nn.Linear(self.hidden_dim, self.physical_dim).to(device)
            res = super(VocalTractODE, self).load_state_dict(state_dict, strict=strict)
            self._verify_topology_integrity()
            return res
        except Exception as e:
            self.dynamics[0] = backup_fc1
            self.dynamics[2] = backup_fc2
            self.hidden_dim = backup_hidden
            raise e


if __name__ == "__main__":
    print("🧠 [TRAINER] Đang khởi tạo M10_Vocal (VocalTractODE) Canonical...")
    from shared_utils import CheckpointManager

    model = VocalTractODE(motor_dim=3, physical_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("💾 Đang ghi Canonical Checkpoint thông qua CheckpointManager...")
    
    pretraining_gens = 15
    
    extra_metadata = {
        "topology": {
            "motor_dim": model.motor_dim,
            "physical_dim": model.physical_dim,
            "hidden_dim": model.hidden_dim
        },
        "generation": 0,
        "pretraining_generation": pretraining_gens,
        "fitness": 0.0,
        "topology_version": 1,
        "mutation_record": f"VocalTractODE Pretraining ({pretraining_gens} gens)",
        "transaction_state": {
            "current_transaction_id": "",
            "status": "COMMITTED",
            "candidate_hash": None,
            "staged_topology_version": 0
        }
    }

    CheckpointManager.save("M10_Vocal", model, optimizer, is_torch=True, extra_states=extra_metadata)
    print("✅ Đã đúc thành công Canonical Checkpoint (Genesis) cho M10_Vocal!")