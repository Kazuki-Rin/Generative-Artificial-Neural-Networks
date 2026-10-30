#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
from typing import Dict
import torch
import torch.nn as nn
import snntorch as snn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

class SproutingSNN(nn.Module):
    def __init__(self, num_inputs=20, num_hidden=64, num_outputs=2):
        super(SproutingSNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        self.topology_version = 1
        self.weight_revision = 1
        
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif2 = snn.Leaky(beta=0.9)
        
        self.mem1 = None
        self.mem2 = None

    def reset_membrane_state(self):
        """[BẢN VÁ AUDIT #25]: Hàm xả rỗng điện thế màng khi kết thúc 1 Episode/Ngủ dậy"""
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def forward(self, x):
        if getattr(self, 'mem1', None) is None or getattr(self, 'mem2', None) is None:
            self.reset_membrane_state()
            
        expected_batch = x.size(1) if x.dim() == 3 else 0
        mem1_valid = getattr(self.mem1, 'dim', lambda: 0)() == 2 and self.mem1.size(0) == expected_batch and self.mem1.size(1) == self.num_hidden
        mem2_valid = getattr(self.mem2, 'dim', lambda: 0)() == 2 and self.mem2.size(0) == expected_batch and self.mem2.size(1) == self.num_outputs
        if not (mem1_valid and mem2_valid):
            self.reset_membrane_state()
            
        if self.mem1.device != x.device:
            self.mem1 = self.mem1.to(x.device)
            self.mem2 = self.mem2.to(x.device)
        
        spk2_rec = []
        mem1 = self.mem1.detach()
        mem2 = self.mem2.detach()
        
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            
        self.mem1 = mem1.detach()
        self.mem2 = mem2.detach()
        
        return torch.stack(spk2_rec, dim=0)

    def _verify_topology_integrity(self) -> bool:
        if self.num_hidden <= 0 or self.num_inputs <= 0 or self.num_outputs <= 0:
            raise RuntimeError(f"[M5_Amygdala] Kích thước topology không hợp lệ: ({self.num_inputs}, {self.num_hidden}, {self.num_outputs})")
        if self.fc1.in_features != self.num_inputs or self.fc1.out_features != self.num_hidden:
            raise RuntimeError(f"[M5_Amygdala] Lệch shape fc1: {tuple(self.fc1.weight.shape)} != ({self.num_hidden}, {self.num_inputs})")
        if self.fc2.in_features != self.num_hidden or self.fc2.out_features != self.num_outputs:
            raise RuntimeError(f"[M5_Amygdala] Lệch shape fc2: {tuple(self.fc2.weight.shape)} != ({self.num_outputs}, {self.num_hidden})")
        for p in self.parameters():
            if not torch.all(torch.isfinite(p)):
                raise RuntimeError("[M5_Amygdala] Phát hiện NaN/Inf trong trọng số SproutingSNN!")
        return True

    def _can_grow_neuron(self, target_hidden: int):
        import json
        max_nodes = 200
        max_model_params = 30000
        global_max_params = 500000
        max_rss_mb = 2048.0
        arch_path = os.path.join(BASE_DIR, "architecture_manifest.json")
        if os.path.exists(arch_path):
            try:
                with open(arch_path, "r", encoding="utf-8") as f:
                    arch = json.load(f)
                gov = arch.get("system_topology", {}).get("resource_governor", arch.get("governor", {}))
                m5_cfg = arch.get("modules", {}).get("M5_Amygdala", {})
                evo_cfg = m5_cfg.get("evolution", {})
                if not evo_cfg.get("structural", True):
                    return False, "evolution.structural=false trong manifest"
                max_nodes = min(
                    int(gov.get("max_nodes_per_module", {}).get("M5_Amygdala", max_nodes)),
                    int(evo_cfg.get("max_nodes", max_nodes))
                )
                max_model_params = min(
                    int(gov.get("max_parameters_per_module", {}).get("M5_Amygdala", max_model_params)),
                    int(evo_cfg.get("max_parameters", gov.get("max_params_per_model", max_model_params)))
                )
                global_max_params = int(gov.get("global_max_parameters", global_max_params))
                max_rss_mb = float(gov.get("max_process_rss_mb", max_rss_mb))
            except Exception:
                pass

        if target_hidden > max_nodes:
            return False, f"Vượt giới hạn node của Governor ({target_hidden} > {max_nodes})"
        projected_params = (self.num_inputs * target_hidden + target_hidden) + (target_hidden * self.num_outputs + self.num_outputs)
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
                        if name == "fc1.weight":
                            nv[:old_hidden, :].copy_(v[:old_hidden, :])
                        elif name == "fc1.bias":
                            nv[:old_hidden].copy_(v[:old_hidden])
                        elif name == "fc2.weight":
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
        old_hidden = self.num_hidden
        new_hidden = old_hidden + 1
        if enforce_governor:
            can_grow, reason = self._can_grow_neuron(new_hidden)
            if not can_grow:
                return False

        backup_fc1 = copy.deepcopy(self.fc1)
        backup_fc2 = copy.deepcopy(self.fc2)
        backup_topo_ver = self.topology_version
        old_named_params = dict(self.named_parameters())
        device = self.fc1.weight.device
        try:
            new_fc1 = nn.Linear(self.num_inputs, new_hidden).to(device)
            new_fc2 = nn.Linear(new_hidden, self.num_outputs).to(device)
            
            with torch.no_grad():
                new_fc1.weight.data[:old_hidden, :].copy_(self.fc1.weight.data)
                new_fc1.bias.data[:old_hidden].copy_(self.fc1.bias.data)
                new_fc1.weight.data[old_hidden, :] = torch.randn(new_fc1.in_features, device=device) * 0.01
                new_fc1.bias.data[old_hidden] = 0.0
                
                new_fc2.weight.data[:, :old_hidden].copy_(self.fc2.weight.data)
                new_fc2.bias.data.copy_(self.fc2.bias.data)
                new_fc2.weight.data[:, old_hidden] = torch.randn(new_fc2.out_features, device=device) * 0.01

            self.fc1 = new_fc1
            self.fc2 = new_fc2
            self.num_hidden = new_hidden
            self._verify_topology_integrity()
            self.topology_version += 1
            self.mem1 = None
            self.mem2 = None
            if optimizer is not None:
                self._migrate_optimizer_tensors(optimizer, old_named_params, dict(self.named_parameters()), old_hidden)
            return True
        except Exception:
            self.fc1 = backup_fc1
            self.fc2 = backup_fc2
            self.num_hidden = old_hidden
            self.topology_version = backup_topo_ver
            self.mem1 = None
            self.mem2 = None
            return False

    add_hidden_neuron = sprout_new_neuron

    def get_topology_signature(self) -> Dict:
        return {
            "num_inputs": self.num_inputs,
            "num_hidden": self.num_hidden,
            "num_outputs": self.num_outputs,
            "topology_version": self.topology_version,
            "weight_revision": self.weight_revision
        }

    def load_state_dict(self, state_dict, strict=True):
        backup_fc1 = copy.deepcopy(self.fc1)
        backup_fc2 = copy.deepcopy(self.fc2)
        backup_hidden = self.num_hidden
        try:
            state_dict = dict(state_dict)
            w1 = state_dict.get("fc1.weight")
            w2 = state_dict.get("fc2.weight")
            if w1 is not None:
                if w1.dim() != 2 or w1.shape[1] != self.num_inputs:
                    raise RuntimeError(f"[M5_Amygdala] fc1.weight shape không hợp lệ: {tuple(w1.shape)}")
                actual_hidden = int(w1.shape[0])
                if w2 is not None and (w2.dim() != 2 or w2.shape[1] != actual_hidden or w2.shape[0] != self.num_outputs):
                    raise RuntimeError(f"[M5_Amygdala] fc2.weight shape {tuple(w2.shape)} không khớp với fc1 ({actual_hidden})")
                if actual_hidden != self.num_hidden:
                    self.num_hidden = actual_hidden
                    device = self.fc1.weight.device if hasattr(self, 'fc1') else torch.device('cpu')
                    self.fc1 = nn.Linear(self.num_inputs, self.num_hidden).to(device)
                    self.fc2 = nn.Linear(self.num_hidden, self.num_outputs).to(device)
                    self.mem1 = None
                    self.mem2 = None
            for mem_key, lif_mod in (("lif1.mem", self.lif1), ("lif2.mem", self.lif2)):
                if hasattr(lif_mod, "mem") and isinstance(lif_mod.mem, torch.Tensor):
                    if mem_key not in state_dict or state_dict[mem_key].shape != lif_mod.mem.shape:
                        state_dict[mem_key] = lif_mod.mem.clone()
            res = super(SproutingSNN, self).load_state_dict(state_dict, strict=strict)
            self.mem1 = None
            self.mem2 = None
            self._verify_topology_integrity()
            return res
        except Exception as e:
            self.fc1 = backup_fc1
            self.fc2 = backup_fc2
            self.num_hidden = backup_hidden
            self.mem1 = None
            self.mem2 = None
            raise e

if __name__ == "__main__":
    print("🧠 [TRAINER] Đang khởi tạo M5_Amygdala (SproutingSNN) Canonical...")
    from shared_utils import CheckpointManager

    model = SproutingSNN(num_inputs=20, num_hidden=64, num_outputs=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pretraining_gens = 15
    
    extra_st = {
        "topology": {"num_inputs": model.num_inputs, "num_hidden": model.num_hidden, "num_outputs": model.num_outputs},
        "generation": 0, 
        "pretraining_generation": pretraining_gens,
        "fitness": 0.0, 
        "topology_version": model.topology_version,
        "mutation_record": f"SproutingSNN Pretraining ({pretraining_gens} gens) - Amygdala Threat Detection",
        "transaction_state": {
            "current_transaction_id": "",
            "status": "COMMITTED",
            "candidate_hash": None,
            "staged_topology_version": 0
        }
    }
    CheckpointManager.save("M5_Amygdala", model, optimizer, is_torch=True, extra_states=extra_st)
    print("✅ Đã đúc thành công Canonical Checkpoint (Genesis) cho M5_Amygdala!")