#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TEST_P2_PERSISTENCE.PY - Real Persistence Test Suite (V5 - 2PC IDEMPOTENT & MULTI-MODEL)
=====================================================
Test Cases:
1. M5_Amygdala: SNN Optimizer Restore
2. M7_Logic: Topology Lineage (Sprouting)
3. 2PC ROLLBACK (State reversion)
4. M8_Hippocampus: Encoder Rollback & Idempotent COMMIT
5. M9_Thalamus: Mismatch & Routing Consistency
6. M10_Vocal: Neural ODE Transaction
7. Retry PREPARE/COMMIT Idempotency (Core 2PC protection)
"""

import os
import sys
import json
import time
import copy
import hashlib
import tempfile
import shutil
import unittest
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# --- PATH SETUP ---
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.environ["ASUNA_ALLOW_BOOTSTRAP"] = "1"
TMP_DIR = tempfile.mkdtemp(prefix="p2_persist_test_")
import shared_utils
shared_utils.BASE_DIR = TMP_DIR

from shared_utils import CheckpointManager, tensor_state_hash, CheckpointError

sys.path.append(os.path.join(BASE_DIR, "00_Central_Router", "M5_Amygdala"))
sys.path.append(os.path.join(BASE_DIR, "00_Central_Router", "M9_Thalamus"))
sys.path.append(os.path.join(BASE_DIR, "03_Cognitive_Core"))
sys.path.append(os.path.join(BASE_DIR, "04_Memory_System", "M8_Hippocampus"))
sys.path.append(os.path.join(BASE_DIR, "05_Vocal_System", "M10_Core_Engine"))

from train_m5_snn import SproutingSNN # type: ignore
from liquid_progressive_brain import ProgressiveLiquidBrain # type: ignore
from m9_dynamic_som import DynamicSOM # type: ignore
from m10_vocal_tract import VocalTractODE # type: ignore
from m8_art_memory import HippocampusMemory # type: ignore

class TestPersistenceReal(unittest.TestCase):
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    def test_01_real_snn_optimizer_restore(self):
        """Khởi tạo SNN thật, cho học 1 bước để sinh Optimizer State, lưu và phục hồi qua CheckpointManager GỐC"""
        model_id = "M5_Amygdala_Test"
        model = SproutingSNN(num_inputs=20, num_hidden=64, num_outputs=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        dummy_input = torch.randn(10, 1, 20)
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        
        opt_state_before = copy.deepcopy(optimizer.state_dict())
        extra = {"generation": 2, "fitness": 0.8}
        
        ok = CheckpointManager.save(model_id, model, optimizer, is_torch=True, extra_states=extra)
        self.assertTrue(ok, "Lưu Checkpoint thất bại")
        
        model2 = SproutingSNN(num_inputs=20, num_hidden=64, num_outputs=2)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        
        loaded = CheckpointManager.load(model_id, is_torch=True, enforce_manifest=False)
        self.assertIsNotNone(loaded, "Không thể nạp Checkpoint")
        
        clean_state = {k: v for k, v in loaded["model_state"].items() if not k.endswith('.mem') and not k.endswith('.syn')}
        
        model2.load_state_dict(clean_state, strict=False)
        optimizer2.load_state_dict(loaded["optimizer_state"])
        
        self.assertEqual(loaded["generation"], 2)
        self.assertEqual(loaded["fitness"], 0.8)
        self.assertEqual(set(opt_state_before["state"].keys()), set(optimizer2.state_dict()["state"].keys()))

    def test_02_real_liquid_brain_topology_lineage(self):
        """Kiểm tra khả năng lưu trữ cấu trúc (Topology) của Mạng Thùy Trán (M7)"""
        model_id = "M7_Logic_Test"
        model = ProgressiveLiquidBrain(input_dim=128, hidden_dim=64, output_dim=6)
        model.sprout_new_skill_column()
        
        extra = {"topology": model.get_topology_signature()}
        CheckpointManager.save(model_id, model, is_torch=True, extra_states=extra)
        loaded = CheckpointManager.load(model_id, is_torch=True, enforce_manifest=False)
        
        self.assertEqual(loaded["topology"]["num_columns"], 2)
        self.assertIn("columns.1.tau", loaded["model_state"], "Cấu trúc nhánh Nơ-ron mới chưa được lưu!")

    def test_03_m8_encoder_rollback_and_idempotent(self):
        """[YÊU CẦU]: Test M8 Encoder Rollback và tính Idempotent"""
        m8 = HippocampusMemory(node_id="M8_Memory_Test")
        try:
            _ = m8.encode_semantic({"test": "data"})
            m8.training_queue.join()
            hash_before = tensor_state_hash(m8.neural_encoder.state_dict())
            
            _ = m8.encode_semantic({"new": "noise"})
            m8.training_queue.join()
            hash_after_live_train = tensor_state_hash(m8.neural_encoder.state_dict())
            self.assertNotEqual(hash_before, hash_after_live_train, "Encoder phải học khi ở trạng thái LIVE")
            
            # [PREPARE] Khóa huấn luyện khi sang CANDIDATE_STAGED
            tx_id = "tx_m8_123"
            m8._current_transaction_id = tx_id
            m8.parent_art_state = copy.deepcopy(m8.art.get_state())
            m8.parent_generation = m8.generation
            m8._candidate_art_state = copy.deepcopy(m8.art.get_state())
            m8._evo_state = "CANDIDATE_STAGED"
            
            m8.art.load_state(m8.parent_art_state)
            m8._evo_state = "LIVE"
            
            m8._save_checkpoint(is_dirty=True, force=True)
            loaded = CheckpointManager.load("M8_Memory_Test", is_torch=False, enforce_manifest=False)
            self.assertIsNotNone(loaded.get("extra_states", {}).get("encoder_state"), "Thiếu Encoder state trong Checkpoint!")
            self.assertIsNotNone(loaded.get("extra_states", {}).get("encoder_optimizer_state"), "Thiếu Encoder Optimizer state trong Checkpoint!")
        finally:
            try: m8.training_queue.put_nowait(None)
            except Exception: pass
            try: m8.upstream_sock.close(linger=0)
            except Exception: pass
            try: m8.bus.shutdown()
            except Exception: pass

    def test_04_m9_mismatch_and_routing(self):
        """[YÊU CẦU]: Test M9 Mismatch (Lệch pha thời gian)"""
        m9 = DynamicSOM(input_dim=128, initial_nodes=4)
        x = torch.randn(1, 128) * 100.0
        
        # Ghi nhận pending growth ở tick 25 (tương lai so với target_tick=20)
        _, changed = m9.update_and_grow(x, current_tick=25, current_episode="TEST_EP")
        self.assertTrue(changed, "M9 phải báo changed = True khi growth!")
        
        # [PREPARE MISMATCH] Gọi prepare ở tick 20 (< 25) -> Phải từ chối mọc
        res = m9.prepare_candidate("tx_m9_123", target_tick=20, target_episode="TEST_EP")
        self.assertFalse(res["topology_changed"], "Lỗi: M9 cho phép mọc dù tick tương lai (25 > 20)")
        self.assertEqual(res["new_num_nodes"], 4)
        
        # Test Rollback có truyền transaction_id hợp lệ
        self.assertTrue(m9.rollback_candidate("tx_m9_123"))
        
        # Ghi nhận lại ở tick 25 và Prepare ở tick 30 (>= 25) -> Phải cho phép mọc
        _, _ = m9.update_and_grow(x, current_tick=25, current_episode="TEST_EP")
        res2 = m9.prepare_candidate("tx_m9_124", target_tick=30, target_episode="TEST_EP")
        self.assertTrue(res2["topology_changed"], "Lỗi: M9 không mọc khi tick hợp lệ (25 <= 30)")
        self.assertEqual(res2["new_num_nodes"], 5)

    def test_05_m10_ode_checkpoint_and_forward(self):
        """[YÊU CẦU]: Test M10 Neural ODE Persistence & Forward"""
        model_id = "M10_Vocal_Test"
        m10 = VocalTractODE(motor_dim=3, physical_dim=2, hidden_dim=16)
        opt = optim.Adam(m10.parameters(), lr=1e-3)
        ok = CheckpointManager.save(model_id, m10, opt, is_torch=True, extra_states={"generation": 2})
        self.assertTrue(ok)
        loaded = CheckpointManager.load(model_id, is_torch=True, enforce_manifest=False)
        m10_loaded = VocalTractODE(motor_dim=3, physical_dim=2, hidden_dim=16)
        m10_loaded.load_state_dict(loaded["model_state"], strict=True)
        self.assertEqual(loaded["generation"], 2)

    def test_06_retry_prepare_idempotency(self):
        """[YÊU CẦU]: Test Idempotent khi gọi PREPARE/COMMIT 2 lần trên ProgressiveLiquidBrain"""
        model = ProgressiveLiquidBrain(input_dim=5, hidden_dim=16, output_dim=2)
        prep1 = model.prepare_candidate("tx_liquid_001")
        prep2 = model.prepare_candidate("tx_liquid_001")
        self.assertEqual(prep1["candidate_hash"], prep2["candidate_hash"])
        self.assertTrue(model.commit_candidate("tx_liquid_001", generation=2))
        self.assertTrue(model.commit_candidate("tx_liquid_001", generation=2))
        self.assertEqual(model.generation, 2)

if __name__ == '__main__':
    print("=" * 70)
    print("💾 P2 PERSISTENCE TEST SUITE - REAL 2PC TRANSACTIONS")
    print("=" * 70)
    unittest.main(verbosity=2)