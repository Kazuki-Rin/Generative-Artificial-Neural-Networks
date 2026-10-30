import os
import sys
import time
import pytest
import torch
import numpy as np
import threading
from unittest.mock import patch, MagicMock

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "00_Central_Router", "M9_Thalamus"))
sys.path.append(os.path.join(BASE_DIR, "04_Memory_System", "M8_Hippocampus"))
sys.path.append(os.path.join(BASE_DIR, "03_Cognitive_Core"))

from m9_dynamic_som import DynamicSOM  # type: ignore
from m8_art_memory import HippocampusMemory, SimplifiedART  # type: ignore
from liquid_progressive_brain import ProgressiveLiquidBrain  # type: ignore

def test_m9_numerical_integrity():
    model = DynamicSOM(input_dim=128, initial_nodes=4)
    x = torch.randn(1, 128) * 1000 # extreme values
    bmu, changed = model.update_and_grow(x)
    assert bmu is not None
    assert torch.isfinite(model.weights).all()

def test_m9_feedback_constraint():
    model = DynamicSOM(input_dim=128, initial_nodes=4)
    model.apply_plasticity(1.0, 0)
    assert model.attention_to_route.data[0, 5].item() == 0.0

def test_m8_inference_responsiveness():
    os.environ["ASUNA_ALLOW_BOOTSTRAP"] = "1"
    node = HippocampusMemory(node_id="M8_Resp_Test")
    try:
        start = time.time()
        for _ in range(100):
            res = node.encode_semantic({"fused_latent": [0.1]*128})
            assert res is not None and len(res) == 128
        duration = time.time() - start
        assert duration < 0.5, "Inference is blocked and too slow"
    finally:
        try: node.training_queue.put_nowait(None)
        except Exception: pass
        try: node.upstream_sock.close(linger=0)
        except Exception: pass
        try: node.bus.shutdown()
        except Exception: pass

def test_m8_legacy_persistence():
    art = SimplifiedART()
    legacy_state = {
        "vigilance": 0.85,
        "input_dim": 128,
        "nodes": [
            {"weight": [0.5]*128, "count": 1}
        ]
    }
    art.load_state(legacy_state)
    assert len(art.memory_nodes) == 1
    assert "node_id" in art.memory_nodes[0]
    assert art.memory_nodes[0]["node_id"].startswith("legacy_")
    
    art.memory_nodes[0]["weight"][0] = 0.6
    
    id1 = art.memory_nodes[0]["node_id"]
    art.load_state(legacy_state)
    id2 = art.memory_nodes[0]["node_id"]
    assert id1 == id2

def test_m6_dimension_match():
    m6 = ProgressiveLiquidBrain(input_dim=128, hidden_dim=128, output_dim=133, name="M6_Language")
    x = torch.randn(1, 128)
    cross_modal = {"thalamus": torch.randn(1, 128), "memory": torch.randn(1, 39)}
    out = m6(x, cross_modal_inputs=cross_modal)
    assert out.shape == (1, 133)