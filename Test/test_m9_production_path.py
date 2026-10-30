import os
import sys
import time
import json
import threading
import torch
import numpy as np
import zmq

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "00_Central_Router", "M9_Thalamus"))
sys.path.append(os.path.join(BASE_DIR, "04_Memory_System", "M8_Hippocampus"))
sys.path.append(os.path.join(BASE_DIR, "03_Cognitive_Core"))

from unittest.mock import patch, MagicMock

import main_life_loop
from main_life_loop import m9_thalamus_process, m8_memory_process, m6_language_process
import m9_dynamic_som  # type: ignore
import pytest
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def mock_get_all_msgs_generator(msg_queue, stop_event, bus_instance):
    def mock_get_all_msgs(sock, bus=None):
        if sock == bus_instance.ctrl_socket:
            ctrl_msgs = [m for m in msg_queue if m and "SYSTEM_CONTROL" in m[0]]
            if ctrl_msgs:
                msg_queue.remove(ctrl_msgs[0])
                return ctrl_msgs[0]
            return []
        else:
            data_msgs = [m for m in msg_queue if m and "SYSTEM_CONTROL" not in m[0]]
            if data_msgs:
                msg_queue.remove(data_msgs[0])
                return data_msgs[0]
            
            if not msg_queue:
                stop_event.set()
            return []
    return mock_get_all_msgs

def test_m9_fused_latent_numerical_stability():
    stop_event = threading.Event()
    with patch("main_life_loop.CoreBus") as MockBus, patch("zmq.Context"), patch("zmq.Poller") as MockPoller, patch("main_life_loop.CheckpointManager.save"), patch("main_life_loop.load_worker_checkpoint_strict", return_value=(None, "DEGRADED")), patch("signal.signal"), patch("main_life_loop._publish_telemetry"):
        bus_instance = MockBus.return_value
        bus_instance.node_id = "M9_Thalamus"
        bus_instance._last_telemetry_time = 0.0
        bus_instance.ctrl_socket = MagicMock()
        
        msg_queue = []
        mock_poller_instance = MockPoller.return_value
        fake_sock = MagicMock()
        mock_poller_instance.poll.return_value = [(bus_instance.ctrl_socket, zmq.POLLIN), (fake_sock, zmq.POLLIN)]
        
        with patch("main_life_loop.get_all_msgs", side_effect=mock_get_all_msgs_generator(msg_queue, stop_event, bus_instance)):
            extreme_vision = {"latent_vector": [1e20] * 128}
            nan_audio = {"acoustic_features": [np.nan] * 128}
            
            awake_pkt = json.dumps({"state": "AWAKE"})
            packet1 = json.dumps({"payload": extreme_vision, "event_id": "ev1", "task_id": "t1", "tick_id": 1})
            packet2 = json.dumps({"payload": nan_audio, "event_id": "ev2", "task_id": "t2", "tick_id": 2})
            
            msg_queue.append([f"SYSTEM_CONTROL {awake_pkt}"])
            msg_queue.append([f"RAW_VISION_M1A {packet1}"])
            msg_queue.append([f"RAW_AUDIO {packet2}"])
            
            m9_thread = threading.Thread(target=m9_thalamus_process, args=(stop_event,))
            m9_thread.start()
            m9_thread.join(timeout=5.0)
            
            published_calls = bus_instance.publish.call_args_list
            routing_payloads = [call[0][1] for call in published_calls if call[1].get('modality') == 'routing']
            
            assert len(routing_payloads) > 0, "M9 should have broadcasted routing."
            for payload in routing_payloads:
                fused = payload["fused_latent"]
                assert len(fused) == 128, "fused_latent must be exactly 128D"
                arr = np.array(fused)
                assert np.all(np.isfinite(arr)), "fused_latent must not contain NaN/Inf"
                assert np.max(np.abs(arr)) <= 1.0, "fused_latent should be bounded by Tanh to [-1, 1]"

def test_recursive_storm_budget():
    """
    Prove M9's per-tick message budget limits broadcasts.
    """
    stop_event = threading.Event()
    with patch("main_life_loop.CoreBus") as MockBus, patch("zmq.Context"), patch("zmq.Poller") as MockPoller, patch("main_life_loop.CheckpointManager.save"), patch("main_life_loop.load_worker_checkpoint_strict", return_value=(None, "DEGRADED")), patch("signal.signal"), patch("main_life_loop._publish_telemetry"):
        bus_instance = MockBus.return_value
        bus_instance.node_id = "M9_Thalamus"
        bus_instance._last_telemetry_time = 0.0
        bus_instance.ctrl_socket = MagicMock()
        
        msg_queue = []
        mock_poller_instance = MockPoller.return_value
        fake_sock = MagicMock()
        mock_poller_instance.poll.return_value = [(bus_instance.ctrl_socket, zmq.POLLIN), (fake_sock, zmq.POLLIN)]
        
        with patch("main_life_loop.get_all_msgs", side_effect=mock_get_all_msgs_generator(msg_queue, stop_event, bus_instance)):
            awake_pkt = json.dumps({"state": "AWAKE"})
            msg_queue.append([f"SYSTEM_CONTROL {awake_pkt}"])
            for i in range(10):
                packet = json.dumps({"payload": {"latent_vector": [0.5]*128}, "event_id": f"ev{i}", "task_id": "t1", "tick_id": 1})
                msg_queue.append([f"RAW_VISION_M1A {packet}"])
                
            m9_thread = threading.Thread(target=m9_thalamus_process, args=(stop_event,))
            m9_thread.start()
            m9_thread.join(timeout=5.0)
            
            published_calls = bus_instance.publish.call_args_list
            routing_payloads = [call[0][1] for call in published_calls if call[1].get('modality') == 'routing']
            
            assert 0 < len(routing_payloads) <= 15, "Should have broadcasted but limited by budget."

def test_recursive_storm_termination():
    """
    P1: Real message-storm integration test.
    Test termination of an M6 -> M9 -> M6 feedback cycle.
    """
    stop_event = threading.Event()
    with patch("main_life_loop.CoreBus") as MockBus, patch("zmq.Context"), patch("zmq.Poller") as MockPoller, patch("main_life_loop.CheckpointManager.save"), patch("main_life_loop.load_worker_checkpoint_strict", return_value=(None, "DEGRADED")), patch("signal.signal"), patch("main_life_loop._publish_telemetry"):
        bus_instance = MockBus.return_value
        bus_instance.node_id = "M9_Thalamus"
        bus_instance._last_telemetry_time = 0.0
        bus_instance.ctrl_socket = MagicMock()
        
        msg_queue = []
        mock_poller_instance = MockPoller.return_value
        fake_sock = MagicMock()
        mock_poller_instance.poll.return_value = [(bus_instance.ctrl_socket, zmq.POLLIN), (fake_sock, zmq.POLLIN)]
        
        with patch("main_life_loop.get_all_msgs", side_effect=mock_get_all_msgs_generator(msg_queue, stop_event, bus_instance)):
            awake_pkt = json.dumps({"state": "AWAKE"})
            msg_queue.append([f"SYSTEM_CONTROL {awake_pkt}"])
            packet = json.dumps({"payload": {"semantic_embedding": [1.0]*39}, "event_id": "ev1", "task_id": "t1", "tick_id": 1})
            msg_queue.append([f"SEMANTICS {packet}"])
            
            m9_thread = threading.Thread(target=m9_thalamus_process, args=(stop_event,))
            m9_thread.start()
            m9_thread.join(timeout=5.0)
            
            published_calls = bus_instance.publish.call_args_list
            routes_to_m6 = [call for call in published_calls if call[0][0] == "THALAMUS_M6"]
            assert len(routes_to_m6) == 0, "Cycle entered! Semantic -> M6 edge should be 0, M9 should not route semantic input back to M6!"

def test_m9_real_production_routing():
    """
    Kiểm tra khởi tạo và thuật toán định tuyến thực tế của DynamicSOM
    (Production Path) mà không dùng Mock CoreBus.
    """
    from m9_dynamic_som import DynamicSOM  # type: ignore
    logger.info("Khởi tạo DynamicSOM thực tế...")
    model = DynamicSOM(input_dim=128, initial_nodes=4)
    
    tensor_in = torch.randn(1, 128, dtype=torch.float32)
    bmu_idx, changed = model.update_and_grow(tensor_in, current_tick=1, current_episode="TEST")
    assert 0 <= bmu_idx < model.num_nodes, "BMU Index bị sai lệch"
    
    attn = {k: 0.1 for k in model.modality_names}
    attn["vision_where"] = 0.8
    dest_mask, probs, topdown = model.get_dynamic_routing(bmu_idx, attn, salience=0.9, inhibition_level=0.0)
    assert isinstance(dest_mask, dict) and any(dest_mask.values()), "Phải định tuyến tới ít nhất 1 module đích"
    assert "gaze" in topdown, "Phải sinh được tín hiệu chú ý Top-Down"

if __name__ == "__main__":
    pytest.main([__file__])
