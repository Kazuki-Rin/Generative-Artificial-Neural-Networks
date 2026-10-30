import os
import sys
import threading
import json
import torch
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
from main_life_loop import m9_thalamus_process

def test_evolution_causal_chain():
    """
    SELF-EVOLUTION - Prove causal runtime chain.
    """
    stop_event = threading.Event()
    
    with patch("main_life_loop.CoreBus") as MockBus, patch("zmq.Context"), patch("zmq.Poller") as MockPoller, patch("main_life_loop.CheckpointManager.save"), patch("main_life_loop.load_worker_checkpoint_strict", return_value=(None, "DEGRADED")), patch("signal.signal"), patch("main_life_loop._publish_telemetry"):
        bus_instance = MockBus.return_value
        bus_instance.node_id = "M9_Thalamus"
        bus_instance._last_telemetry_time = 0.0
        bus_instance.ctrl_socket = MagicMock()
        import m9_dynamic_som  # type: ignore
        captured_model = []
        original_update_and_grow = m9_dynamic_som.DynamicSOM.update_and_grow
        def HookedUpdate(*args, **kwargs):
            if args[0] not in captured_model:
                captured_model.append(args[0])
            return original_update_and_grow(*args, **kwargs)
            
        with patch("m9_dynamic_som.DynamicSOM.update_and_grow", new=HookedUpdate):
            msg_queue = []
            
            mock_poller_instance = MockPoller.return_value
            fake_sock = MagicMock()
            poll_state = {"consumed_in_tick": False}

            def side_effect_poll(*args, **kwargs):
                poll_state["consumed_in_tick"] = False
                if not msg_queue:
                    stop_event.set()
                return [(fake_sock, zmq.POLLIN), (bus_instance.ctrl_socket, zmq.POLLIN)]

            mock_poller_instance.poll.side_effect = side_effect_poll

            def mock_get_all_msgs_dual(sock, bus):
                if poll_state["consumed_in_tick"]:
                    return []
                sock_type = "ctrl" if sock == bus_instance.ctrl_socket else "data"
                if msg_queue and msg_queue[0][0] == sock_type:
                    poll_state["consumed_in_tick"] = True
                    return msg_queue.pop(0)[1]
                return []
                    
            with patch("main_life_loop.get_all_msgs", side_effect=mock_get_all_msgs_dual):
                far_input_packet = json.dumps({
                    "payload": {"latent_vector": [100.0] * 128}, 
                    "event_id": "ev1", 
                    "task_id": "t1",
                    "tick_id": 1,
                    "episode_id": "TEST"
                })
                
                prepare_cmd = json.dumps({
                    "type": "CONTROL",
                    "command": "EVOLUTION_PREPARE", 
                    "transaction_id": "tx1",
                    "payload": {
                        "transaction_id": "tx1",
                        "target_tick": 1,
                        "target_episode": "TEST"
                    },
                    "target_node": "M9_Thalamus",
                    "tick_id": 1,
                    "episode_id": "TEST"
                })
                
                commit_cmd = json.dumps({
                    "type": "CONTROL",
                    "command": "GLOBAL_COMMIT",
                    "transaction_id": "tx1",
                    "payload": {"transaction_id": "tx1"},
                    "target_node": "M9_Thalamus",
                    "tick_id": 1,
                    "episode_id": "TEST"
                })
                
                far_input_packet2 = json.dumps({
                    "payload": {"latent_vector": [100.0] * 128}, 
                    "event_id": "ev2", 
                    "task_id": "t2",
                    "tick_id": 2,
                    "episode_id": "TEST"
                })
                
                awake_cmd = json.dumps({"state": "AWAKE"})
                msg_queue.append(("ctrl", [f"SYSTEM_CONTROL {awake_cmd}"]))
                msg_queue.append(("data", [f"RAW_VISION_M1A {far_input_packet}"]))
                msg_queue.append(("ctrl", [f"SYSTEM_CONTROL {prepare_cmd}"]))
                msg_queue.append(("ctrl", [f"SYSTEM_CONTROL {commit_cmd}"]))
                msg_queue.append(("data", [f"RAW_VISION_M1A {far_input_packet2}"]))
                
                m9_thread = threading.Thread(target=m9_thalamus_process, args=(stop_event,))
                m9_thread.start()
                m9_thread.join(timeout=5.0)
                
                assert len(captured_model) > 0, "Model was not instantiated"
                model = captured_model[0]
                
                assert model.num_nodes == 5, f"Topology did not grow! Expected 5, got {model.num_nodes}"
                
                published_calls = bus_instance.publish.call_args_list
                routing_payloads = [call[0][1] for call in published_calls if call[1].get('modality') == 'routing']
                
                bmus = [p["neural_focus"][0] for p in routing_payloads]
                assert len(bmus) >= 2, f"Should have evaluated routing at least twice, got {len(bmus)}"
                assert bmus[-1] == 4.0, f"Final routing didn't use the evolved node! BMUs were {bmus}"
                
                print("test_evolution_causal_chain: PASS")

if __name__ == "__main__":
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    except ImportError:
        pass
    test_evolution_causal_chain()
