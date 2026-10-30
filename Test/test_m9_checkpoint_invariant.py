from loguru import logger
import sys
import os
import torch

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "00_Central_Router", "M9_Thalamus"))
from m9_dynamic_som import DynamicSOM  # type: ignore


def test_checkpoint_invariant():
    # 1. Construct/load a valid M9 model
    model = DynamicSOM(input_dim=128, initial_nodes=4)
    model._apply_routing_constraints()
    
    # 2. Confirm: attention_to_route[0,5] == 0
    assert model.attention_to_route[0, 5] == 0.0, "Initial invariant broken"
    
    # 3. Construct a deliberately corrupted checkpoint whose matrix has attention_to_route[0,5] = nonzero_value
    state_dict = model.state_dict()
    state_dict['attention_to_route'][0, 5] = 9.99
    
    # Check that another legitimate weight is kept
    state_dict['attention_to_route'][1, 1] = 8.88
    
    # 4. Load it through the SAME production checkpoint-loading function
    # In PyTorch, this is `load_state_dict`
    model.load_state_dict(state_dict)
    
    # 5. After loading, assert attention_to_route[0,5] == 0
    assert model.attention_to_route[0, 5] == 0.0, "Invariant bypassed by load_state_dict!"
    
    # 6. Also verify other legitimate routing weights survive unchanged
    assert model.attention_to_route[1, 1] == 8.88, "Other weights were corrupted!"
    
    print("test_checkpoint_invariant: PASS")

if __name__ == "__main__":
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    except ImportError:
        pass
        
    test_checkpoint_invariant()
