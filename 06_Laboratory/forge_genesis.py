#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FORGE GENESIS - Canonical Bootstrap Entry Point (Single Source of Truth)
========================================================================
- Tạo checkpoint Gen 0 cho TẤT CẢ models trong architecture_manifest.json.
- Hỗ trợ: NEAT (pkl), PyTorch (pt), M0 JSON (json), M8 ART (pkl).
- Deterministic: Cùng SEED -> Cùng CHECKPOINT HASH.
- Sử dụng CheckpointManager.save() để đảm bảo format nhất quán với Runtime.
"""

import os
import sys
import json
import time
import hashlib
import random
import importlib.util
import importlib
import pickle
import binascii

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- SEED FOR DETERMINISM (P0-10: Canonical Bootstrap) ---
GENESIS_SEED = 42
random.seed(GENESIS_SEED)
try:
    import numpy as np
    np.random.seed(GENESIS_SEED)
except ImportError:
    pass
try:
    import torch
    torch.manual_seed(GENESIS_SEED)
except ImportError:
    pass

# --- IMPORTS ---
try:
    import neat
except ImportError:
    neat = None
    print("⚠️ NEAT not installed. NEAT models (M1a, M1b, M3, M4) will be skipped.")

# Import shared utilities to break circular imports
try:
    from shared_utils import MODEL_SPECS, CheckpointManager, CheckpointError
except Exception as e:
    print(f"💀 FATAL: Cannot import MODEL_SPECS or CheckpointManager from shared_utils: {e}")
    sys.exit(1)

# =========================================================================
# HELPER: Dynamic Class Loader
# =========================================================================
def load_class_from_file(file_path: str, class_name: str):
    """Load class từ file Python bất kỳ (không cần __init__.py)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location("forge_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["forge_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# =========================================================================
# MODEL FACTORY: Tạo state_dict/genome cho từng model_id
# =========================================================================
def create_genesis_state(model_id: str, spec: dict) -> tuple:
    """
    Trả về: (state_object, is_torch_bool, extra_metadata_dict)
    state_object: genome (NEAT), state_dict (Torch), hoặc dict (M0 JSON).
    """
    runtime = spec.get("runtime", "")
    topology = spec.get("topology", {})
    fmt = spec.get("checkpoint_format", "pkl")
    extra = {
        "generation": 0,
        "topology_version": 1,
        "mutation_record": "Genesis Forge (Deterministic Seed: 42)",
        "topology": topology
    }

    # ---------------------------------------------------------------------
    # 1. NEAT Models (DynamicNeatBrain) -> FORMAT PKL
    # ---------------------------------------------------------------------
    if runtime == "DynamicNeatBrain":
        if neat is None: raise RuntimeError("NEAT not available")
        config_rel = {
            "M1a_Dorsal": "01_Sensory_Inputs/M1_Eye_Vision/config-m1a-cppn",
            "M1b_Ventral": "01_Sensory_Inputs/M1_Eye_Vision/config-m1b-cppn",
            "M3_Spatial": "01_Sensory_Inputs/M3_Spatial_Parietal/config-m3",
            "M4_Motor": "02_Motor_Actuators/M4_Motor_Cortex/config-m4",
        }.get(model_id)
        
        if not config_rel: raise ValueError(f"No NEAT config mapping for {model_id}")
        config_path = os.path.join(BASE_DIR, config_rel)
        if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")

        # Fix config if needed
        with open(config_path, "r") as f: cfg = f.read()
        if "num_hidden" not in cfg and "[DefaultGenome]" in cfg:
            cfg = cfg.replace("[DefaultGenome]", "[DefaultGenome]\nnum_hidden = 0")
            with open(config_path, "w") as f: f.write(cfg)

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        pop = neat.Population(config)
        # Deterministic: Population uses internal RNG seeded by global random.seed above
        genome_id, genome = next(iter(pop.population.items()))
        genome.fitness = 0.0
        return genome, False, extra

    # ---------------------------------------------------------------------
    # 2. M0 Cerebrospinal (Neuromodulator) -> FORMAT JSON
    # ---------------------------------------------------------------------
    if model_id == "M0_Cerebrospinal":
        state = {
            "hormonal_state": {
                "dopamine": 0.5, "adrenaline": 0.1, "serotonin": 0.5,
                "cortisol": 0.1, "threat": 0.0, "novelty": 0.0,
                "salience": 0.0, "arousal": 0.1
            },
            "global_latent_state": [0.0] * 128
        }
        return state, False, extra  # is_torch=False -> CheckpointManager saves as JSON

    # ---------------------------------------------------------------------
    # 3. PyTorch / Custom Models -> FORMAT PT / PKL
    # ---------------------------------------------------------------------
    # Mapping: runtime_name -> (relative_path, class_name, init_kwargs_fn)
    MODEL_FACTORY = {
        "LiquidAuditoryNet": (
            "01_Sensory_Inputs/M2_Ear_Hearing/train_m2_esnn.py", 
            "EvolvingSNN",
            lambda top: {"num_inputs": 20, "initial_classes": 3}
        ),
        "LiquidAmygdalaNet": (
            "00_Central_Router/M5_Amygdala/train_m5_snn.py", 
            "SproutingSNN",
            lambda top: {"num_inputs": 20, "num_hidden": top.get("num_hidden", 64), "num_outputs": 2}
        ),
        "LiquidProgressiveBrain": (
            "03_Cognitive_Core/liquid_progressive_brain.py", 
            "ProgressiveLiquidBrain",
            lambda top: {
                "input_dim": 128, 
                "hidden_dim": 128 if model_id == "M6_Language" else 64, 
                "output_dim": 133 if model_id == "M6_Language" else 6,
                "name": model_id
            }
        ),
        "DynamicSOM": (
            "00_Central_Router/M9_Thalamus/m9_dynamic_som.py", 
            "DynamicSOM",
            lambda top: {"input_dim": 128, "initial_nodes": 4, "error_threshold": 15.0}
        ),
        "LiquidVocalNet": (
            "05_Vocal_System/M10_Core_Engine/m10_vocal_tract.py", 
            "VocalTractODE",
            lambda top: {"motor_dim": 3, "physical_dim": 2}
        ),
        # P0-10 FIX: M8 ART Memory Import
        "ARTMemorySystem": (
            "04_Memory_System/M8_Hippocampus/m8_art_memory.py", 
            "SimplifiedART",
            lambda top: {"vigilance": 0.85, "input_dim": 128}
        ),
    }

    if runtime not in MODEL_FACTORY:
        raise ValueError(f"Unsupported runtime '{runtime}' for model {model_id}")

    rel_path, class_name, kwargs_fn = MODEL_FACTORY[runtime]
    file_path = os.path.join(BASE_DIR, rel_path)
    
    ModelClass = load_class_from_file(file_path, class_name)
    init_kwargs = kwargs_fn(topology)
    
    # Instantiate
    model_instance = ModelClass(**init_kwargs)
    if hasattr(model_instance, "get_topology_signature"):
        extra["topology"] = model_instance.get_topology_signature()
    elif hasattr(model_instance, "get_topology_info"):
        extra["topology"] = model_instance.get_topology_info()
    elif hasattr(model_instance, "columns"):
        extra["topology"] = {"num_columns": len(model_instance.columns)}
    
    # Get state dict
    if hasattr(model_instance, 'state_dict'):
        state = model_instance.state_dict()
        is_torch = True
    else:
        # Fallback for non-Module objects (like SimplifiedART get_state)
        if hasattr(model_instance, 'get_state'):
            state = model_instance.get_state()
        else:
            state = model_instance.__dict__
        is_torch = False

    return state, is_torch, extra

# =========================================================================
# MAIN FORGE FUNCTION
# =========================================================================
def forge_real_neat():
    print("=" * 70)
    print("🧬 CANONICAL GENESIS FORGE - DETERMINISTIC BOOTSTRAP (SEED=42)")
    print("=" * 70)
    
    models_to_forge = {k: v for k, v in MODEL_SPECS.items() if v.get("checkpoint_format") and v.get("checkpoint_format") != "none"}
    
    if not models_to_forge:
        print("⚠️ No models with checkpoint_format found.")
        return

    print(f"🔍 Found {len(models_to_forge)} models to forge: {list(models_to_forge.keys())}")
    
    success_count = 0
    failed = []

    for model_id, spec in models_to_forge.items():
        print(f"\n--- [{model_id}] Format: {spec.get('checkpoint_format')} | Runtime: {spec.get('runtime')} ---")
        try:
            state_obj, is_torch, extra = create_genesis_state(model_id, spec)
            
            # Save via CheckpointManager (handles format selection, atomic write, hash verify)
            ok = CheckpointManager.save(model_id, state_obj, optimizer=None, is_torch=is_torch, extra_states=extra)
            
            if ok:
                fmt = spec.get("checkpoint_format", "pkl")
                path = os.path.join(BASE_DIR, "models", model_id, f"checkpoint.{fmt}")
                print(f"   ✅ SUCCESS -> {path}")
                success_count += 1
            else:
                print(f"   ❌ FAILED: CheckpointManager.save returned False")
                failed.append(model_id)

        except Exception as e:
            print(f"   ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed.append(model_id)

    # Summary
    print("\n" + "=" * 70)
    print(f"🎉 GENESIS COMPLETE: {success_count}/{len(models_to_forge)} OK")
    if failed:
        print(f"❌ FAILED: {failed}")
    print("=" * 70)
    
    if failed:
        sys.exit(1)

# =========================================================================
# ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    # Ensure we are in the right directory context
    os.chdir(BASE_DIR)
    forge_real_neat()