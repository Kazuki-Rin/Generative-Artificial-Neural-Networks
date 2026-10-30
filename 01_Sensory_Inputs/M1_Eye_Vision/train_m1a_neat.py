#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train M1a_Dorsal (NEAT) - "Where" Pathway
Input: 20D retinal vector (5x4 downsampled)
Output: 2D predicted position (x, y)
Checkpoint: models/M1a_Dorsal/checkpoint.pkl (Schema V5)

This trainer creates a canonical checkpoint compatible with DynamicNeatBrain wrapper
used in main_life_loop.py for runtime evolution.
"""

import os
import sys
import neat
import math
import random
import pickle

# --- THÊM KHỐI NÀY ĐỂ ĐỊNH TUYẾN THƯ MỤC GỐC ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import shared utilities to break circular imports
from shared_utils import BASE_DIR, CheckpointManager

def eval_genomes(genomes, config):
    """Hàm đánh giá fitness cho Recurrent NEAT: Dự đoán tọa độ đốm sáng."""
    for genome_id, genome in genomes:
        # [BẢN VÁ P0 - LỖI SỐ 1]: Đổi thành RecurrentNetwork để khớp với Runtime
        net = neat.nn.RecurrentNetwork.create(genome, config)
        fitness = 0.0
        
        # 5 bài toán ngẫu nhiên mỗi generation
        for _ in range(5):
            targ_x, targ_y = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
            
            # Tạo input 20D: background noise + 1 hotspot
            inputs = [random.uniform(0.0, 0.1) for _ in range(20)]
            col = min(4, int(targ_x * 5))
            row = min(3, int(targ_y * 4))
            idx = row * 5 + col
            if 0 <= idx < 20:
                inputs[idx] = 1.0
            
            try:
                # [BẢN VÁ P0 - LỖI SỐ 2]: Reset state và chạy 3 time-steps để ổn định Recurrent loops
                net.reset()
                output = None
                for _ in range(3):
                    output = net.activate(inputs)
                
                if output and len(output) >= 2:
                    pred_x, pred_y = output[0], output[1]
                    pred_x = max(0.0, min(1.0, pred_x))
                    pred_y = max(0.0, min(1.0, pred_y))
                    
                    error = math.sqrt((pred_x - targ_x)**2 + (pred_y - targ_y)**2)
                    fitness += max(0.0, 2.0 - error)
                else:
                    fitness -= 1.0
            except Exception as e:
                import logging
                logging.warning(f"M1a Eval Error: {e}")
                fitness -= 1.0
                
        genome.fitness = fitness

def run(config_path):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    if config.genome_config.num_outputs != 2:
        print(f"⚠️ CẢNH BÁO: Config num_outputs={config.genome_config.num_outputs}, Manifest yêu cầu 2.")

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("🚀 Khởi động Trạm Huấn luyện M1a_Dorsal (RECURRENT NEAT - Where Pathway)...")
    print(f"   Config: {config_path}")
    print(f"   Inputs: {config.genome_config.num_inputs} | Outputs: {config.genome_config.num_outputs}")

    pretraining_gens = 15
    winner = p.run(eval_genomes, pretraining_gens)

    print(f"\n🏆 Champion Fitness: {winner.fitness:.4f}")
    print(f"   Nodes: {len(winner.nodes)} | Connections: {len(winner.connections)}")

    try:
        canonical_path = CheckpointManager.get_canonical_path("M1a_Dorsal")
        print(f"   💾 Đang lưu Canonical Checkpoint (Schema V5): {canonical_path}")

        extra = {
            "generation": 0, 
            "pretraining_generation": pretraining_gens,
            "topology_version": 1,
            "mutation_record": f"Recurrent NEAT Pretraining ({pretraining_gens} gens)",
            "topology": {
                "num_inputs": config.genome_config.num_inputs,
                "num_outputs": config.genome_config.num_outputs,
                "num_nodes": len(winner.nodes),
                "num_connections": len(winner.connections)
            },
            "transaction_state": {
                "current_transaction_id": "",
                "status": "COMMITTED",
                "candidate_hash": None,
                "staged_topology_version": 0
            }
        }

        # Lưu thẳng Genome Object. Thằng DynamicNeatBrain trong main_life_loop 
        # sẽ load cái này lên và tự động wrap nó thành live_state hợp lệ cho luồng 2PC.
        success = CheckpointManager.save(
            "M1a_Dorsal",
            winner,              
            optimizer=None,      
            is_torch=False,      
            extra_states=extra
        )

        if success:
            print(f"   ✅ Lưu thành công Schema V5 (Hash validated). Sẵn sàng cho 2-Phase Commit!")
        else:
            print(f"   ❌ Lưu thất bại!")

    except Exception as e:
        print(f"   💀 LỖI NGHIÊM TRỌNG khi lưu checkpoint: {e}")
        import traceback; traceback.print_exc()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-m1a-cppn')
    
    if not os.path.exists(config_path):
        print(f"❌ Không tìm thấy config: {config_path}")
        sys.exit(1)
        
    run(config_path)