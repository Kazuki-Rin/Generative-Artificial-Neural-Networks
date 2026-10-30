#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train M4_Motor (NEAT) - Motor Control
Input: 10D [curr_x, curr_y, targ_x, targ_y, padding...]
Output: 3D [dx, dy, click]
Checkpoint: models/M4_Motor/checkpoint.pkl (Schema V5)

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
    """Hàm đánh giá fitness cho Recurrent NEAT: Điều khiển tay vươn tới mục tiêu."""
    for genome_id, genome in genomes:
        # [BẢN VÁ P0 - LỖI SỐ 1]: Đổi thành RecurrentNetwork để khớp với Runtime
        net = neat.nn.RecurrentNetwork.create(genome, config)
        fitness = 0.0
        
        # 5 bài toán ngẫu nhiên mỗi generation
        for _ in range(5):
            curr_x, curr_y = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
            targ_x, targ_y = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
            
            inputs = [curr_x, curr_y, targ_x, targ_y] + [0.0] * 6
            
            try:
                # [BẢN VÁ P0 - LỖI SỐ 2]: Reset state và chạy 3 time-steps để ổn định Recurrent loops
                net.reset()
                output = None
                for _ in range(3):
                    output = net.activate(inputs)
                
                if output and len(output) >= 3:
                    dx, dy, click = output[0], output[1], output[2]
                    
                    pred_x = curr_x + dx
                    pred_y = curr_y + dy
                    
                    pred_x = max(0.0, min(1.0, pred_x))
                    pred_y = max(0.0, min(1.0, pred_y))
                    
                    error = math.sqrt((pred_x - targ_x)**2 + (pred_y - targ_y)**2)
                    
                    fitness += max(0.0, 1.0 - error)
                    
                    if click > 0.5 and error < 0.1:
                        fitness += 0.5
                else:
                    fitness -= 1.0  
            except Exception as e:
                import logging
                logging.warning(f"M4 Eval Error: {e}")
                fitness -= 1.0
                
        genome.fitness = fitness

def run(config_path):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    if config.genome_config.num_inputs != 10:
        print(f"❌ LỖI CẤU HÌNH: Config num_inputs={config.genome_config.num_inputs}, Trainer yêu cầu 10.")
        sys.exit(1)
    if config.genome_config.num_outputs != 3:
        print(f"❌ LỖI CẤU HÌNH: Config num_outputs={config.genome_config.num_outputs}, Trainer yêu cầu 3 (dx, dy, click).")
        sys.exit(1)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("🚀 Khởi động Trạm Huấn luyện M4_Motor (RECURRENT NEAT - Motor Control)...")
    print(f"   Config: {config_path}")
    print(f"   Inputs: {config.genome_config.num_inputs} | Outputs: {config.genome_config.num_outputs}")

    pretraining_gens = 15
    winner = p.run(eval_genomes, pretraining_gens)

    print(f"\n🏆 Champion Fitness: {winner.fitness:.4f}")
    print(f"   Nodes: {len(winner.nodes)} | Connections: {len(winner.connections)}")

    try:
        canonical_path = CheckpointManager.get_canonical_path("M4_Motor")
        print(f"   💾 Đang lưu Canonical Checkpoint (Schema V5): {canonical_path}")

        extra = {
            "generation": 0,
            "pretraining_generation": pretraining_gens,
            "topology_version": 1,
            "mutation_record": f"Recurrent NEAT Pretraining ({pretraining_gens} gens) - Reaching Task",
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

        # Lưu thẳng Genome Object cho luồng 2PC.
        success = CheckpointManager.save(
            "M4_Motor",
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
    config_path = os.path.join(local_dir, 'config-m4')
    
    if not os.path.exists(config_path):
        print(f"❌ Không tìm thấy config: {config_path}")
        sys.exit(1)
        
    run(config_path)