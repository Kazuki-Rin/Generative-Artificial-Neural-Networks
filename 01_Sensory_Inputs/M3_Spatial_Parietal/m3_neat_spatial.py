#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train M3_Spatial (NEAT) - Spatial Prediction
Input: 10D history (5 previous x,y positions)
Output: 2D predicted next position (x, y)
Checkpoint: models/M3_Spatial/checkpoint.pkl (Schema V5)

This trainer creates a canonical checkpoint compatible with DynamicNeatBrain wrapper
used in main_life_loop.py for runtime evolution.
"""

import os
import sys
import neat
import math
import random
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from shared_utils import BASE_DIR, CheckpointManager

def eval_genomes(genomes, config):
    """Hàm đánh giá fitness cho Recurrent NEAT: Dự đoán vị trí tiếp theo từ lịch sử đa phương thức."""
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        fitness = 0.0
        
        for _ in range(5):
            x = random.uniform(0.2, 0.8)
            y = random.uniform(0.2, 0.8)
            vx = random.uniform(-0.1, 0.1)
            vy = random.uniform(-0.1, 0.1)
            motor_dx = random.uniform(-1.0, 1.0)
            motor_dy = random.uniform(-1.0, 1.0)
            m3_focus = random.uniform(0.0, 1.0)
            
            inputs = [x, y, vx, vy, motor_dx, motor_dy, 0.0, 0.0, m3_focus, 0.0]
            
            true_next_x = max(0.0, min(1.0, x + vx + (motor_dx * 0.1 * m3_focus)))
            true_next_y = max(0.0, min(1.0, y + vy + (motor_dy * 0.1 * m3_focus)))
            
            try:
                net.reset()
                output = None
                for _ in range(3):
                    output = net.activate(inputs)

                if output and len(output) >= 2:
                    pred_x, pred_y = output[0], output[1]
                    
                    pred_x = max(0.0, min(1.0, pred_x))
                    pred_y = max(0.0, min(1.0, pred_y))
                    
                    error = math.sqrt((pred_x - true_next_x)**2 + (pred_y - true_next_y)**2)
                    fitness += max(0.0, 2.0 - error)
                else:
                    fitness -= 1.0
            except Exception as e:
                import logging
                logging.warning(f"M3 Eval Error: {e}")
                fitness -= 1.0
                
        genome.fitness = fitness

def run(config_path):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    if config.genome_config.num_outputs != 2:
        print(f"❌ LỖI CẤU HÌNH: Config num_outputs={config.genome_config.num_outputs}, Manifest & Trainer yêu cầu 2.")
        sys.exit(1)

    if config.genome_config.num_inputs != 10:
        print(f"⚠️ CẢNH BÁO: Config num_inputs={config.genome_config.num_inputs}, Trainer dùng 10 (history 5 steps).")

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("🚀 Khởi động Trạm Huấn luyện M3_Spatial (RECURRENT NEAT - Spatial Prediction)...")
    print(f"   Config: {config_path}")
    print(f"   Inputs: {config.genome_config.num_inputs} | Outputs: {config.genome_config.num_outputs}")

    pretraining_gens = 15
    winner = p.run(eval_genomes, pretraining_gens)

    print(f"\n🏆 Champion Fitness: {winner.fitness:.4f}")
    print(f"   Nodes: {len(winner.nodes)} | Connections: {len(winner.connections)}")

    try:
        canonical_path = CheckpointManager.get_canonical_path("M3_Spatial")
        print(f"   💾 Đang lưu Canonical Checkpoint (Schema V5): {canonical_path}")

        extra = {
            "generation": 0,
            "pretraining_generation": pretraining_gens,
            "topology_version": 1,
            "mutation_record": f"Recurrent NEAT Pretraining ({pretraining_gens} gens) - Spatial Trajectory Prediction",
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
            "M3_Spatial",
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
    config_path = os.path.join(local_dir, 'config-m3')
    
    if not os.path.exists(config_path):
        print(f"❌ Không tìm thấy config: {config_path}")
        sys.exit(1)
        
    run(config_path)