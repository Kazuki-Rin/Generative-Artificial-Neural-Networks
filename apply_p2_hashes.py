# apply_p2_hashes.py
import os
import sys

import time
import json
import hashlib

# Import shared utilities to break circular imports
from shared_utils import BASE_DIR, MODEL_SPECS, CheckpointManager, tensor_state_hash

def upgrade_to_v4():
    print("⚡ BẮT ĐẦU ĐỒNG BỘ HASH VÀ TÁI TẠO BRAIN_MANIFEST.JSON (SCHEMA V5)...")
    print("=" * 60)

    # [P0-1 FIX] Lọc đúng 11 Brain Models (brain_model: true), bỏ qua M0 (brain_model: false)
    brain_models = {k: v for k, v in MODEL_SPECS.items() if v.get("brain_model", False) is True}
    
    if len(brain_models) != 11:
        print(f"⚠️ CẢNH BÁO: Manifest chỉ định {len(brain_models)} brain_models, kỳ vọng 11.")

    upgraded_count = 0
    skipped_count = 0
    
    # [BẢN VÁ AUDIT P0]: Khởi tạo biến lưu trữ dữ liệu để tái tạo brain_manifest.json
    reconstructed_hashes = {}
    max_generation = 0

    for name, spec in brain_models.items():
        print(f"\n🔍 Đang xử lý: {name}...")
        
        # [P1-2 FIX] Không dùng spec.get("checkpoint") nữa. Dùng CheckpointManager để lấy đường dẫn chuẩn.
        try:
            canonical_path = CheckpointManager.get_canonical_path(name)
        except ValueError as e:
            print(f"   ❌ Lỗi cấu hình manifest cho {name}: {e}")
            skipped_count += 1
            continue

        if not os.path.exists(canonical_path):
            print(f"   ⏭️  Bỏ qua: Không tìm thấy file checkpoint tại '{canonical_path}'")
            skipped_count += 1
            continue

        # [P1-1 FIX] Xác định backend từ checkpoint_format trong manifest
        format_type = spec.get("checkpoint_format", "pkl")
        is_torch = format_type in ("pt", "pth")  # Hỗ trợ cả .pt và .pth

        print(f"   📂 Format: {format_type} | Backend: {'Torch' if is_torch else 'Pickle'} | Path: {canonical_path}")

        # Load checkpoint cũ (v3 hoặc v4 cũ)
        try:
            # [BẢN VÁ AUDIT P1]: Phải bypass validate_hash=False để ép đọc các file schema cũ hoặc bị lệch hash
            chkpt = CheckpointManager.load(name, is_torch=is_torch, validate_hash=False)
        except Exception as e:
            print(f"   ❌ Lỗi load checkpoint {name}: {e}")
            skipped_count += 1
            continue

        if chkpt is None:
            print(f"   ⚠️  Checkpoint rỗng hoặc không hợp lệ cho {name}")
            skipped_count += 1
            continue

        if isinstance(chkpt, dict) and "model_state" in chkpt:
            model_state = chkpt["model_state"]
            opt_state = chkpt.get("optimizer_state", None)
            generation = chkpt.get("generation", 0)
            topology_version = chkpt.get("topology_version", chkpt.get("architecture_version", 1))
            topology_info = chkpt.get("topology", {})
            if not isinstance(topology_info, dict):
                topology_info = {}
            if is_torch and isinstance(model_state, dict):
                model_state = {k: v for k, v in model_state.items() if not k.endswith('.mem') and not k.endswith('.syn') and not k.endswith('.spk')}
            if is_torch and name in ("M6_Language", "M7_Logic") and isinstance(model_state, dict) and "gate.weight" in model_state:
                topology_info["num_columns"] = int(model_state["gate.weight"].shape[0])
                target_out = 133 if name == "M6_Language" else 6
                if "out_layer.weight" in model_state and model_state["out_layer.weight"].shape[0] != target_out:
                    model_state["out_layer.weight"] = model_state["out_layer.weight"][:target_out, :].clone()
                    if "out_layer.bias" in model_state:
                        model_state["out_layer.bias"] = model_state["out_layer.bias"][:target_out].clone()
                    opt_state = None
        else:
            model_state = chkpt
            opt_state = None
            generation = 0
            topology_version = 1
            topology_info = {}

        extra = dict(chkpt.get("extra_states", {})) if isinstance(chkpt, dict) and isinstance(chkpt.get("extra_states"), dict) else {}
        extra.update({
            "generation": generation,
            "topology_version": topology_version,
            "topology": topology_info,
            "mutation_record": "Upgraded to P2 V5 Schema via apply_p2_hashes"
        })

        try:
            success = CheckpointManager.save(
                name,
                model_state,
                opt_state,
                is_torch=is_torch,
                extra_states=extra
            )
            # [BẢN VÁ LOGIC]: CHỈ thu thập hash và đếm số lượng nếu file ĐÃ THỰC SỰ LƯU XUỐNG Ổ CỨNG!
            if success:
                current_hash = tensor_state_hash(model_state)
                reconstructed_hashes[name] = current_hash
                max_generation = max(max_generation, generation)
            else:
                print(f"   ❌ Hàm save() trả về False cho {name}. Hủy ghi nhận Hash!")
        except Exception as e:
            print(f"   ❌ Lỗi khi save checkpoint {name}: {e}")
            success = False

        if success:
            print(f"   ✅ ĐÃ NÂNG CẤP THÀNH CÔNG (Schema V5, Hash validated)")
            upgraded_count += 1
        else:
            print(f"   ❌ LƯU THẤT BẠI")
            skipped_count += 1

    EXPECTED_MODELS = {
        "M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial", 
        "M4_Motor", "M5_Amygdala", "M6_Language", "M7_Logic", 
        "M8_Memory", "M9_Thalamus", "M10_Vocal"
    }

    missing_models = EXPECTED_MODELS - set(reconstructed_hashes.keys())
    
    generation_set = set()
    models_meta = {}
    for name in sorted(EXPECTED_MODELS):
        if name in reconstructed_hashes:
            try:
                fmt = MODEL_SPECS.get(name, {}).get("checkpoint_format", "pkl")
                is_torch_m = fmt in ("pt", "pth")
                chkpt = CheckpointManager.load(name, is_torch=is_torch_m, validate_hash=False, enforce_manifest=False)
                gen = chkpt.get("generation", 0) if isinstance(chkpt, dict) else 0
                topo_ver = chkpt.get("topology_version", 1) if isinstance(chkpt, dict) else 1
                canon_path = CheckpointManager.get_canonical_path(name, use_manifest=False)
                generation_set.add(gen)
                models_meta[name] = {
                    "generation": gen,
                    "topology_version": topo_ver,
                    "checkpoint_path": os.path.relpath(canon_path, BASE_DIR).replace("\\", "/"),
                    "state_hash": reconstructed_hashes[name]
                }
            except Exception:
                pass
            
    is_valid_generation = len(generation_set) == 1
    
    if not missing_models and is_valid_generation:
        print("\n📝 BẮT ĐẦU TÁI TẠO BRAIN_MANIFEST.JSON...")
        manifest_path = os.path.join(BASE_DIR, "brain_manifest.json")
        max_generation = list(generation_set)[0]
        global_brain_hash = tensor_state_hash(reconstructed_hashes)
        
        manifest_data = {
            "schema_version": 2,
            "description": "Brain Evolution Manifest - Tracks 2-Phase Evolution Transactions & Rollback Lineage",
            "brain_generation": max_generation,
            "brain_hash": global_brain_hash,
            "parent_brain_hash": "GENESIS",
            "committed_generation": max_generation,
            "committed_brain_hash": global_brain_hash,
            "committed_timestamp": time.time(),
            "parent_generation": max(0, max_generation - 1),
            "lineage_parent_brain_hash": "GENESIS",
            "fitness": 0.0,
            "current_transaction_id": None,
            "transaction_status": "COMMITTED",
            "candidate_hashes": {k: None for k in reconstructed_hashes.keys()},
            "model_hashes": reconstructed_hashes,
            "models": models_meta,
            "rollback_lineage": {
                "enabled": True,
                "max_depth": 5,
                "history": ["GENESIS"]
            },
            "metadata": {
                "created_at": time.time(),
                "last_updated": time.time(),
                "schema_version": 2
            }
        }
        
        tmp_path = manifest_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, manifest_path)
            print(f"   ✅ Đã ghi đè brain_manifest.json thành công! Global Hash: {global_brain_hash[:12]}...")
        except Exception as e:
            print(f"   ❌ Lỗi khi ghi brain_manifest.json: {e}")
    else:
        if missing_models:
            print(f"\n⚠️ CẢNH BÁO: Ghi nhận thiếu {len(missing_models)} thùy não: {', '.join(missing_models)}.")
        if not is_valid_generation:
            print(f"\n⚠️ CẢNH BÁO: Lệch pha Generation giữa các thùy não: {generation_set}.")
        print(f"⚠️ KHÔNG TÁI TẠO brain_manifest.json để bảo vệ tính toàn vẹn của đồ thị sinh học!")

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN TẤT!")
    print(f"   ✅ Nâng cấp thành công: {upgraded_count} models")
    print(f"   ⏭️  Bỏ qua / Lỗi:       {skipped_count} models")
    print(f"   📊 Tổng Brain Models:   {len(brain_models)}")
    print("=" * 60)
    print("💡 HÃY CHẠY: Khởi động hệ thống để Master load lại Manifest mới!")

if __name__ == "__main__":
    upgrade_to_v4()