#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import torch
import pickle
import shutil
import copy

from shared_utils import BASE_DIR, MODEL_SPECS, CheckpointManager, tensor_state_hash

def detect_checkpoint_format(loaded_data):
    """
    Phân tích và bóc tách đúng model_state từ các định dạng checkpoint cũ.
    Tránh tình trạng bọc nhầm (nesting) dict trong dict.
    """
    result = {
        "model_state": None,
        "optimizer_state": None,
        "generation": 0,
        "topology_version": 1
    }
    
    if not isinstance(loaded_data, dict):
        result["model_state"] = loaded_data
        return result

    if "model_state" in loaded_data:
        result["model_state"] = loaded_data["model_state"]
        result["optimizer_state"] = loaded_data.get("optimizer_state")
        result["generation"] = loaded_data.get("generation", 0)
        result["topology_version"] = loaded_data.get("topology_version", loaded_data.get("architecture_version", 1))
        return result

    possible_model_keys = ["state_dict", "model_state_dict", "model", "weights", "network"]
    for key in possible_model_keys:
        if key in loaded_data:
            result["model_state"] = loaded_data[key]
            result["optimizer_state"] = loaded_data.get("optimizer_state", loaded_data.get("optimizer"))
            result["generation"] = loaded_data.get("generation", loaded_data.get("epoch", 0))
            return result

    result["model_state"] = loaded_data
    return result

def verify_checkpoint_structure(model_id, chkpt, is_torch):
    """Xác minh model_id, schema, generation, topology_version, key và shape trọng số trước khi sửa."""
    if not isinstance(chkpt, dict) or "model_state" not in chkpt:
        return False, "Thiếu trường model_state trong schema"
    
    recorded_id = chkpt.get("model_id")
    if recorded_id is not None and recorded_id != model_id:
        return False, f"Sai lệch model_id (Kỳ vọng: {model_id}, Thực tế: {recorded_id})"
        
    gen = chkpt.get("generation", 0)
    topo_ver = chkpt.get("topology_version", chkpt.get("architecture_version", 1))
    if not isinstance(gen, int) or gen < 0 or not isinstance(topo_ver, int) or topo_ver < 1:
        return False, f"Metadata không hợp lệ (generation={gen}, topology_version={topo_ver})"

    state = chkpt["model_state"]
    if state is None:
        return False, "model_state rỗng (None)"

    if is_torch:
        if not isinstance(state, dict) or len(state) == 0:
            return False, "PyTorch state_dict phải là dict không rỗng"
        tensor_count = 0
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                if k.endswith((".mem", ".syn")) and v.numel() == 0:
                    continue
                tensor_count += 1
                if v.numel() == 0 or not torch.isfinite(v).all():
                    return False, f"Tensor '{k}' rỗng hoặc chứa NaN/Inf"
        if tensor_count == 0:
            return False, "PyTorch state_dict không chứa bất kỳ Tensor trọng số nào"
        if model_id in ("M6_Language", "M7_Logic"):
            if "gate.weight" not in state or "out_layer.weight" not in state:
                return False, "Thiếu key bắt buộc (gate.weight / out_layer.weight) của ProgressiveLiquidBrain"
            num_cols = state["gate.weight"].shape[0]
            col_indices = {int(k.split(".")[1]) for k in state.keys() if k.startswith("columns.") and k.split(".")[1].isdigit()}
            if num_cols < 1 or len(col_indices) != num_cols:
                return False, f"Số cột trong gate.weight ({num_cols}) không khớp số cột thực tế ({len(col_indices)})"
        elif model_id == "M9_Thalamus":
            if "weights" not in state or state["weights"].dim() != 2 or state["weights"].shape[1] != 128:
                return False, "Shape ma trận SOM M9 không hợp lệ hoặc thiếu 'weights'"
            if "som_eligibility" in state and state["som_eligibility"].shape != state["weights"].shape:
                return False, f"Lệch số node giữa weights ({tuple(state['weights'].shape)}) và som_eligibility ({tuple(state['som_eligibility'].shape)})"
            if "attention_to_route" in state and state["attention_to_route"].shape != (5, 11):
                return False, f"Shape attention_to_route của M9 không hợp lệ: {tuple(state['attention_to_route'].shape)}"
    else:
        if model_id == "M8_Memory":
            if not isinstance(state, dict) or "nodes" not in state or "input_dim" not in state:
                return False, "Cấu trúc ART state của M8 không hợp lệ"
        elif model_id in ("M1a_Dorsal", "M1b_Ventral", "M3_Spatial", "M4_Motor"):
            genome = state.get("genome", state) if isinstance(state, dict) else state
            if not (hasattr(genome, "nodes") and hasattr(genome, "connections")):
                return False, "Cấu trúc NEAT genome không hợp lệ (thiếu nodes/connections)"
                
    return True, "OK"

def audit_and_quarantine(dry_run=True):
    print("="*60)
    mode_str = "DRY RUN (Chỉ Chẩn Đoán Đọc, Tuyệt Đối Không Ghi)" if dry_run else "RECOVERY MODE (Phục Hồi Có Sao Lưu)"
    print(f"🛡️ KIỂM TRA TÍNH TOÀN VẸN VÀ CÁCH LY - {mode_str} 🛡️")
    print("="*60)
    
    manifest_path = os.path.join(BASE_DIR, "brain_manifest.json")
    target_committed_gen = 0
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fm:
                val = json.load(fm).get("committed_generation", 0)
                target_committed_gen = int(val) if val is not None else 0
        except Exception:
            pass

    for model_id in MODEL_SPECS:
        path = CheckpointManager.get_canonical_path(model_id)
        tmp_path = path + ".tmp"
        if os.path.exists(tmp_path):
            print(f"⚠ [ORPHAN_TMP] {model_id}: Phát hiện file tạm mồ côi {tmp_path}")
            if not dry_run:
                shutil.move(tmp_path, tmp_path + f".quarantine_{int(time.time())}")
                
        if not os.path.exists(path):
            continue
        
        is_torch = MODEL_SPECS[model_id].get("checkpoint_format", "pkl") in ("pt", "pth")
        try:
            if path.endswith(".json") or model_id == "M0_Cerebrospinal":
                with open(path, 'r', encoding="utf-8") as f:
                    chkpt = json.load(f)
            elif is_torch:
                chkpt = torch.load(path, map_location='cpu', weights_only=False)
            else:
                with open(path, 'rb') as f:
                    chkpt = pickle.load(f)
                    
            if isinstance(chkpt, dict) and "state_hash" in chkpt and "model_state" in chkpt:
                struct_ok, reason = verify_checkpoint_structure(model_id, chkpt, is_torch)
                if not struct_ok:
                    print(f"❌ [INVALID_STRUCTURE] {model_id}: {reason}. Từ chối phục hồi!")
                    continue
                    
                actual_hash = tensor_state_hash(chkpt["model_state"])
                hash_mismatch = (actual_hash != chkpt["state_hash"])
                chkpt_gen = chkpt.get("generation", 0)
                gen_mismatch = (model_id != "M0_Cerebrospinal" and chkpt_gen != target_committed_gen)

                base_keys = {"model_state", "optimizer_state", "state_hash", "schema_version", "timestamp", "hash_version", "extra_states"}
                raw_extra = copy.deepcopy(chkpt.get("extra_states")) if isinstance(chkpt.get("extra_states"), dict) else {}
                for k, v in chkpt.items():
                    if k not in base_keys:
                        raw_extra.setdefault(k, copy.deepcopy(v))

                has_stale_tx = (
                    raw_extra.get("candidate_transaction_id") is not None
                    or bool(raw_extra.get("is_candidate", False))
                    or chkpt.get("phase") not in (None, "LIVE", "COMMITTED")
                )
                cur_topo = chkpt.get("topology") if isinstance(chkpt.get("topology"), dict) and chkpt.get("topology") else (raw_extra.get("topology") if isinstance(raw_extra.get("topology"), dict) else {})
                topo_mismatch = (
                    is_torch and model_id in ("M6_Language", "M7_Logic")
                    and isinstance(chkpt.get("model_state"), dict) and "gate.weight" in chkpt["model_state"]
                    and cur_topo.get("num_columns") != int(chkpt["model_state"]["gate.weight"].shape[0])
                )

                if hash_mismatch or gen_mismatch or has_stale_tx or topo_mismatch:
                    if hash_mismatch:
                        print(f"🚨 [HASH_MISMATCH] {model_id}: Trọng số hợp lệ nhưng Hash lệch so với thuật toán hiện tại.")
                    if gen_mismatch:
                        print(f"🚨 [GEN_MISMATCH] {model_id}: Thế hệ ({chkpt_gen}) lệch với Manifest ({target_committed_gen}).")
                    if has_stale_tx:
                        print(f"🚨 [STALE_CANDIDATE] {model_id}: Tồn tại giao dịch ứng viên chưa hoàn tất.")
                    if topo_mismatch:
                        print(f"🚨 [TOPO_MISMATCH] {model_id}: Metadata num_columns lệch với trọng số thực tế.")

                    if dry_run:
                        print(f"   -> [DRY RUN] Chỉ báo cáo, không ghi đè Checkpoint.")
                    else:
                        backup_path = path + f".backup_{int(time.time())}"
                        shutil.copy2(path, backup_path)
                        print(f"   -> 💾 Đã sao lưu trước khi phục hồi tại: {backup_path}")

                        state_to_save = chkpt["model_state"]
                        topo_ver = chkpt.get("topology_version", chkpt.get("architecture_version", 1))
                        extra = raw_extra

                        if model_id == "M8_Memory" and gen_mismatch:
                            chroma_dir = os.path.join(BASE_DIR, "models", "M8_Memory", "chroma_db")
                            if os.path.exists(chroma_dir):
                                chroma_bak = chroma_dir + f"_backup_{int(time.time())}"
                                shutil.copytree(chroma_dir, chroma_bak, dirs_exist_ok=True)
                                print(f"   -> 💾 Đã sao lưu ChromaDB tại: {chroma_bak}")
                            if extra.get("parent_art_state") and extra.get("parent_generation") == target_committed_gen:
                                state_to_save = extra["parent_art_state"]
                                topo_ver = extra.get("parent_topology_version", topo_ver)
                            extra["pending_memories"] = []

                        topo_dict = copy.deepcopy(extra.get("topology", {})) if isinstance(extra.get("topology"), dict) else {}
                        if is_torch and model_id in ("M6_Language", "M7_Logic") and "gate.weight" in state_to_save:
                            topo_dict["num_columns"] = int(state_to_save["gate.weight"].shape[0])

                        extra.update({
                            "model_id": model_id,
                            "generation": target_committed_gen if model_id != "M0_Cerebrospinal" else chkpt_gen,
                            "topology_version": topo_ver,
                            "topology": topo_dict,
                            "transaction_id": None,
                            "phase": "COMMITTED",
                            "is_candidate": False,
                            "candidate_art_state": None,
                            "candidate_transaction_id": None,
                            "mutation_record": "Healed and aligned with committed generation"
                        })
                        if CheckpointManager.save(model_id, state_to_save, chkpt.get("optimizer_state"), is_torch=is_torch, extra_states=extra):
                            print(f"   ✅ Đã phục hồi và đồng bộ {model_id} thành công!")
                        else:
                            print(f"   -> ❌ Lỗi ghi đè khi phục hồi {model_id}.")
                else:
                    print(f"✅ {model_id}: Schema V5 chuẩn. Cấu trúc, Thế hệ và Hash nguyên vẹn.")
            else:
                print(f"⚠ [SCHEMA_OLD] {model_id}: Phát hiện Schema cũ. Cần Migration.")
                extracted = detect_checkpoint_format(chkpt)
                candidate_chkpt = {
                    "model_id": model_id,
                    "model_state": extracted["model_state"],
                    "generation": extracted["generation"],
                    "topology_version": extracted["topology_version"]
                }
                struct_ok, reason = verify_checkpoint_structure(model_id, candidate_chkpt, is_torch)
                if not struct_ok:
                    print(f"❌ [INVALID_LEGACY_DATA] {model_id}: {reason}. Dừng Migration!")
                    continue
                if dry_run:
                    print(f"   -> [DRY RUN] Chỉ báo cáo, bỏ qua thao tác Migration.")
                else:
                    print(f"   -> 🛠️ Đang Migrate lên Schema V5...")
                    backup_path = path + f".backup_{int(time.time())}"
                    shutil.copy2(path, backup_path)
                    print(f"   -> 💾 Đã tạo bản sao lưu tại: {backup_path}")
                    topo_dict = {}
                    if is_torch and model_id in ("M6_Language", "M7_Logic") and "gate.weight" in extracted["model_state"]:
                        topo_dict["num_columns"] = int(extracted["model_state"]["gate.weight"].shape[0])
                    extra = {
                        "model_id": model_id,
                        "generation": target_committed_gen if model_id != "M0_Cerebrospinal" else extracted["generation"],
                        "topology_version": extracted["topology_version"],
                        "topology": topo_dict,
                        "mutation_record": "Migrated from Legacy Schema"
                    }
                    if CheckpointManager.save(model_id, extracted["model_state"], extracted["optimizer_state"], is_torch=is_torch, extra_states=extra):
                        print("   ✅ Migration thành công!")
                    else:
                        print("   -> ❌ Lỗi ghi file khi Migration.")
                
        except (pickle.UnpicklingError, EOFError, json.JSONDecodeError, RuntimeError) as e:
            print(f"❌ [CORRUPTION] {model_id}: Dữ liệu vật lý hỏng thật sự ({e})")
            if dry_run:
                print(f"   -> [DRY RUN] Bỏ qua thao tác cách ly (Quarantine).")
            else:
                quarantine_path = path + f".quarantine_{int(time.time())}"
                try:
                    shutil.move(path, quarantine_path)
                    print(f"   -> ☣️ ĐÃ CÁCH LY FILE HỎNG TẠI: {quarantine_path}")
                except Exception as move_e:
                    print(f"   -> ⚠️ Không thể cách ly file: {move_e}")

        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            print(f"📦 [DEPENDENCY_ERROR] {model_id}: Thiếu thư viện để Unpickle ({e}). Bỏ qua.")
            
        except Exception as e:
            print(f"⚠️ [UNKNOWN_ERROR] {model_id}: Lỗi đọc/ghi ({e}). BỎ QUA CÁCH LY.")

    print("\n" + "="*60)
    print("🧠 KIỂM TRA ĐỒNG BỘ TOÀN CỤC 11/11 THÙY NÃO")
    print("="*60)
    
    EXPECTED_MODELS = {
        "M1a_Dorsal", "M1b_Ventral", "M2_Ear", "M3_Spatial", 
        "M4_Motor", "M5_Amygdala", "M6_Language", "M7_Logic", 
        "M8_Memory", "M9_Thalamus", "M10_Vocal"
    }
    
    generation_set = set()
    valid_count = 0
    verified_manifest_models = {}
    
    for model_id in sorted(EXPECTED_MODELS):
        path = CheckpointManager.get_canonical_path(model_id)
        if not os.path.exists(path):
            print(f"❌ THIẾU FILE: {model_id}")
            if model_id == "M7_Logic" and not dry_run:
                print("   -> ⚠️ BẮT ĐẦU BOOTSTRAP M7_Logic TỪ KIẾN TRÚC RUNTIME...")
                try:
                    sys.path.append(os.path.join(BASE_DIR, "03_Cognitive_Core"))
                    import importlib
                    ProgressiveLiquidBrain = importlib.import_module("liquid_progressive_brain").ProgressiveLiquidBrain
                    blank_m7 = ProgressiveLiquidBrain(
                        input_dim=128,
                        hidden_dim=64,
                        output_dim=6,
                        name="M7_Prefrontal"
                    )
                    blank_m7.sprout_new_skill_column()
                    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, blank_m7.parameters()), lr=1e-3)
                    
                    sync_gen = target_committed_gen
                    actual_cols = int(blank_m7.state_dict()["gate.weight"].shape[0])
                    if hasattr(blank_m7, "get_topology_signature"):
                        actual_topo = blank_m7.get_topology_signature()
                    else:
                        actual_topo = {"num_columns": actual_cols, "input_dim": 128, "hidden_dim": 64, "output_dim": 6}
                    actual_topo["num_columns"] = actual_cols
                    extra_m7 = {
                        "model_id": "M7_Logic",
                        "generation": sync_gen,
                        "topology_version": getattr(blank_m7, "topology_version", 1),
                        "topology": actual_topo,
                        "mutation_record": "Bootstrapped from runtime architecture"
                    }
                    if CheckpointManager.save("M7_Logic", blank_m7, opt, is_torch=True, extra_states=extra_m7):
                        print(f"   ✅ Bootstrap M7_Logic thành công tại thế hệ {sync_gen} (num_columns={actual_topo['num_columns']})!")
                    else:
                        print("   -> ❌ Lỗi lưu file khi Bootstrap M7_Logic.")
                        continue
                except Exception as e:
                    print(f"   -> ❌ Bootstrap M7 thất bại: {e}")
                    continue
            else:
                continue
            
        try:
            is_torch = MODEL_SPECS[model_id].get("checkpoint_format", "pkl") in ("pt", "pth")
            if path.endswith(".json") or model_id == "M0_Cerebrospinal":
                with open(path, 'r', encoding="utf-8") as f:
                    chkpt = json.load(f)
            elif is_torch:
                chkpt = torch.load(path, map_location='cpu', weights_only=False)
            else:
                with open(path, 'rb') as f:
                    chkpt = pickle.load(f)
                    
            struct_ok, reason = verify_checkpoint_structure(model_id, chkpt, is_torch)
            if not struct_ok:
                print(f"❌ KHÔNG ĐẠT XÁC MINH CẤU TRÚC ({model_id}): {reason}")
                continue

            if is_torch and model_id in ("M6_Language", "M7_Logic"):
                extra_dict = chkpt.get("extra_states") if isinstance(chkpt.get("extra_states"), dict) else {}
                topo_meta = chkpt.get("topology") if isinstance(chkpt.get("topology"), dict) else extra_dict.get("topology", {})
                if not isinstance(topo_meta, dict) or topo_meta.get("num_columns") != int(chkpt["model_state"]["gate.weight"].shape[0]):
                    print(f"❌ KHÔNG ĐẠT METADATA TOPOLOGY ({model_id}): num_columns chưa đồng bộ!")
                    continue
                
            extra_check = chkpt.get("extra_states", {}) if isinstance(chkpt.get("extra_states"), dict) else {}
            cand_tx = chkpt.get("candidate_transaction_id") or extra_check.get("candidate_transaction_id")
            if cand_tx is not None or extra_check.get("is_candidate") or chkpt.get("phase") not in (None, "LIVE", "COMMITTED"):
                print(f"❌ CHƯA SẠCH GIAO DỊCH ỨNG VIÊN ({model_id}: tx={cand_tx}, phase={chkpt.get('phase')})")
                continue
                
            actual_hash = tensor_state_hash(chkpt["model_state"])
            if actual_hash != chkpt.get("state_hash"):
                print(f"❌ KHÔNG ĐẠT XÁC MINH HASH ({model_id}): Hash vẫn sai lệch!")
                continue
                
            gen = chkpt.get("generation", 0)
            generation_set.add(gen)
            valid_count += 1
            verified_manifest_models[model_id] = {
                "generation": gen,
                "topology_version": chkpt.get("topology_version", chkpt.get("architecture_version", 1)),
                "checkpoint_path": os.path.relpath(path, BASE_DIR).replace("\\", "/"),
                "state_hash": actual_hash
            }
        except Exception as e:
            print(f"❌ KHÔNG THỂ ĐỌC FILE: {model_id} - {e}")
            
    manifest_status = "UNKNOWN"
    committed_gen = None
    manifest_path = os.path.join(BASE_DIR, "brain_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                man_data = json.load(f)
                manifest_status = man_data.get("transaction_status", "UNKNOWN")
                committed_gen = man_data.get("committed_generation", 0)
        except Exception:
            pass

    print(f"\n📊 TỔNG KẾT:")
    print(f"   - Số model đạt chuẩn toàn diện (Schema + Shape + Hash): {valid_count}/11")
    print(f"   - Số Generation trên Checkpoint: {len(generation_set)} {list(generation_set) if generation_set else []}")
    print(f"   - Thế hệ đã Commit trên Manifest: {committed_gen}")
    print(f"   - Trạng thái Manifest: {manifest_status}")
    
    if valid_count != 11:
        print("🚨 LỖI: Hệ thống không đủ 11/11 thùy não hợp lệ!")
        return False
    if len(generation_set) != 1:
        print("🚨 LỖI: Hệ thống bị Split-Brain (Lệch pha Generation giữa các thùy não)!")
        return False
        
    unified_gen = next(iter(generation_set))
    if committed_gen is not None and unified_gen != committed_gen:
        print(f"🚨 LỖI: Thế hệ của 11 checkpoint ({unified_gen}) lệch với committed_generation của Manifest ({committed_gen})!")
        return False
        
    if not dry_run:
        print(f"🛠️ Đã xác minh đủ 11/11 model đồng nhất ở thế hệ {unified_gen}. Đang đồng bộ Manifest -> COMMITTED...")
        try:
            man_data = {}
            now_ts = time.time()
            if os.path.exists(manifest_path):
                manifest_backup = manifest_path + f".backup_{int(now_ts)}"
                shutil.copy2(manifest_path, manifest_backup)
                print(f"   -> 💾 Đã sao lưu Manifest gốc làm bằng chứng tại: {manifest_backup}")
                with open(manifest_path, "r", encoding="utf-8") as f:
                    man_data = json.load(f)
            synced_hashes = {m_id: info["state_hash"] for m_id, info in verified_manifest_models.items()}
            new_brain_hash = tensor_state_hash(synced_hashes)
            man_data["transaction_status"] = "COMMITTED"
            man_data["brain_generation"] = unified_gen
            man_data["committed_generation"] = unified_gen
            man_data["brain_hash"] = new_brain_hash
            man_data["committed_brain_hash"] = new_brain_hash
            man_data["committed_timestamp"] = now_ts
            man_data["current_transaction_id"] = None
            man_data["candidate_hashes"] = {m_id: None for m_id in verified_manifest_models}
            man_data["model_hashes"] = synced_hashes
            man_data["models"] = verified_manifest_models
            if isinstance(man_data.get("metadata"), dict):
                man_data["metadata"]["last_updated"] = now_ts
            with open(manifest_path, "w", encoding="utf-8") as fw:
                json.dump(man_data, fw, indent=4)
            manifest_status = "COMMITTED"
        except Exception as e:
            print(f"🚨 LỖI: Không thể sửa chữa Manifest: {e}")
            return False
                
    if manifest_status != "COMMITTED":
        print(f"🚨 LỖI: Manifest vẫn ở trạng thái không an toàn ({manifest_status})!")
        return False
        
    print("✅ HỆ THỐNG ĐỒNG BỘ 100%. SẴN SÀNG BOOT.")
    return True

if __name__ == "__main__":
    is_dry_run = "--active" not in sys.argv
    is_valid = audit_and_quarantine(dry_run=is_dry_run)
    if not is_valid:
        print("\n💀 KIỂM TRA TOÀN DIỆN THẤT BẠI. DỪNG HỆ THỐNG ĐỂ TRÁNH HƯ HỎNG DỮ LIỆU.")
        sys.exit(1)
    else:
        sys.exit(0)