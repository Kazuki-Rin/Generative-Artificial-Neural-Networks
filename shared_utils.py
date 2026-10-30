#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared Utilities Module
========================
Breaks circular imports by providing shared utilities:
- CheckpointManager (Schema V5 - Transactional & Atomic)
- tensor_state_hash (BẢN VÁ P0 - Ổn định hóa)
- topology_signature_hash
- MODEL_SPECS loader
- BASE_DIR constant
"""

import os
import sys
import time
import json
import hashlib
import pickle
import shutil
import torch
import numpy as np
from loguru import logger

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'): return obj.tolist()
        if hasattr(obj, 'item'): return obj.item()
        if isinstance(obj, set): return list(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

# =========================================================================
# BASE_DIR - Root of the project
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# =========================================================================
# MODEL_SPECS Loader
# =========================================================================
def load_model_specs():
    manifest_path = os.path.join(BASE_DIR, "architecture_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                arch = json.load(f)
                return arch.get("modules", {})
        except Exception as e:
            logger.warning(f"Could not load MODEL_SPECS: {e}")
    return {}

MODEL_SPECS = load_model_specs()

# =========================================================================
# tensor_state_hash - Compute SHA256 hash of model state
# =========================================================================
HASH_ALGORITHM_VERSION = "v1"

def tensor_state_hash(state, version=HASH_ALGORITHM_VERSION):
    if hasattr(state, "state_dict") and callable(state.state_dict):
        state = state.state_dict()
    h = hashlib.sha256()
    if version in ("v1", "v1_raw_pickle"):
        h.update(HASH_ALGORITHM_VERSION.encode('utf-8'))
    try:
        if version in ("v1", "v0") and isinstance(state, dict) and any(torch.is_tensor(v) for v in state.values() if isinstance(v, torch.Tensor)):
            normalized_state = {}
            for k in sorted(state.keys(), key=str):
                v = state[k]
                if torch.is_tensor(v):
                    normalized_state[k] = v.detach().clone().cpu().to(torch.float32).contiguous().numpy()
                else:
                    normalized_state[k] = v
            blob = pickle.dumps(normalized_state, protocol=4)
            h.update(blob)
        elif version == "legacy_tensor_bytes" and isinstance(state, dict):
            for k in sorted(state.keys(), key=str):
                h.update(str(k).encode('utf-8'))
                v = state[k]
                if torch.is_tensor(v):
                    h.update(v.detach().cpu().numpy().tobytes())
                else:
                    h.update(pickle.dumps(v, protocol=4))
        else:
            blob = pickle.dumps(state, protocol=4)
            h.update(blob)
    except Exception as e:
        logger.warning(f"Lỗi Pickling khi Hash {type(state)}: {e}. Fallback sang chuỗi hóa.")
        try:
            safe_dict = {str(k): v for k, v in state.items()} if isinstance(state, dict) else state
            state_str = json.dumps(safe_dict, sort_keys=True, cls=SafeJSONEncoder) if isinstance(safe_dict, dict) else str(state)
        except Exception as e2:
            logger.warning(f"Lỗi chuỗi hóa JSON khi Hash: {e2}. Fallback cuối cùng sang chuỗi thô (Raw String).")
            state_str = str(state)
        h.update(state_str.encode('utf-8'))
    return h.hexdigest()

def verify_state_hash(state, expected_hash, recorded_version=None):
    """
    Kiểm tra state_hash và nhận diện phiên bản hash cũ ("v1_raw_pickle", "v0", "legacy_raw", "legacy_tensor_bytes").
    Tuyệt đối không tự ý ghi đè hay bỏ qua khi hash không khớp phiên bản hiện hành.
    """
    target_ver = recorded_version or HASH_ALGORITHM_VERSION
    actual_v1 = tensor_state_hash(state, version="v1")
    if expected_hash == actual_v1 and target_ver == "v1":
        return True, "v1", actual_v1

    for legacy_ver in ("v1_raw_pickle", "v0", "legacy_raw", "legacy_tensor_bytes"):
        if expected_hash == tensor_state_hash(state, version=legacy_ver):
            if recorded_version == legacy_ver:
                return True, legacy_ver, actual_v1
            return False, f"LEGACY_HASH_DETECTED({legacy_ver})", actual_v1

    return False, "HASH_CORRUPTION_OR_MODIFIED_WEIGHTS", actual_v1

# =========================================================================
# topology_signature_hash - Compute SHA256 of Model Structure
# =========================================================================
def topology_signature_hash(topology_dict):
    """
    Tạo mã băm chữ ký cấu trúc mạng.
    Bao gồm các keys như: num_nodes, num_columns, hidden_dim, edges, layer_sizes...
    """
    if not topology_dict or not isinstance(topology_dict, dict):
        return "UNKNOWN_TOPOLOGY"
        
    try:
        str_key_topo = {str(k): v for k, v in topology_dict.items()}
        sorted_topo = {k: str_key_topo[k] for k in sorted(str_key_topo.keys())}
        topo_str = json.dumps(sorted_topo, sort_keys=True, cls=SafeJSONEncoder)
    except Exception as e:
        logger.warning(f"Lỗi JSON khi hash topology: {e}. Fallback sang chuỗi thô.")
        topo_str = str(topology_dict)
        
    h = hashlib.sha256(topo_str.encode('utf-8'))
    return f"topo_{h.hexdigest()[:12]}"


# =========================================================================
# CheckpointManager - Schema V5 (Atomic & Strict Consistency)
# =========================================================================
class CheckpointError(Exception):
    pass

class CheckpointManager:
    @staticmethod
    def get_active_manifest_info(model_id=None):
        manifest_path = os.path.join(BASE_DIR, "brain_manifest.json")
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                man_data = json.load(f)
            info = {
                "transaction_status": man_data.get("transaction_status", "UNKNOWN"),
                "committed_generation": man_data.get("committed_generation", 0),
                "current_transaction_id": man_data.get("current_transaction_id"),
                "model_entry": man_data.get("models", {}).get(model_id) if model_id else None
            }
            return info
        except Exception as e:
            logger.warning(f"Không thể đọc brain_manifest.json: {e}")
            return None

    @staticmethod
    def get_canonical_path(model_id, use_manifest=True):
        specs = MODEL_SPECS
        format_type = specs.get(model_id, {}).get("checkpoint_format", "pkl")
        if model_id == "M0_Cerebrospinal":
            format_type = "json"

        if use_manifest and model_id != "M0_Cerebrospinal":
            man_info = CheckpointManager.get_active_manifest_info(model_id)
            model_entry = man_info.get("model_entry") if man_info else None
            if isinstance(model_entry, dict) and model_entry.get("generation") == man_info.get("committed_generation"):
                rel_path = model_entry.get("checkpoint_path")
                if rel_path:
                    manifest_resolved = os.path.normpath(os.path.join(BASE_DIR, rel_path))
                    os.makedirs(os.path.dirname(manifest_resolved), exist_ok=True)
                    return manifest_resolved

        path = os.path.join(BASE_DIR, "models", model_id, f"checkpoint.{format_type}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @staticmethod
    def get_candidate_path(model_id, transaction_id=None):
        specs = MODEL_SPECS
        format_type = specs.get(model_id, {}).get("checkpoint_format", "pkl")
        if model_id == "M0_Cerebrospinal":
            format_type = "json"
        suffix = f"candidate_{transaction_id}" if transaction_id else "candidate"
        path = os.path.join(BASE_DIR, "models", model_id, f"checkpoint.{suffix}.{format_type}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @staticmethod
    def backup_checkpoint(model_id, reason="recovery"):
        path = CheckpointManager.get_canonical_path(model_id)
        if not os.path.exists(path) or os.path.isdir(path):
            return None
        backup_path = f"{path}.backup_{reason}_{int(time.time())}"
        shutil.copy2(path, backup_path)
        logger.info(f"💾 Đã sao lưu checkpoint {model_id} tại: {backup_path}")
        return backup_path

    @staticmethod
    def verify_checkpoint_integrity(model_id, chkpt, is_torch=True, enforce_manifest=True):
        if not isinstance(chkpt, dict) or "model_state" not in chkpt:
            raise CheckpointError(f"[{model_id}] Sai schema: Thiếu trường model_state.")
        if chkpt.get("schema_version") != 5:
            raise CheckpointError(f"[{model_id}] Schema version không hợp lệ: {chkpt.get('schema_version')}.")
        if chkpt.get("model_id") != model_id:
            raise CheckpointError(f"[{model_id}] Sai lệch model_id trong checkpoint: {chkpt.get('model_id')}.")

        phase = chkpt.get("phase", "LIVE")
        if phase not in ("LIVE", "COMMITTED"):
            raise CheckpointError(f"[{model_id}] Checkpoint đang ở phase '{phase}', không phải bản đã commit/live.")

        gen = chkpt.get("generation", 0)
        topo_ver = chkpt.get("topology_version", 1)
        if not isinstance(gen, int) or gen < 0 or not isinstance(topo_ver, int) or topo_ver < 1:
            raise CheckpointError(f"[{model_id}] generation ({gen}) hoặc topology_version ({topo_ver}) không hợp lệ.")

        extra_dict = chkpt.get("extra_states") if isinstance(chkpt.get("extra_states"), dict) else {}
        topo_dict = chkpt.get("topology") if isinstance(chkpt.get("topology"), dict) and chkpt.get("topology") else extra_dict.get("topology", {})
        if isinstance(topo_dict, dict) and len(topo_dict) > 0 and chkpt.get("topology_signature") not in (None, "UNKNOWN_TOPOLOGY", "JSON_STATIC"):
            expected_sig = topology_signature_hash(topo_dict)
            if chkpt.get("topology_signature") != expected_sig:
                raise CheckpointError(f"[{model_id}] Sai lệch topology_signature ({chkpt.get('topology_signature')} != {expected_sig}).")

        state = chkpt["model_state"]
        if state is None:
            raise CheckpointError(f"[{model_id}] model_state rỗng (None).")

        if is_torch and not CheckpointManager.get_canonical_path(model_id).endswith(".json"):
            if not isinstance(state, dict) or len(state) == 0:
                raise CheckpointError(f"[{model_id}] PyTorch state_dict phải là dict không rỗng.")
            tensor_count = 0
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if k.endswith((".mem", ".syn")) and v.numel() == 0:
                        continue
                    tensor_count += 1
                    if v.numel() == 0 or not torch.isfinite(v).all():
                        raise CheckpointError(f"[{model_id}] Tensor '{k}' rỗng hoặc chứa NaN/Inf.")
            if tensor_count == 0:
                raise CheckpointError(f"[{model_id}] state_dict không chứa bất kỳ Tensor nào.")
            if model_id in ("M6_Language", "M7_Logic"):
                if "gate.weight" not in state or "out_layer.weight" not in state:
                    raise CheckpointError(f"[{model_id}] Thiếu gate.weight hoặc out_layer.weight.")
                num_cols = int(state["gate.weight"].shape[0])
                col_indices = {int(k.split(".")[1]) for k in state.keys() if k.startswith("columns.") and k.split(".")[1].isdigit()}
                if num_cols < 1 or len(col_indices) != num_cols:
                    raise CheckpointError(f"[{model_id}] Số cột gate.weight ({num_cols}) lệch với columns ({len(col_indices)}).")
                if isinstance(topo_dict, dict) and "num_columns" in topo_dict and topo_dict["num_columns"] != num_cols:
                    raise CheckpointError(f"[{model_id}] Metadata num_columns ({topo_dict['num_columns']}) lệch với shape ({num_cols}).")
            elif model_id == "M9_Thalamus":
                    if "weights" not in state or state["weights"].dim() != 2 or state["weights"].shape[1] != 128:
                        raise CheckpointError(f"[{model_id}] Shape weights của M9 không hợp lệ.")
                    if "som_eligibility" in state and state["som_eligibility"].shape != state["weights"].shape:
                        raise CheckpointError(f"[{model_id}] Lệch số node giữa weights và som_eligibility của M9.")
                    if "attention_to_route" in state and state["attention_to_route"].shape != (5, 11):
                        raise CheckpointError(f"[{model_id}] Shape attention_to_route của M9 không hợp lệ.")
        elif model_id != "M0_Cerebrospinal":
            if model_id == "M8_Memory":
                if not isinstance(state, dict) or "nodes" not in state or "input_dim" not in state:
                    raise CheckpointError(f"[{model_id}] Cấu trúc ART state của M8 không hợp lệ.")
            elif model_id in ("M1a_Dorsal", "M1b_Ventral", "M3_Spatial", "M4_Motor"):
                genome = state.get("genome", state) if isinstance(state, dict) else state
                if not (hasattr(genome, "nodes") and hasattr(genome, "connections")):
                    raise CheckpointError(f"[{model_id}] Cấu trúc NEAT genome không hợp lệ.")

        if enforce_manifest and model_id != "M0_Cerebrospinal":
            man_info = CheckpointManager.get_active_manifest_info(model_id)
            if man_info is not None:
                committed_gen = man_info.get("committed_generation")
                if committed_gen is not None and int(committed_gen) != gen:
                    raise CheckpointError(
                        f"[{model_id}] Lệch thế hệ với brain_manifest.json (Checkpoint Gen={gen}, Manifest Committed Gen={committed_gen})."
                    )
                active_tx = man_info.get("current_transaction_id")
                tx_status = man_info.get("transaction_status")
                chkpt_tx = chkpt.get("transaction_id")
                model_entry = man_info.get("model_entry")
                if tx_status != "COMMITTED":
                    if active_tx is not None and chkpt_tx == active_tx:
                        raise CheckpointError(
                            f"[{model_id}] Giao dịch '{chkpt_tx}' đang ở trạng thái '{tx_status}', chưa được commit toàn cục."
                        )
                    if not isinstance(model_entry, dict) or model_entry.get("generation") != gen:
                        raise CheckpointError(
                            f"[{model_id}] Manifest đang ở trạng thái '{tx_status}' và chưa có bản ghi xác thực cho thế hệ {gen}."
                        )
                if isinstance(model_entry, dict) and model_entry.get("generation") == gen and model_entry.get("state_hash") and phase == "COMMITTED":
                    if chkpt.get("state_hash") != model_entry["state_hash"]:
                        raise CheckpointError(f"[{model_id}] state_hash COMMITTED không khớp với brain_manifest.json.")

    @staticmethod
    def save(model_id, live_state, optimizer=None, is_torch=True, extra_states=None):
        extra_states = extra_states or {}
        clean_extra = {k: v for k, v in extra_states.items() if k != "extra_states"}
        generation = extra_states.get("generation", 0)
        tx_id = extra_states.get("transaction_id")
        raw_phase = extra_states.get("phase")
        is_candidate = bool(extra_states.get("is_candidate", False)) or raw_phase in ("PREPARED", "CANDIDATE", "CANDIDATE_STAGED")
        if is_candidate:
            phase = raw_phase if raw_phase in ("PREPARED", "CANDIDATE", "CANDIDATE_STAGED") else "CANDIDATE_STAGED"
            path = CheckpointManager.get_candidate_path(model_id, tx_id or extra_states.get("candidate_transaction_id"))
        else:
            phase = raw_phase or ("LIVE" if tx_id is None else "COMMITTED")
            path = CheckpointManager.get_canonical_path(model_id)
            if extra_states.get("backup_before_save", False) and os.path.exists(path):
                CheckpointManager.backup_checkpoint(model_id, reason="pre_save")
        
        cand_hash = extra_states.get("candidate_hash")
        parent_state_hash = extra_states.get("parent_state_hash")

        if path.endswith(".json"):
            safe_model_state = json.loads(json.dumps(live_state, cls=SafeJSONEncoder))
            
            json_checkpoint = {
                "schema_version": 5,
                "model_id": model_id,
                "hash_algorithm": HASH_ALGORITHM_VERSION,
                "state_hash": tensor_state_hash(safe_model_state),
                "topology_signature": "JSON_STATIC",
                "topology_version": extra_states.get("topology_version", 1),
                "generation": generation,
                "parent_generation": extra_states.get("parent_generation", max(0, generation - 1)),
                "mutation_log": "Base",
                "topology": {},
                "model_state": safe_model_state,
                "optimizer_state": None,
                "fitness": 0.0,
                "transaction_id": tx_id,
                "phase": phase,
                "candidate_hash": cand_hash,
                "parent_state_hash": parent_state_hash,
                "extra_states": clean_extra, 
                "metadata": {"timestamp": time.time()}
            }
            tmp_path = path + ".tmp"
            for attempt in range(3):
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(json_checkpoint, f, indent=4, cls=SafeJSONEncoder)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, path)
                    return True
                except Exception as e:
                    logger.warning(f"JSON Checkpoint save attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1)
            raise CheckpointError(f"Persistent Lock Failed for JSON Checkpoint: {model_id}")

        if is_torch and hasattr(live_state, 'state_dict'):
            model_state = live_state.state_dict()
        else:
            model_state = live_state
            
        opt_state = optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer
        state_hash = tensor_state_hash(model_state)
        topo_dict = extra_states.get("topology", {})
        topo_signature = topology_signature_hash(topo_dict)
        
        checkpoint = {
            "schema_version": 5,
            "model_id": model_id,
            "hash_algorithm": HASH_ALGORITHM_VERSION,
            "state_hash": state_hash,
            "topology_signature": topo_signature,
            "topology_version": extra_states.get("topology_version", extra_states.get("architecture_version", 1)),
            "generation": generation,
            "parent_generation": extra_states.get("parent_generation", max(0, generation - 1)),
            "mutation_log": extra_states.get("mutation_record", extra_states.get("mutation_log", "Base")),
            "topology": topo_dict,
            "model_state": model_state,
            "optimizer_state": opt_state,
            "fitness": extra_states.get("fitness", 0.0),
            "transaction_id": tx_id,
            "phase": phase,
            "candidate_hash": cand_hash,
            "parent_state_hash": parent_state_hash,
            "extra_states": clean_extra, 
            "metadata": {"timestamp": time.time()}
        }
        
        if os.path.isdir(path):
            raise CheckpointError(f"CRITICAL: Checkpoint path '{path}' is a Directory, not a File! Aborting.")
            
        tmp_path = path + ".tmp"
        
        for attempt in range(3):
            try:
                with open(tmp_path, 'wb') as f:
                    if is_torch:
                        torch.save(checkpoint, f)
                    else:
                        pickle.dump(checkpoint, f, protocol=4)
                    f.flush()
                    os.fsync(f.fileno()) 
                
                os.replace(tmp_path, path) 
                return True
            except Exception as e:
                logger.warning(f"Checkpoint save attempt {attempt + 1} failed: {e}")
                time.sleep(0.1)
        return False

    @staticmethod
    def load(model_id, is_torch=True, validate_hash=True, enforce_manifest=True):
        path = CheckpointManager.get_canonical_path(model_id, use_manifest=enforce_manifest)
        
        if os.path.isdir(path):
            logger.warning(f"CRITICAL: Checkpoint path '{path}' is a Directory! Renaming to .quarantine")
            try:
                os.rename(path, path + ".quarantine")
            except Exception as e:
                logger.error(f"Không thể cách ly Directory {path}: {e}")
            return None
            
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found for {model_id}. Initializing NEW.")
            return None
            
        try:
            if path.endswith(".json"):
                with open(path, 'r', encoding="utf-8") as f:
                    chkpt = json.load(f)
            elif is_torch:
                chkpt = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
            else:
                with open(path, 'rb') as f:
                    chkpt = pickle.load(f)
                    
            if isinstance(chkpt, dict) and "state_hash" in chkpt and "model_state" in chkpt:
                recorded_ver = chkpt.get("hash_algorithm")
                hash_ok, hash_diag, actual_v1 = verify_state_hash(
                    chkpt["model_state"], chkpt["state_hash"], recorded_version=recorded_ver
                )
                
                if validate_hash and not hash_ok:
                    raise CheckpointError(
                        f"Tính toàn vẹn vi phạm tại {model_id} ({hash_diag}). "
                        f"Hash lưu trữ {str(chkpt['state_hash'])[:8]} != thực tế {actual_v1[:8]}."
                    )
                if "architecture_version" in chkpt and "topology_version" not in chkpt:
                    chkpt["topology_version"] = chkpt.pop("architecture_version")
                
                for key in ["transaction_id", "phase", "candidate_hash", "parent_state_hash"]:
                    if key not in chkpt:
                        chkpt[key] = "LIVE" if key == "phase" else None
                if chkpt["phase"] is None:
                    chkpt["phase"] = "LIVE"

                if "extra_states" not in chkpt:
                    chkpt["extra_states"] = {
                        k: v for k, v in chkpt.items()
                        if k not in [
                            "schema_version", "model_id", "hash_algorithm", "state_hash",
                            "topology_version", "generation", "parent_generation", "mutation_log",
                            "topology", "model_state", "optimizer_state", "fitness", "metadata",
                            "topology_signature", "transaction_id", "phase", "candidate_hash", "parent_state_hash"
                        ]
                    }

                if validate_hash:
                    CheckpointManager.verify_checkpoint_integrity(
                        model_id, chkpt, is_torch=is_torch, enforce_manifest=enforce_manifest
                    )
                return chkpt
            else:
                if validate_hash:
                    CheckpointManager.backup_checkpoint(model_id, reason="legacy_format")
                    raise CheckpointError(
                        f"[{model_id}] Phát hiện checkpoint định dạng cũ (thiếu state_hash/schema V5). "
                        f"Đã tạo bản sao lưu; vui lòng chạy công cụ phục hồi đã xác minh."
                    )
                logger.warning(f"Checkpoint {model_id}: Legacy format detected in diagnostic mode.")
                extracted_gen = chkpt.get("generation", 0) if isinstance(chkpt, dict) else 0
                extracted_fit = chkpt.get("fitness", 0.0) if isinstance(chkpt, dict) else 0.0
                extracted_opt = chkpt.get("optimizer_state", None) if isinstance(chkpt, dict) else None
                extracted_topo = chkpt.get("topology", {}) if isinstance(chkpt, dict) else {}
                extracted_topo_ver = chkpt.get("topology_version", chkpt.get("architecture_version", 1)) if isinstance(chkpt, dict) else 1
                real_state = chkpt.get("model_state", chkpt.get("state", chkpt)) if isinstance(chkpt, dict) else chkpt

                return {
                    "schema_version": 5,
                    "model_id": model_id,
                    "hash_algorithm": HASH_ALGORITHM_VERSION,
                    "state_hash": tensor_state_hash(real_state),
                    "topology_signature": topology_signature_hash(extracted_topo),
                    "topology_version": extracted_topo_ver,
                    "generation": extracted_gen,
                    "parent_generation": max(0, extracted_gen - 1),
                    "mutation_log": "Migrated",
                    "topology": extracted_topo,
                    "model_state": real_state,
                    "optimizer_state": extracted_opt,
                    "fitness": extracted_fit,
                    "transaction_id": None,
                    "phase": "RECOVERY_REQUIRED",
                    "candidate_hash": None,
                    "parent_state_hash": None,
                    "extra_states": {},
                    "metadata": {"timestamp": time.time()}
                }
        except CheckpointError:
            raise
        except (pickle.UnpicklingError, EOFError, json.JSONDecodeError, RuntimeError) as e:
            logger.error(f"CORRUPTION THẬT SỰ tại {model_id}: {e}. Đang cách ly file.")
            time.sleep(0.5) 
            try: 
                os.rename(path, path + f".corrupted_{int(time.time())}")
            except Exception as rename_err: 
                logger.error(f"Thất bại khi đổi tên file hỏng {model_id}: {rename_err}")
            raise CheckpointError(f"Dữ liệu vật lý hỏng: {e}")
        except Exception as e:
            raise CheckpointError(f"Lỗi Load tại {model_id}: {e}")

    @staticmethod
    def commit_candidate(model_id, transaction_id=None, is_torch=True):
        cand_path = CheckpointManager.get_candidate_path(model_id, transaction_id)
        if not os.path.exists(cand_path):
            return False
        try:
            if cand_path.endswith(".json"):
                with open(cand_path, "r", encoding="utf-8") as f:
                    chkpt = json.load(f)
            elif is_torch:
                chkpt = torch.load(cand_path, map_location=torch.device("cpu"), weights_only=False)
            else:
                with open(cand_path, "rb") as f:
                    chkpt = pickle.load(f)

            hash_ok, hash_diag, _ = verify_state_hash(
                chkpt.get("model_state"), chkpt.get("state_hash"), recorded_version=chkpt.get("hash_algorithm")
            )
            if not hash_ok:
                raise CheckpointError(f"Candidate hash không hợp lệ ({hash_diag}).")

            chkpt_copy = dict(chkpt)
            chkpt_copy["phase"] = "COMMITTED"
            CheckpointManager.verify_checkpoint_integrity(model_id, chkpt_copy, is_torch=is_torch, enforce_manifest=False)

            CheckpointManager.backup_checkpoint(model_id, reason=f"pre_commit_{transaction_id or 'tx'}")
            extra = dict(chkpt.get("extra_states", {}))
            extra.update({
                "generation": chkpt.get("generation", 0),
                "parent_generation": chkpt.get("parent_generation", max(0, chkpt.get("generation", 0) - 1)),
                "topology_version": chkpt.get("topology_version", 1),
                "topology": chkpt.get("topology", {}),
                "mutation_record": chkpt.get("mutation_log", extra.get("mutation_record", "Base")),
                "fitness": chkpt.get("fitness", extra.get("fitness", 0.0)),
                "candidate_hash": chkpt.get("candidate_hash"),
                "parent_state_hash": chkpt.get("parent_state_hash"),
                "transaction_id": transaction_id or chkpt.get("transaction_id"),
                "phase": "COMMITTED",
                "is_candidate": False,
                "candidate_transaction_id": None
            })
            ok = CheckpointManager.save(
                model_id, chkpt["model_state"], chkpt.get("optimizer_state"), is_torch=is_torch, extra_states=extra
            )
            if ok and os.path.exists(cand_path):
                os.remove(cand_path)
            return ok
        except Exception as e:
            logger.error(f"Lỗi khi commit candidate cho {model_id}: {e}")
            return False

    @staticmethod
    def discard_candidate(model_id, transaction_id=None):
        cand_path = CheckpointManager.get_candidate_path(model_id, transaction_id)
        if os.path.exists(cand_path):
            try:
                os.remove(cand_path)
                return True
            except Exception as e:
                logger.warning(f"Không thể xóa candidate {cand_path}: {e}")
        return False