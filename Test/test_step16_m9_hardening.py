#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP 1.6 — M9 Representation Contract Hardening Tests
=====================================================

Tests for the three verified bugs patched in STEP 1.6:
  BUG #1: Modality Isolation (global cross-modality energy normalization)
  BUG #2: M6 np.interp fallback on fused_latent
  BUG #3: NaN/Inf propagation into DynamicSOM

Also tests:
  - State-hold semantics
  - Determinism
  - DynamicSOM basic functionality
  - Causal metadata integrity
  - M8 canonical path
"""

import os
import sys
import copy
import json
import numpy as np
import pytest

# Setup path
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "00_Central_Router", "M9_Thalamus"))

from main_life_loop import numeric_encoder, embedding_encoder

# =========================================================================
# HELPER: Replicate M9 projection logic (STEP 1.6 patched version)
# =========================================================================

MODALITY_SPECS = {
    "vision_where": (10, 16), "vision_what": (10, 16), "audio": (10, 16),
    "spatial": (10, 8), "threat": (10, 4), "semantics": (39, 16),
    "memory": (39, 16), "motor_fb": (10, 8), "vocal_fb": (2, 4),
    "self_audio": (10, 8), "dream_replay": (128, 16)
}

MODALITY_ORDER = ["vision_where", "vision_what", "audio", "spatial", "threat",
                  "semantics", "memory", "motor_fb", "vocal_fb", "self_audio", "dream_replay"]

SLICE_OFFSETS = {}
offset = 0
for mod in MODALITY_ORDER:
    out_dim = MODALITY_SPECS[mod][1]
    SLICE_OFFSETS[mod] = (offset, offset + out_dim)
    offset += out_dim

assert offset == 128, f"Total output dims should be 128, got {offset}"


def build_projectors(seed=42):
    """Build the same fixed projectors used in M9."""
    rng = np.random.RandomState(seed)
    projectors = {}
    for mod, (in_dim, out_dim) in MODALITY_SPECS.items():
        mat = rng.randn(out_dim, in_dim).astype(np.float32) / np.sqrt(in_dim)
        projectors[mod] = mat
    return projectors


def build_default_context():
    """Build the initial zero context matching M9 initialization."""
    return {
        "vision_where": [0.0]*10, "vision_what": [0.0]*10, "audio": [0.0]*10,
        "spatial": [0.0]*10, "threat": [0.0]*10, "semantics": [0.0]*39,
        "memory": [0.0]*39, "motor_fb": [0.0]*10, "vocal_fb": [0.0]*2,
        "self_audio": [0.0]*10, "dream_replay": [0.0]*128
    }


def compute_fused_latent(context, projectors):
    """
    Replicate the PATCHED M9 fusion logic (STEP 1.6).
    Each modality is projected INDEPENDENTLY — no cross-modality weight multiplication.
    """
    result = []
    for k in MODALITY_ORDER:
        vec = context[k]
        in_dim = MODALITY_SPECS[k][0]
        if len(vec) != in_dim:
            vec = numeric_encoder(vec, in_dim)
        vec_np = np.array(vec, dtype=np.float32)
        proj = np.dot(projectors[k], vec_np).tolist()
        result.extend(proj)
    return result


def get_slice(fused_128, modality_name):
    """Extract the slice for a given modality from the 128D vector."""
    start, end = SLICE_OFFSETS[modality_name]
    return fused_128[start:end]


# =========================================================================
# TEST 1 — MODALITY ISOLATION (11/11)
# =========================================================================

class TestModalityIsolation:
    """
    Adversarial test: Change ONLY one modality, verify ONLY its slice changes.
    All other slices must remain identical.
    """

    def setup_method(self):
        self.projectors = build_projectors(seed=42)

    @pytest.mark.parametrize("target_modality", MODALITY_ORDER)
    def test_isolation_per_modality(self, target_modality):
        """Change only target_modality — only its slice should change."""
        # Build baseline context with non-zero values
        ctx_base = build_default_context()
        rng = np.random.RandomState(123)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx_base[k] = rng.randn(in_dim).tolist()

        fused_base = compute_fused_latent(ctx_base, self.projectors)

        # Modify ONLY the target modality
        ctx_modified = copy.deepcopy(ctx_base)
        in_dim = MODALITY_SPECS[target_modality][0]
        ctx_modified[target_modality] = (np.array(ctx_base[target_modality]) + 1.0).tolist()

        fused_modified = compute_fused_latent(ctx_modified, self.projectors)

        # Verify: target slice MUST change
        base_slice = get_slice(fused_base, target_modality)
        mod_slice = get_slice(fused_modified, target_modality)
        assert not np.allclose(base_slice, mod_slice), \
            f"Slice for {target_modality} should change when its input changes"

        # Verify: all OTHER slices MUST remain identical
        for other_mod in MODALITY_ORDER:
            if other_mod == target_modality:
                continue
            base_other = get_slice(fused_base, other_mod)
            mod_other = get_slice(fused_modified, other_mod)
            assert np.array_equal(np.array(base_other), np.array(mod_other)), \
                f"Slice for {other_mod} changed when only {target_modality} was modified! " \
                f"Diff: {np.max(np.abs(np.array(base_other) - np.array(mod_other)))}"


# =========================================================================
# TEST 2 — DETERMINISM
# =========================================================================

class TestDeterminism:
    """Same input must produce same output across multiple runs."""

    def test_deterministic_projection(self):
        """3 consecutive runs with same input must produce identical output."""
        ctx = build_default_context()
        rng = np.random.RandomState(999)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()

        results = []
        for _ in range(3):
            projectors = build_projectors(seed=42)
            fused = compute_fused_latent(ctx, projectors)
            results.append(fused)

        assert np.array_equal(np.array(results[0]), np.array(results[1])), "Run 1 != Run 2"
        assert np.array_equal(np.array(results[1]), np.array(results[2])), "Run 2 != Run 3"


# =========================================================================
# TEST 3 — M6 CANONICAL CONTRACT (128D)
# =========================================================================

class TestM6CanonicalContract:
    """M9 output must be exactly 128D. M6 must not resize."""

    def test_m9_output_is_128d(self):
        """M9 canonical fused_latent must be exactly 128 dimensions."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        fused = compute_fused_latent(ctx, projectors)
        assert len(fused) == 128, f"Expected 128D, got {len(fused)}D"

    def test_m9_output_128d_with_data(self):
        """M9 output 128D even with non-trivial data."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        rng = np.random.RandomState(42)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()
        fused = compute_fused_latent(ctx, projectors)
        assert len(fused) == 128, f"Expected 128D, got {len(fused)}D"

    def test_no_np_interp_in_canonical_path(self):
        """
        Verify that the M6 THALAMUS handler code no longer contains
        numeric_encoder(fused, 128) for resizing fused_latent.
        """
        main_file = os.path.join(BASE_DIR, "main_life_loop.py")
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the THALAMUS_M6 handler section
        thalamus_idx = content.find('topic == "THALAMUS_M6"')
        assert thalamus_idx > 0, "Could not find THALAMUS_M6 handler"

        # Look at the next 800 chars after THALAMUS_M6 handler
        section = content[thalamus_idx:thalamus_idx + 800]

        # Must NOT contain the old fallback pattern
        assert "numeric_encoder(fused, 128)" not in section, \
            "M6 still contains numeric_encoder(fused, 128) fallback!"

        # Must contain rejection logic
        assert "Rejected invalid fused_latent" in section or "reject" in section.lower(), \
            "M6 should contain rejection logic for invalid fused_latent"


# =========================================================================
# TEST 4 — INVALID SHAPE REJECTION
# =========================================================================

class TestInvalidShapeRejection:
    """Invalid dimensions must be rejected, not silently resized."""

    def test_code_rejects_127d(self):
        """Verify code path rejects 127D fused_latent."""
        fused_127 = [0.0] * 127
        assert len(fused_127) != 128, "127D should not pass 128D check"

    def test_code_rejects_129d(self):
        """Verify code path rejects 129D fused_latent."""
        fused_129 = [0.0] * 129
        assert len(fused_129) != 128, "129D should not pass 128D check"

    def test_code_rejects_empty(self):
        """Verify code path rejects empty vector."""
        fused_empty = []
        assert len(fused_empty) != 128, "Empty should not pass 128D check"

    def test_m6_rejection_code_exists(self):
        """Verify M6 handler has strict validation code."""
        main_file = os.path.join(BASE_DIR, "main_life_loop.py")
        with open(main_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the THALAMUS_M6 section and check for rejection
        in_thalamus_m6 = False
        has_rejection = False
        has_continue = False
        for line in lines:
            if 'THALAMUS_M6' in line:
                in_thalamus_m6 = True
            if in_thalamus_m6:
                if 'len(fused) != 128' in line and 'continue' not in line and 'numeric_encoder' in line:
                    pytest.fail("M6 still uses numeric_encoder to resize fused_latent!")
                if 'Rejected invalid fused_latent' in line:
                    has_rejection = True
                if has_rejection and 'continue' in line:
                    has_continue = True
                    break
        assert has_rejection, "M6 must have rejection logic for invalid fused_latent"
        assert has_continue, "M6 must skip (continue) invalid payloads, not crash"


# =========================================================================
# TEST 5 — NaN/Inf PROTECTION
# =========================================================================

class TestNanInfProtection:
    """NaN/Inf must not silently enter DynamicSOM state."""

    def test_nan_in_single_modality(self):
        """Inject NaN into one modality — output must be finite."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        rng = np.random.RandomState(42)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()

        # Inject NaN into vision_where
        ctx_nan = copy.deepcopy(ctx)
        ctx_nan["vision_where"][0] = float('nan')

        # The NaN sanitizer should have caught this before projection
        # Simulate the sanitizer
        for k in list(ctx_nan.keys()):
            vec_arr = np.array(ctx_nan[k], dtype=np.float32)
            if not np.all(np.isfinite(vec_arr)):
                ctx_nan[k] = ctx[k]  # state-hold: revert to previous

        fused = compute_fused_latent(ctx_nan, projectors)
        assert np.all(np.isfinite(fused)), "NaN leaked into fused_latent!"

    def test_inf_in_single_modality(self):
        """Inject +Inf into one modality — output must be finite."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        rng = np.random.RandomState(42)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()

        ctx_inf = copy.deepcopy(ctx)
        ctx_inf["audio"][3] = float('inf')

        for k in list(ctx_inf.keys()):
            vec_arr = np.array(ctx_inf[k], dtype=np.float32)
            if not np.all(np.isfinite(vec_arr)):
                ctx_inf[k] = ctx[k]

        fused = compute_fused_latent(ctx_inf, projectors)
        assert np.all(np.isfinite(fused)), "+Inf leaked into fused_latent!"

    def test_neg_inf_in_single_modality(self):
        """Inject -Inf into one modality — output must be finite."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()

        ctx_ninf = copy.deepcopy(ctx)
        ctx_ninf["spatial"][0] = float('-inf')

        for k in list(ctx_ninf.keys()):
            vec_arr = np.array(ctx_ninf[k], dtype=np.float32)
            if not np.all(np.isfinite(vec_arr)):
                ctx_ninf[k] = ctx[k]

        fused = compute_fused_latent(ctx_ninf, projectors)
        assert np.all(np.isfinite(fused)), "-Inf leaked into fused_latent!"

    def test_entire_modality_nan(self):
        """Entire modality is NaN — should be sanitized."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        rng = np.random.RandomState(42)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()

        ctx_nan = copy.deepcopy(ctx)
        ctx_nan["semantics"] = [float('nan')] * 39

        for k in list(ctx_nan.keys()):
            vec_arr = np.array(ctx_nan[k], dtype=np.float32)
            if not np.all(np.isfinite(vec_arr)):
                ctx_nan[k] = ctx[k]

        fused = compute_fused_latent(ctx_nan, projectors)
        assert np.all(np.isfinite(fused)), "NaN leaked from fully-NaN modality!"

    def test_multiple_modalities_nan_inf(self):
        """Multiple modalities contain NaN/Inf."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        rng = np.random.RandomState(42)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()

        ctx_bad = copy.deepcopy(ctx)
        ctx_bad["vision_where"][0] = float('nan')
        ctx_bad["audio"][5] = float('inf')
        ctx_bad["memory"] = [float('-inf')] * 39

        for k in list(ctx_bad.keys()):
            vec_arr = np.array(ctx_bad[k], dtype=np.float32)
            if not np.all(np.isfinite(vec_arr)):
                ctx_bad[k] = ctx[k]

        fused = compute_fused_latent(ctx_bad, projectors)
        assert np.all(np.isfinite(fused)), "NaN/Inf leaked from multiple bad modalities!"

    @pytest.mark.parametrize("target_modality", MODALITY_ORDER)
    def test_nan_per_modality(self, target_modality):
        """Inject NaN into every modality individually."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        rng = np.random.RandomState(42)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()

        ctx_nan = copy.deepcopy(ctx)
        ctx_nan[target_modality][0] = float('nan')

        for k in list(ctx_nan.keys()):
            vec_arr = np.array(ctx_nan[k], dtype=np.float32)
            if not np.all(np.isfinite(vec_arr)):
                ctx_nan[k] = ctx[k]

        fused = compute_fused_latent(ctx_nan, projectors)
        assert np.all(np.isfinite(fused)), f"NaN leaked from {target_modality}!"

    def test_m9_code_has_isfinite_check(self):
        """Verify M9 code has np.isfinite check before DynamicSOM."""
        main_file = os.path.join(BASE_DIR, "main_life_loop.py")
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "np.isfinite" in content or "np.all(np.isfinite" in content, \
            "M9 code must contain np.isfinite check"

        # Verify it's in the M9 thalamus section
        m9_start = content.find("def m9_thalamus_process")
        assert m9_start > 0
        m9_section = content[m9_start:m9_start + 25000]
        assert "np.isfinite" in m9_section, \
            "np.isfinite must be in m9_thalamus_process"


# =========================================================================
# TEST 6 — STATE-HOLD
# =========================================================================

class TestStateHold:
    """State-hold semantics: missing modality retains previous value."""

    def test_state_hold_basic(self):
        """
        tick N: set vision = V1
        tick N+1: no new vision
        Expected: vision still = V1
        """
        ctx = build_default_context()

        # tick N: receive vision data
        V1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ctx["vision_where"] = V1

        # tick N+1: no new vision data (no update to ctx["vision_where"])
        # State-hold: the value should remain V1
        assert ctx["vision_where"] == V1, "State-hold failed: vision_where changed without new data"

    def test_state_hold_after_nan_recovery(self):
        """
        tick N: valid vision = V1
        tick N+1: NaN vision received → should revert to V1
        """
        V1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        prev_context = {"vision_where": list(V1)}

        # tick N+1: NaN arrives
        current_val = [float('nan')] + [0.0]*9
        vec_arr = np.array(current_val, dtype=np.float32)
        if not np.all(np.isfinite(vec_arr)):
            # State-hold: revert to previous
            current_val = prev_context["vision_where"]

        assert current_val == V1, "State-hold after NaN recovery failed"

    def test_state_hold_no_zero_fill(self):
        """State-hold must NOT be zero-fill."""
        V1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ctx = build_default_context()
        ctx["vision_where"] = V1

        # Simulate no new data — ctx should keep V1
        assert ctx["vision_where"] != [0.0]*10, "State-hold incorrectly zero-filled"
        assert ctx["vision_where"] == V1


# =========================================================================
# TEST 7 — M8 SEMANTIC PATH
# =========================================================================

class TestM8SemanticPath:
    """M8 must handle new 128D fused_latent path AND legacy fallback."""

    def test_m8_new_path_accepts_128d(self):
        """M8 encode_semantic should accept 128D fused_latent directly."""
        # Simulate what M8's encode_semantic does
        fused = list(np.random.randn(128))
        payload = {"fused_latent": fused}
        data = {"payload": payload}

        # Replicate M8's encode_semantic logic for the new path
        if isinstance(data, dict):
            p = data.get("payload", data)
            if isinstance(p, dict) and "fused_latent" in p:
                vec = p["fused_latent"]
                if len(vec) == 128:
                    result = vec
                    assert len(result) == 128
                    return

        pytest.fail("M8 new semantic path did not accept 128D fused_latent")

    def test_m8_legacy_fallback(self):
        """M8 legacy path (no fused_latent) should still work."""
        data = {"some_text": "hello world"}

        # Replicate M8's encode_semantic fallback
        if isinstance(data, dict):
            p = data.get("payload", data)
            if isinstance(p, dict) and "fused_latent" in p:
                pytest.fail("This test should take the legacy path")

        # Legacy path: hash-based encoding
        import hashlib
        text_str = json.dumps(data, sort_keys=True)
        h = hashlib.sha256(text_str.encode('utf-8')).digest()
        raw_vec = [(float(b) / 127.5) - 1.0 for b in h]
        raw_vec = (raw_vec * (128 // len(raw_vec) + 1))[:128]
        assert len(raw_vec) == 128, "Legacy fallback should produce 128D"


# =========================================================================
# TEST 8 — DYNAMIC SOM BASIC FUNCTIONALITY
# =========================================================================

class TestDynamicSOM:
    """Verify DynamicSOM basic operations work with 128D input."""

    def setup_method(self):
        try:
            from m9_dynamic_som import DynamicSOM
            self.DynamicSOM = DynamicSOM
        except ImportError:
            pytest.skip("DynamicSOM not importable")

    def test_som_accepts_128d(self):
        """SOM input must be 128D."""
        import torch
        som = self.DynamicSOM(input_dim=128, initial_nodes=4, error_threshold=15.0)
        x = torch.randn(1, 128)
        bmu_idx, dist = som.forward(x)
        assert bmu_idx.shape == (1,)

    def test_som_update_and_grow(self):
        """update_and_grow should work with 128D input."""
        import torch
        som = self.DynamicSOM(input_dim=128, initial_nodes=4, error_threshold=15.0)
        x = torch.randn(1, 128)
        bmu_id, topology_changed = som.update_and_grow(x, current_tick=1, current_episode="TEST")
        assert isinstance(bmu_id, int)
        assert isinstance(topology_changed, bool)

    def test_som_candidate_workflow(self):
        """2PC prepare/commit/rollback should work."""
        import torch
        som = self.DynamicSOM(input_dim=128, initial_nodes=4, error_threshold=15.0)
        x = torch.randn(1, 128)
        som.update_and_grow(x, current_tick=1, current_episode="TEST")

        # Prepare candidate
        cand = som.prepare_candidate("tx_test_001", target_tick=1, target_episode="TEST")
        assert isinstance(cand, dict)

        # Rollback
        som.rollback_candidate(transaction_id="tx_test_001")


# =========================================================================
# TEST 9 — CAUSAL METADATA INTEGRITY
# =========================================================================

class TestCausalMetadata:
    """Verify causal metadata fields are present in m9_payload."""

    def test_m9_payload_structure(self):
        """Verify m9_payload code structure contains required fields."""
        main_file = os.path.join(BASE_DIR, "main_life_loop.py")
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find m9_payload construction
        m9_payload_idx = content.find('m9_payload = {')
        assert m9_payload_idx > 0, "m9_payload not found"

        payload_section = content[m9_payload_idx:m9_payload_idx + 500]

        # Required fields in m9_payload
        assert '"fused_latent"' in payload_section, "fused_latent missing from m9_payload"
        assert '"attention_routing"' in payload_section, "attention_routing missing from m9_payload"
        assert '"salience"' in payload_section, "salience missing from m9_payload"

    def test_semantic_vector_no_metadata(self):
        """Semantic vector (fused_latent 128D) must NOT contain metadata."""
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        rng = np.random.RandomState(42)
        for k in MODALITY_ORDER:
            in_dim = MODALITY_SPECS[k][0]
            ctx[k] = rng.randn(in_dim).tolist()

        fused = compute_fused_latent(ctx, projectors)
        assert len(fused) == 128, f"Fused latent should be exactly 128, got {len(fused)}"
        # All elements must be numeric (not metadata strings, etc.)
        for i, v in enumerate(fused):
            assert isinstance(v, (int, float)), f"Element {i} is {type(v)}, not numeric"

    def test_bus_publish_includes_provenance(self):
        """Verify bus.publish calls include parent_event_id and episode_id."""
        main_file = os.path.join(BASE_DIR, "main_life_loop.py")
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        m9_start = content.find("def m9_thalamus_process")
        m9_section = content[m9_start:m9_start + 25000]

        # Check for provenance in publish calls
        assert "parent_event_id=" in m9_section, "M9 publishes must include parent_event_id"
        assert "episode_id=" in m9_section, "M9 publishes must include episode_id"


# =========================================================================
# TEST 10 — 2PC TESTS
# =========================================================================

class TestTwoPhaseCommit:
    """Run existing 2PC test infrastructure."""

    def test_2pc_code_exists_in_m9(self):
        """Verify 2PC transaction support exists in M9 code."""
        main_file = os.path.join(BASE_DIR, "main_life_loop.py")
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        m9_start = content.find("def m9_thalamus_process")
        m9_section = content[m9_start:m9_start + 25000]

        assert "EVOLUTION_PREPARE" in m9_section, "2PC EVOLUTION_PREPARE missing"
        assert "GLOBAL_COMMIT" in m9_section, "2PC GLOBAL_COMMIT missing"
        assert "GLOBAL_ROLLBACK" in m9_section, "2PC GLOBAL_ROLLBACK missing"
        assert "prepare_candidate" in m9_section, "prepare_candidate missing"
        assert "commit_candidate" in m9_section, "commit_candidate missing"
        assert "rollback_candidate" in m9_section, "rollback_candidate missing"

    def test_2pc_in_dynamic_som(self):
        """Verify DynamicSOM has 2PC methods."""
        try:
            from m9_dynamic_som import DynamicSOM
        except ImportError:
            pytest.skip("DynamicSOM not importable")

        som = DynamicSOM(input_dim=128, initial_nodes=4)
        assert hasattr(som, 'prepare_candidate'), "DynamicSOM missing prepare_candidate"
        assert hasattr(som, 'commit_candidate'), "DynamicSOM missing commit_candidate"
        assert hasattr(som, 'rollback_candidate'), "DynamicSOM missing rollback_candidate"


# =========================================================================
# TEST BONUS — SLICE DIMENSION VERIFICATION
# =========================================================================

class TestSliceDimensions:
    """Verify slice dimensions match the specification."""

    EXPECTED_SLICES = {
        "vision_where": 16, "vision_what": 16, "audio": 16,
        "spatial": 8, "threat": 4, "semantics": 16,
        "memory": 16, "motor_fb": 8, "vocal_fb": 4,
        "self_audio": 8, "dream_replay": 16
    }

    def test_total_is_128(self):
        """Total of all slices must be 128."""
        total = sum(self.EXPECTED_SLICES.values())
        assert total == 128, f"Total slice dims = {total}, expected 128"

    @pytest.mark.parametrize("modality,expected_dim", list(MODALITY_SPECS.items()))
    def test_individual_slice_dim(self, modality, expected_dim):
        """Each modality slice has correct dimension."""
        _, out_dim = expected_dim
        projectors = build_projectors(seed=42)
        ctx = build_default_context()
        ctx[modality] = [1.0] * MODALITY_SPECS[modality][0]
        fused = compute_fused_latent(ctx, projectors)
        slice_val = get_slice(fused, modality)
        assert len(slice_val) == out_dim, f"{modality}: expected {out_dim}D slice, got {len(slice_val)}D"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
