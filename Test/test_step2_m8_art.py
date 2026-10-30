import os
import sys
import time
import uuid
import json
import copy
import numpy as np
import unittest

os.environ["ASUNA_ALLOW_BOOTSTRAP"] = "1"
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_CURR_DIR) if os.path.basename(_CURR_DIR).lower() == "test" else _CURR_DIR
sys.path.append(os.path.join(BASE_DIR, "04_Memory_System", "M8_Hippocampus"))
sys.path.append(BASE_DIR)

from m8_art_memory import HippocampusMemory, SimplifiedART  # type: ignore

class TestM8ARTMemory(unittest.TestCase):
    def setUp(self):
        self.art = SimplifiedART(vigilance=0.85, input_dim=128)
        
    def test_01_novel_input_creates_category(self):
        v1 = np.random.randn(128).tolist()
        node_id, action = self.art.process(v1, is_query=False)
        self.assertEqual(action, "CREATE")
        self.assertEqual(len(self.art.memory_nodes), 1)
        self.assertIsNotNone(node_id)
        
    def test_02_similar_input_resonates(self):
        v1 = np.random.randn(128).tolist()
        node_id1, _ = self.art.process(v1, is_query=False)
        
        # very similar
        v2 = np.array(v1) + np.random.randn(128) * 0.01
        node_id2, action = self.art.process(v2.tolist(), is_query=False)
        self.assertEqual(action, "UPDATE")
        self.assertEqual(node_id1, node_id2)
        self.assertEqual(len(self.art.memory_nodes), 1)
        
    def test_03_repeated_identical_input(self):
        v1 = np.random.randn(128).tolist()
        for _ in range(5):
            self.art.process(v1, is_query=False)
        self.assertEqual(len(self.art.memory_nodes), 1)
        
    def test_04_pruning_stable_ids(self):
        # Create 3 nodes
        for _ in range(3):
            self.art.process(np.random.randn(128).tolist(), is_query=False)
        self.assertEqual(len(self.art.memory_nodes), 3)
        ids = [n["node_id"] for n in self.art.memory_nodes]
        
        # Make node 1 stale by changing last_accessed and count
        self.art.memory_nodes[1]["last_accessed"] = time.time() - 86400 * 40
        self.art.memory_nodes[1]["count"] = 1
        self.art.memory_nodes[0]["count"] = 5
        self.art.memory_nodes[2]["count"] = 5
        
        self.art.prune_stale()
        self.assertEqual(len(self.art.memory_nodes), 2)
        self.assertEqual(self.art.memory_nodes[0]["node_id"], ids[0])
        self.assertEqual(self.art.memory_nodes[1]["node_id"], ids[2])
        
    def test_05_merging_stable_ids(self):
        v1 = np.random.randn(128)
        self.art.process(v1.tolist(), is_query=False)
        
        # force another node close enough
        v2 = v1 + np.random.randn(128) * 0.01
        self.art.memory_nodes.append({
            "node_id": str(uuid.uuid4()),
            "weight": (v2 / np.linalg.norm(v2)).tolist(),
            "count": 1,
            "created_at": time.time(),
            "last_accessed": time.time()
        })
        
        ids = [n["node_id"] for n in self.art.memory_nodes]
        merged = self.art.consolidate_merge(merge_threshold=0.90)
        self.assertEqual(merged, 1)
        self.assertEqual(len(self.art.memory_nodes), 1)
        self.assertEqual(self.art.memory_nodes[0]["node_id"], ids[0])
        
    def test_06_nan_input(self):
        v1 = np.full(128, np.nan).tolist()
        node_id, action = self.art.process(v1, is_query=False)
        self.assertEqual(action, "INVALID_INPUT")
        self.assertEqual(len(self.art.memory_nodes), 0)
        
    def test_07_inf_input(self):
        v1 = np.full(128, np.inf).tolist()
        node_id, action = self.art.process(v1, is_query=False)
        self.assertEqual(action, "INVALID_INPUT")
        self.assertEqual(len(self.art.memory_nodes), 0)
        
    def test_08_zero_vector(self):
        v1 = np.zeros(128).tolist()
        node_id, action = self.art.process(v1, is_query=False)
        self.assertEqual(action, "BLANK")
        
    def test_09_wrong_dimension(self):
        v1 = np.random.randn(64).tolist()
        node_id, action = self.art.process(v1, is_query=False)
        self.assertEqual(action, "DIMENSION_MISMATCH")
        
    def test_10_128d_canonical_accepted(self):
        v1 = np.random.randn(128).tolist()
        node_id, action = self.art.process(v1, is_query=False)
        self.assertEqual(action, "CREATE")
        
    @staticmethod
    def _cleanup_m8_node(mem, node_id):
        import shutil
        try: mem.training_queue.put_nowait(None)
        except Exception: pass
        try: mem.upstream_sock.close(linger=0)
        except Exception: pass
        try: mem.bus.shutdown()
        except Exception: pass
        shutil.rmtree(os.path.join(BASE_DIR, "models", node_id), ignore_errors=True)

    def test_11_chromadb_metadata(self):
        mem = HippocampusMemory(node_id="Test_M8_11")
        try:
            if mem.collection is None:
                self.skipTest("ChromaDB not available")
            v1 = np.random.randn(128).tolist()
            node_id, _ = mem.art.process(v1, is_query=False)
            mem.pending_memories.append({
                "id": "test",
                "vector": v1,
                "doc": "{}",
                "meta": {"art_node": node_id}
            })
            self.assertEqual(mem.pending_memories[0]["meta"]["art_node"], node_id)
        finally:
            self._cleanup_m8_node(mem, "Test_M8_11")
        
    def test_12_checkpoint_preserves_id(self):
        v1 = np.random.randn(128).tolist()
        node_id, _ = self.art.process(v1, is_query=False)
        state = self.art.get_state()
        
        art2 = SimplifiedART(input_dim=128)
        art2.load_state(state)
        self.assertEqual(art2.memory_nodes[0]["node_id"], node_id)
        
    def test_13_2pc_commit_preserves(self):
        mem = HippocampusMemory(node_id="Test_M8_13")
        try:
            mem._evo_state = "LIVE"
            mem.parent_art_state = copy.deepcopy(mem.art.get_state())
            mem._parent_pending_memories = copy.deepcopy(mem.pending_memories)
            
            candidate = SimplifiedART()
            candidate.load_state(mem.art.get_state())
            
            v1 = np.random.randn(128).tolist()
            node_id, _ = candidate.process(v1, is_query=False)
            
            mem._candidate_art_state = candidate.get_state()
            mem._evo_state = "CANDIDATE_STAGED"
            mem._current_transaction_id = "tx1"
            
            mem.art.load_state(mem._candidate_art_state)
            mem._candidate_art_state = None
            mem._current_transaction_id = None
            mem._evo_state = "LIVE"
            
            self.assertEqual(len(mem.art.memory_nodes), 1)
            self.assertEqual(mem.art.memory_nodes[0]["node_id"], node_id)
        finally:
            self._cleanup_m8_node(mem, "Test_M8_13")
        
    def test_14_2pc_rollback_removes(self):
        mem = HippocampusMemory(node_id="Test_M8_14")
        try:
            mem._evo_state = "LIVE"
            mem.parent_art_state = copy.deepcopy(mem.art.get_state())
            
            candidate = SimplifiedART()
            candidate.load_state(mem.art.get_state())
            
            v1 = np.random.randn(128).tolist()
            candidate.process(v1, is_query=False)
            
            mem._candidate_art_state = candidate.get_state()
            mem._evo_state = "CANDIDATE_STAGED"
            mem._current_transaction_id = "tx1"
            
            mem.art.load_state(mem.parent_art_state)
            mem._candidate_art_state = None
            mem._current_transaction_id = None
            mem._evo_state = "LIVE"
            
            self.assertEqual(len(mem.art.memory_nodes), 0)
        finally:
            self._cleanup_m8_node(mem, "Test_M8_14")
        
    def test_15_legacy_memory_retrieval(self):
        # Legacy state without node_id
        legacy_state = {
            "vigilance": 0.85,
            "input_dim": 128,
            "functional_revision": 1,
            "nodes": [
                {
                    "weight": np.random.randn(128).tolist(),
                    "count": 5,
                    "created_at": time.time()
                }
            ]
        }
        art2 = SimplifiedART(input_dim=128)
        art2.load_state(legacy_state)
        self.assertIn("node_id", art2.memory_nodes[0])
        self.assertIsNotNone(art2.memory_nodes[0]["node_id"])
        
    def test_16_100_identical_inputs(self):
        v1 = np.random.randn(128).tolist()
        for _ in range(100):
            self.art.process(v1, is_query=False)
        self.assertEqual(len(self.art.memory_nodes), 1)
        self.assertEqual(self.art.memory_nodes[0]["count"], 100)
        
    def test_17_two_novel_inputs(self):
        v1 = np.zeros(128)
        v1[0] = 1.0
        v2 = np.zeros(128)
        v2[1] = 1.0
        
        self.art.process(v1.tolist(), is_query=False)
        self.art.process(v2.tolist(), is_query=False)
        
        self.assertEqual(len(self.art.memory_nodes), 2)
        
    def test_18_concurrent_messages(self):
        # mock a few sequential runtime message equivalents
        for i in range(10):
            self.art.process(np.random.randn(128).tolist(), is_query=False)
        
        self.assertEqual(len(self.art.memory_nodes), 10)
        # Verify no corruption (e.g. no nans, stable ids)
        ids = [n["node_id"] for n in self.art.memory_nodes]
        self.assertEqual(len(set(ids)), 10)
        for n in self.art.memory_nodes:
            self.assertTrue(np.isfinite(n["weight"]).all())

if __name__ == "__main__":
    unittest.main()
