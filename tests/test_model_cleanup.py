"""Regression checks for compact weights and original-checkpoint compatibility.

Run with: python -m unittest discover -s tests -v
Optionally set LQRBIND_LEGACY_MODEL to a pre-cleanup model.py for exact comparison.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_lqrbind as evaluator
from model import _is_legacy_unused_key


class ModelCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / "configs/lqrbind_paper_checkpoint.json").read_text())
        cls.bundle = evaluator.build_bundle(cls.config)
        cls.checkpoint = torch.load(
            ROOT / "checkpoints/lqrbind.pt", map_location="cpu", weights_only=False
        )
        cls.model = evaluator.build_model(cls.config, cls.bundle).eval()
        cls.model.load_state_dict(cls.checkpoint["model"], strict=True)

    def test_parameter_count_and_retired_slots(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 2541513)
        self.assertFalse(any(_is_legacy_unused_key(k) for k in self.model.state_dict()))
        retired = sum(v.numel() for k, v in self.checkpoint["model"].items()
                      if _is_legacy_unused_key(k))
        self.assertIn(retired, (0, 2132045))

    def test_strict_loading_and_compact_round_trip(self):
        original = self.checkpoint["model"]
        original_keys = tuple(original)
        self.model.load_state_dict(original, strict=True)
        self.assertEqual(tuple(original), original_keys)
        compact = self.model.state_dict()
        model = evaluator.build_model(self.config, self.bundle).eval()
        model.load_state_dict(compact, strict=True)
        graph = Batch.from_data_list(self.bundle.test18[:2])
        with torch.no_grad():
            torch.testing.assert_close(model(graph), self.model(graph), rtol=0, atol=0)
        bad = compact.copy()
        bad["head.unrecognized.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(bad, strict=True)
        bad = compact.copy()
        del bad["input_proj.0.weight"]
        with self.assertRaises(RuntimeError):
            model.load_state_dict(bad, strict=True)
        bad = compact.copy()
        bad["input_proj.0.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(bad, strict=True)

    def test_all_remaining_parameters_receive_gradients(self):
        model = evaluator.build_model(self.config, self.bundle)
        model.load_state_dict(self.checkpoint["model"], strict=True)
        model.train()
        graph = Batch.from_data_list(self.bundle.train[:3])
        logits = model(graph)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, graph.y.float()).backward()
        missing = [k for k, p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_released_predictions(self):
        expected = json.loads((ROOT / "configs/lqrbind_paper_metrics.json").read_text())
        for name, dataset in self.datasets():
            with self.subTest(dataset=name):
                prediction = evaluator.predict(self.model, dataset, 36, torch.device("cpu"))
                with np.load(ROOT / "expected_predictions" / f"{name}_predictions.npz") as ref:
                    np.testing.assert_array_equal(prediction["labels"], ref["labels"])
                    np.testing.assert_allclose(prediction["probabilities"], ref["probabilities"],
                                               rtol=0, atol=5e-4)
                metrics = evaluator.compute_binary_metrics(
                    prediction["labels"], prediction["probabilities"], expected[name]["threshold"]
                )
                for metric, actual in [("precision", "precision"), ("recall", "recall"),
                                       ("mcc", "mcc"), ("auc", "auroc")]:
                    self.assertAlmostEqual(metrics[actual], expected[name][metric], delta=5e-4)

    def test_exact_pre_cleanup_predictions(self):
        source = os.environ.get("LQRBIND_LEGACY_MODEL")
        if not source:
            self.skipTest("Set LQRBIND_LEGACY_MODEL to compare against pre-cleanup source")
        spec = importlib.util.spec_from_file_location("legacy_lqrbind_model", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = self.config
        legacy = module.LQRBind(
            input_dim=self.bundle.train[0].x.size(-1), hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"], dropout=config["dropout"],
            ttn_bond_dim=config["ttn_bond_dim"], quantum_state_dim=config["quantum_state_dim"],
            candidate_topk=config["candidate_topk"], ligand_vocab_size=self.bundle.ligand_vocab_size,
            ligand_feature_dim=self.bundle.ligand_feature_dim,
        ).eval()
        checkpoint_path = os.environ.get("LQRBIND_LEGACY_CHECKPOINT")
        if not checkpoint_path:
            self.skipTest("Set LQRBIND_LEGACY_CHECKPOINT to the original checkpoint")
        original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        legacy.load_state_dict(original["model"], strict=True)
        for name, dataset in self.datasets():
            with self.subTest(dataset=name):
                before = evaluator.predict(legacy, dataset, 36, torch.device("cpu"))
                after = evaluator.predict(self.model, dataset, 36, torch.device("cpu"))
                np.testing.assert_array_equal(before["probabilities"], after["probabilities"])

    def datasets(self):
        return [("test18", self.bundle.test18), ("apo_test", self.bundle.apo),
                ("conformational_test", self.bundle.conformational)]


if __name__ == "__main__":
    unittest.main()
