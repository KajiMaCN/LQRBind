#!/usr/bin/env python3
"""Reproduce the paper metrics using the saved dataset-specific thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src"
sys.path.insert(0, str(SOURCE_DIR))

from data import build_lqrbind_data_bundle, set_seed  # noqa: E402
from metrics import compute_binary_metrics  # noqa: E402
from model import LQRBind  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "lqrbind_paper_checkpoint.json"
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "lqrbind_seed497.pt")
    parser.add_argument(
        "--thresholds", type=Path, default=ROOT / "configs" / "lqrbind_paper_metrics.json"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "reproduced_metrics.csv"
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def dataset_threshold(reference: dict) -> float:
    """Read the saved evaluation threshold for one dataset."""
    if "threshold" not in reference:
        raise ValueError("Dataset reference is missing its evaluation threshold.")
    threshold = float(reference["threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Dataset threshold must be finite and between 0 and 1.")
    return threshold


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_bundle(config: dict):
    data_root = resolve_path(config["data_root"])
    return build_lqrbind_data_bundle(
        train60_processed_pt=str(data_root / "train60.pt"),
        test18_processed_pt=str(data_root / "test18.pt"),
        apo_processed_pt=str(data_root / "apo_test.pt"),
        conformational_processed_pts=[
            str(data_root / f"conformational_{split}.pt")
            for split in ("0", "1", "2")
        ],
        metadata_csv_path=str(resolve_path(config["metadata_csv"])),
        max_ligand_candidates=int(config["ligand_candidate_cap"]),
        seed=int(config["split_seed"]),
    )


def build_model(config: dict, bundle) -> LQRBind:
    return LQRBind(
        input_dim=bundle.train[0].x.size(-1),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        ttn_bond_dim=int(config["ttn_bond_dim"]),
        quantum_state_dim=int(config["quantum_state_dim"]),
        candidate_topk=int(config["candidate_topk"]),
        ligand_feature_dim=bundle.ligand_feature_dim,
        use_seq_view=not config.get("disable_seq_view", False),
        use_sec_view=not config.get("disable_sec_view", False),
        use_long_view=not config.get("disable_long_view", False),
    )


def predict(model: torch.nn.Module, dataset: list, batch_size: int, device: torch.device) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            probabilities.append(torch.sigmoid(model(batch)).cpu())
            labels.append(batch.y.cpu())
    return {
        "labels": torch.cat(labels).numpy(),
        "probabilities": torch.cat(probabilities).numpy(),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = json.loads(args.thresholds.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]), deterministic=config.get("deterministic", False))
    device = resolve_device(args.device)

    print(f"Loading LQRBind data from {resolve_path(config['data_root'])}")
    bundle = build_bundle(config)
    model = build_model(config, bundle).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"Loaded checkpoint on {device}; evaluation thresholds from {args.thresholds}")

    datasets = {
        "test18": bundle.test18,
        "apo_test": bundle.apo,
        "conformational_test": bundle.conformational,
    }
    display_names = {
        "test18": "Test18", "apo_test": "Apo test", "conformational_test": "Conformational test"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output.parent / "reproduced_predictions"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    passed = True
    for name, dataset in datasets.items():
        target = expected[name]
        threshold = dataset_threshold(target)
        prediction = predict(model, dataset, int(config["batch_size"]), device)
        metrics = compute_binary_metrics(
            prediction["labels"], prediction["probabilities"], threshold
        )
        actual = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "mcc": metrics["mcc"],
            "auc": metrics["auroc"],
        }
        reference_path = ROOT / "expected_predictions" / f"{name}_predictions.npz"
        with np.load(reference_path) as reference:
            label_match = np.array_equal(prediction["labels"], reference["labels"])
            probability_max_abs_diff = float(
                np.max(np.abs(prediction["probabilities"] - reference["probabilities"]))
            )
        differences = {
            key: abs(value - float(target[key]))
            for key, value in actual.items()
        }
        metric_pass = all(value <= args.tolerance for value in differences.values())
        prediction_pass = label_match and probability_max_abs_diff <= args.tolerance
        dataset_pass = metric_pass and prediction_pass
        passed = passed and dataset_pass

        row = {
            "dataset": display_names[name],
            "threshold": threshold,
            "threshold_source": "paper_metrics.dataset",
            **actual,
            "max_metric_abs_diff": max(differences.values()),
            "prediction_max_abs_diff": probability_max_abs_diff,
            "status": "PASS" if dataset_pass else "FAIL",
        }
        rows.append(row)
        print(
            f"{display_names[name]}: threshold={threshold:.9f} "
            f"P={actual['precision']:.6f} R={actual['recall']:.6f} "
            f"MCC={actual['mcc']:.6f} AUC={actual['auc']:.6f} [{row['status']}]"
        )

        if args.save_predictions:
            np.savez_compressed(prediction_dir / f"{name}_predictions.npz", **prediction)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
