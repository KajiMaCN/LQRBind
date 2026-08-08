from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from losses import LQRBindLoss
from metrics import compute_binary_metrics, select_threshold
from model import LQRBind
from data import build_lqrbind_data_bundle, set_seed


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data", "processed")
DEFAULT_METADATA_CSV = os.path.join(REPO_ROOT, "data", "metadata", "rna_ligand_metadata.csv")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "lqrbind")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the released LQRBind architecture.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ttn-bond-dim", type=int, default=16)
    parser.add_argument("--quantum-state-dim", type=int, default=16)
    parser.add_argument("--ligand-candidate-cap", type=int, default=8)
    parser.add_argument("--candidate-topk", type=int, default=18)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--disable-seq-view", action="store_true")
    parser.add_argument("--disable-sec-view", action="store_true")
    parser.add_argument("--disable-long-view", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_positive_negative(dataset: list) -> tuple[int, int]:
    positives = sum(int(graph.y.sum().item()) for graph in dataset)
    total = sum(int(graph.y.numel()) for graph in dataset)
    return positives, total - positives


def make_loader(dataset: list, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def collect_predictions(model, loader, device: torch.device):
    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).detach().cpu())
            labels.append(batch.y.detach().cpu())

    return {
        "probabilities": torch.cat(probabilities).numpy(),
        "labels": torch.cat(labels).numpy(),
    }


def summarize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "mcc": float(metrics["mcc"]),
        "auc": float(metrics["auroc"]),
    }


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    bundle = build_lqrbind_data_bundle(
        train60_processed_pt=os.path.join(args.data_root, "train60.pt"),
        test18_processed_pt=os.path.join(args.data_root, "test18.pt"),
        apo_processed_pt=os.path.join(args.data_root, "apo_test.pt"),
        conformational_processed_pts=[
            os.path.join(args.data_root, f"conformational_{split_name}.pt")
            for split_name in ("0", "1", "2")
        ],
        metadata_csv_path=args.metadata_csv,
        max_ligand_candidates=args.ligand_candidate_cap,
        seed=args.split_seed,
    )
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    training_dataset = bundle.train

    input_dim = bundle.train[0].x.size(-1)

    train_loader = make_loader(training_dataset, args.batch_size, shuffle=True)
    validation_loader = make_loader(bundle.validation, args.batch_size, shuffle=False)
    test18_loader = make_loader(bundle.test18, args.batch_size, shuffle=False)
    apo_loader = make_loader(bundle.apo, args.batch_size, shuffle=False)
    conformational_loader = make_loader(bundle.conformational, args.batch_size, shuffle=False)

    model = LQRBind(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        ttn_bond_dim=args.ttn_bond_dim,
        quantum_state_dim=args.quantum_state_dim,
        candidate_topk=args.candidate_topk,
        ligand_vocab_size=bundle.ligand_vocab_size,
        ligand_feature_dim=bundle.ligand_feature_dim,
        use_seq_view=not args.disable_seq_view,
        use_sec_view=not args.disable_sec_view,
        use_long_view=not args.disable_long_view,
    ).to(device)

    positives, negatives = count_positive_negative(training_dataset)
    raw_positive_weight = negatives / max(positives, 1)
    positive_weight = 1.0 + (raw_positive_weight - 1.0)
    criterion = LQRBindLoss(positive_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score = float("-inf")
    best_threshold = 0.5
    best_epoch = -1
    history = []
    epochs_without_improvement = 0

    print(
        f"[LQRBind] device={device.type} "
        f"train={len(bundle.train)} val={len(bundle.validation)} "
        f"test18={len(bundle.test18)} apo={len(bundle.apo)} con={len(bundle.conformational)} "
        f"pos_weight={positive_weight:.2f} deterministic={args.deterministic}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * int(batch.y.numel())
            epoch_nodes += int(batch.y.numel())

        validation_outputs = collect_predictions(model, validation_loader, device)
        threshold, validation_metrics = select_threshold(
            validation_outputs["labels"], validation_outputs["probabilities"], criterion="mcc"
        )
        train_loss = epoch_loss / max(epoch_nodes, 1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_threshold": threshold,
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_mcc": validation_metrics["mcc"],
                "validation_auc": validation_metrics["auroc"],
            }
        )

        print(
            f"[LQRBind] epoch={epoch} "
            f"train_loss={train_loss:.4f} val_precision={validation_metrics['precision']:.4f} "
            f"val_recall={validation_metrics['recall']:.4f} val_mcc={validation_metrics['mcc']:.4f} "
            f"val_auc={validation_metrics['auroc']:.4f}",
            flush=True,
        )

        score = validation_metrics["auroc"]
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_threshold = threshold
            best_epoch = epoch
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "threshold": threshold,
                "validation_metrics": validation_metrics,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training never produced a valid validation checkpoint.")

    torch.save(best_state, os.path.join(args.output_dir, "checkpoint.pt"))
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)

    model.load_state_dict(best_state["model"])

    datasets = {
        "validation": validation_loader,
        "test18": test18_loader,
        "apo_test": apo_loader,
        "conformational_test": conformational_loader,
    }
    collected_outputs = {
        name: collect_predictions(model, loader, device)
        for name, loader in datasets.items()
    }
    full_results = {
        name: compute_binary_metrics(outputs["labels"], outputs["probabilities"], best_threshold)
        for name, outputs in collected_outputs.items()
    }
    full_results["best_epoch"] = int(best_epoch)
    full_results["best_threshold"] = float(best_threshold)

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(full_results, handle, indent=2)
    evaluation_rows = []
    for dataset_name in ("test18", "apo_test", "conformational_test"):
        row = {"dataset": dataset_name}
        row.update(summarize_metrics(full_results[dataset_name]))
        evaluation_rows.append(row)
    pd.DataFrame(evaluation_rows).to_csv(
        os.path.join(args.output_dir, "evaluation_summary.csv"), index=False
    )

    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_threshold": best_threshold,
                "validation": summarize_metrics(full_results["validation"]),
                "test18": summarize_metrics(full_results["test18"]),
                "apo_test": summarize_metrics(full_results["apo_test"]),
                "conformational_test": summarize_metrics(full_results["conformational_test"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
