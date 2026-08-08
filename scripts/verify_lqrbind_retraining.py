#!/usr/bin/env python3
"""Compare a completed deterministic retraining run with the packaged reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory produced by train_lqrbind.py")
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "configs" / "lqrbind_retraining_reference.json",
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    failures = []

    actual_sha = sha256(args.run_dir / "checkpoint.pt")
    if actual_sha != reference["checkpoint_sha256"]:
        failures.append(f"checkpoint SHA-256 differs: {actual_sha}")
    if int(metrics["best_epoch"]) != int(reference["best_epoch"]):
        failures.append(f"best_epoch differs: {metrics['best_epoch']}")
    if abs(float(metrics["best_threshold"]) - float(reference["best_threshold"])) > args.tolerance:
        failures.append(f"best_threshold differs: {metrics['best_threshold']}")
    validation_auc = float(metrics["validation"]["auroc"])
    if abs(validation_auc - float(reference["validation_auc"])) > args.tolerance:
        failures.append(f"validation AUC differs: {validation_auc}")

    for name, target in reference["datasets"].items():
        actual_auc = float(metrics[name]["auroc"])
        if abs(actual_auc - float(target["auc"])) > args.tolerance:
            failures.append(f"{name} AUC differs: {actual_auc}")

    if failures:
        print("Deterministic retraining verification: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Deterministic retraining verification: PASS")
    print(f"checkpoint_sha256={actual_sha}")
    for name in reference["datasets"]:
        print(f"{name}_auc={metrics[name]['auroc']:.15f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
