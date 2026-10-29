#!/usr/bin/env python3
"""Launch the published seed-497 LQRBind training configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRY = ROOT / "src" / "train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "lqrbind_retraining.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "lqrbind_retrained")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run one epoch to validate the training path."
    )
    return parser.parse_args()


def resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(TRAIN_ENTRY),
        "--data-root",
        resolve_path(config.pop("data_root")),
        "--metadata-csv",
        resolve_path(config.pop("metadata_csv")),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
    ]

    for key, value in config.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend([option, str(value)])

    if args.smoke_test:
        command.extend(["--epochs", "1", "--patience", "1"])

    print("Training command:")
    print(" ".join(command))
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if config.get("deterministic", False):
        environment["PYTHONHASHSEED"] = str(config["seed"])
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
