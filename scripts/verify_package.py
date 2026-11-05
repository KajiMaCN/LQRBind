#!/usr/bin/env python3
"""Check package completeness and file integrity without running inference."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        print("MANIFEST.sha256 is missing", file=sys.stderr)
        return 1

    failures = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
        checked += 1

    for seed in (148, 184, 224, 389, 497):
        path = ROOT / "checkpoints" / f"lqrbind_seed{seed}.pt"
        if not path.is_file():
            failures.append(f"missing checkpoint: {path.name}")
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required_checkpoint_keys = {"model", "threshold", "validation_metrics", "evaluation_thresholds"}
        if seed == 497:
            required_checkpoint_keys.add("reference_metrics")
        missing_keys = required_checkpoint_keys.difference(checkpoint)
        if missing_keys:
            failures.append(f"{path.name} keys missing: {sorted(missing_keys)}")
        thresholds = checkpoint.get("evaluation_thresholds", {})
        for name in ("test18", "apo_test", "conformational_test"):
            try:
                value = thresholds[name]
                valid = not isinstance(value, bool) and math.isfinite(float(value)) and 0 <= float(value) <= 1
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                failures.append(f"{path.name}: missing or invalid evaluation threshold for {name}")

    if failures:
        print("Package verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Verified {checked} files and five checkpoint structures: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
