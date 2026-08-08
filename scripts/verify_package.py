#!/usr/bin/env python3
"""Check package completeness and file integrity without running inference."""

from __future__ import annotations

import hashlib
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

    checkpoint = torch.load(ROOT / "checkpoints" / "lqrbind.pt", map_location="cpu", weights_only=False)
    required_checkpoint_keys = {"model", "optimizer", "threshold", "validation_metrics"}
    missing_keys = required_checkpoint_keys.difference(checkpoint)
    if missing_keys:
        failures.append(f"checkpoint keys missing: {sorted(missing_keys)}")

    if failures:
        print("Package verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Verified {checked} files and checkpoint structure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
