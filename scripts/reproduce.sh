#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/verify_package.py
python scripts/evaluate_lqrbind.py --device auto --save-predictions
