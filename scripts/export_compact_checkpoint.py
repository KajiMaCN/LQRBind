#!/usr/bin/env python3
"""Export inference weights without inactive parameters or optimizer state."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import evaluate_lqrbind as evaluator
from model import _is_legacy_unused_key


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError('Choose a new output path; the original checkpoint is preserved.')
    torch.set_num_threads(1)
    original = torch.load(args.input, map_location='cpu', weights_only=False)
    config = json.loads((evaluator.ROOT / 'configs/lqrbind_paper_checkpoint.json').read_text())
    bundle = evaluator.build_bundle(config)
    model = evaluator.build_model(config, bundle)
    model.load_state_dict(original['model'], strict=True)
    active = model.state_dict()
    removed = sorted(set(original['model']) - set(active))
    assert all(_is_legacy_unused_key(key) for key in removed)
    for key, tensor in active.items():
        assert torch.equal(tensor, original['model'][key]), key
    compact = {key: value for key, value in original.items() if key != 'optimizer'}
    compact['model'] = active
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    reloaded = torch.load(args.output, map_location='cpu', weights_only=False)
    # Bypass the compatibility filter: exported keys must match the model exactly.
    torch.nn.Module.load_state_dict(model, reloaded['model'], strict=True)
    for key, tensor in active.items():
        assert torch.equal(tensor, reloaded['model'][key]), key
    report = {
        'source_sha256': digest(args.input), 'compact_sha256': digest(args.output),
        'source_bytes': args.input.stat().st_size, 'compact_bytes': args.output.stat().st_size,
        'parameters': sum(p.numel() for p in model.parameters()),
        'removed_parameter_elements': sum(original['model'][k].numel() for k in removed),
        'removed_keys': removed, 'active_weights_identical': True,
        'strict_load_without_filter': True, 'optimizer_removed': 'optimizer' in original,
        'purpose': 'Inference checkpoint, not an optimizer-resume checkpoint',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'removed_keys'}, indent=2))


if __name__ == '__main__':
    main()
