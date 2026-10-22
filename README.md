# LQRBind

English | [简体中文](docs/README_zh.md)

**Ligand-Conditioned Quantum-Inspired Multi-View Learning for RNA Binding Site Prediction**

This repository provides the official PyTorch implementation and reproducibility package for LQRBind. LQRBind predicts RNA small-molecule binding sites at residue level by combining multi-view RNA graph encoding, low-rank Tree Tensor Network (TTN) fusion, quantum-inspired structural encoding (QSE), and candidate-ligand conditioning.

<p align="center">
  <img src="assets/lqrbind_overview.png" width="100%" alt="Overview of the LQRBind framework">
</p>

<p align="center"><em>Overview of LQRBind. Residue and candidate-ligand inputs are encoded over three RNA graph views, hierarchically fused by a low-rank TTN with adaptive gating, and scored by ligand-conditioned QSE.</em></p>

## Highlights

- **Multi-view RNA representation:** jointly models sequence adjacency, secondary-structure contacts, and long-range contacts.
- **Hierarchical cross-view fusion:** uses a low-rank TTN and adaptive gating to integrate complementary RNA relations.
- **Quantum-inspired structural encoding:** represents residue environments through amplitude--phase states and learned density states.
- **Ligand-conditioned prediction:** incorporates molecular features of candidate ligands to refine residue-level binding scores.
- **Reproducible release:** includes processed benchmark data, the paper checkpoint, the training configuration, and exact prediction checks.

## Installation

The package was verified with Python 3.11.11, PyTorch 2.6.0, PyTorch Geometric 2.7.0, and scikit-learn 1.8.0. Evaluation supports CPU and CUDA devices; CUDA is recommended for training.

### Conda

```bash
git clone https://github.com/KajiMaCN/LQRBind.git
cd LQRBind
conda env create -f environment.yml
conda activate lqrbind
```

### pip

Install a PyTorch build compatible with the local CUDA driver first when the default wheel is unsuitable. Then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

Verify the release files and reproduce inference from the provided checkpoint:

```bash
bash scripts/reproduce.sh
```

The command performs two checks:

1. `scripts/verify_package.py` validates file checksums and checkpoint structure.
2. `scripts/evaluate_lqrbind.py` rebuilds the model inputs, regenerates predictions, and compares them with the stored reference outputs.

The reproduced metrics are written to `results/reproduced_metrics.csv`. To run the evaluator directly:

```bash
python scripts/evaluate_lqrbind.py --device auto --save-predictions
```

### Released checkpoint verification

The released seed-497 checkpoint reproduces the following historical operating points:

| Dataset | Precision | Recall | MCC | AUC |
|---|---:|---:|---:|---:|
| Test18 | 0.662 | 0.454 | 0.375 | 0.771 |
| Apo test | 0.467 | 0.553 | 0.420 | 0.805 |
| Conformational test | 0.746 | 0.552 | 0.413 | 0.759 |

LQRBind contains 2,541,513 trainable parameters.

## Training

Removing inactive modules changes the random initialization sequence for training from scratch. The historical `lqrbind_retraining_reference.json` applies only to the pre-cleanup implementation.

Inspect the complete training command without starting a run:

```bash
python scripts/train_lqrbind.py --dry-run
```

Run a one-epoch pipeline check:

```bash
python scripts/train_lqrbind.py --device cuda --smoke-test
```

Start the complete training configuration:

```bash
python scripts/train_lqrbind.py --device cuda
```

## Repository structure

```text
LQRBind/
├── assets/                  # README figures
├── checkpoints/             # Released model checkpoint
├── configs/                 # Evaluation and retraining configurations
├── data/                    # Processed benchmark data and ligand metadata
├── docs/                    # Chinese README
├── expected_predictions/    # Reference outputs for exact verification
├── scripts/                 # Evaluation, training, and verification entry points
├── src/                     # Model, data, loss, metric, and training code
├── tests/                   # Checkpoint compatibility and prediction regression tests
├── environment.yml
└── requirements.txt
```

## Citation

This manuscript is currently under review. Citation information will be updated upon publication.

```bibtex
@unpublished{ma2026lqrbind,
  title  = {Ligand-Conditioned Quantum-Inspired Multi-View Learning for RNA Binding Site Prediction},
  author = {Ma, Zhihao and Wang, Hailing and Chen, Yu and Xi, Xidong and Cao, Guitao and Tan, Yaw Sing and Deng, Lei and Wu, Min},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Issues

If you encounter a problem or have a question, please open an issue in this repository.
