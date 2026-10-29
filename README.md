# LQRBind

English | [简体中文](docs/README_zh.md)

**Ligand-Conditioned Quantum-Inspired Multi-View Learning for RNA Binding Site Prediction**

This repository provides the official PyTorch implementation and reproducibility package for LQRBind. LQRBind predicts binding residues conditioned on an RNA-associated candidate ligand set. It combines multi-view RNA graph encoding, low-rank Tree Tensor Network (TTN) fusion, and quantum-inspired structural encoding (QSE) to output one binding probability for each residue in the input RNA.

<p align="center">
  <img src="assets/lqrbind_overview.png" width="100%" alt="Overview of the LQRBind framework">
</p>

<p align="center"><em>Overview of LQRBind. Residue and candidate-ligand inputs are encoded over three RNA graph views, hierarchically fused by a low-rank TTN with adaptive gating, and scored by ligand-conditioned QSE.</em></p>

## Highlights

- **Multi-view RNA representation:** jointly models sequence adjacency, secondary-structure contacts, and long-range contacts.
- **Hierarchical cross-view fusion:** uses a low-rank TTN and adaptive gating to integrate complementary RNA relations.
- **Quantum-inspired structural encoding:** represents residue environments through amplitude--phase states and learned density states.
- **Ligand-set-conditioned prediction:** aggregates molecular features of RNA-associated candidate ligands to refine residue-level binding scores.
- **Reproducibility materials:** includes processed benchmark data, model weights, training configurations, a candidate-source inventory, and reference prediction checks.

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

Verify the release files and reproduce inference from the default checkpoint, `checkpoints/lqrbind_seed497.pt`:

```bash
bash scripts/reproduce.sh
```

The command performs two checks:

1. `scripts/verify_package.py` validates the checksums of files listed in the manifest and the structure of the default checkpoint.
2. `scripts/evaluate_lqrbind.py` rebuilds the model inputs, regenerates residue-level prediction probabilities, and computes classification metrics using the dataset-specific thresholds in `configs/lqrbind_paper_metrics.json`.

The reproduced metrics are written to `results/reproduced_metrics.csv`. To run the evaluator directly:

```bash
python scripts/evaluate_lqrbind.py --device auto --save-predictions
```

### Model weights and candidate provenance

The `checkpoints/` directory provides model weights for seeds 148, 184, 224, 389, and 497.

The inference model configuration is provided in `configs/lqrbind_paper_checkpoint.json`.

`data/metadata/candidate_sources.csv` lists test inputs, candidate-ligand components, and their source records. The RNA–ligand association metadata used for candidate construction are provided in `data/metadata/rna_ligand_metadata.csv`.

### Results reported in the paper

The following table contains the results reported in the paper, also stored in `configs/lqrbind_paper_metrics.json`.

| Dataset | Precision | Recall | MCC | AUC |
|---|---:|---:|---:|---:|
| Test18 | 0.662 | 0.454 | 0.375 | 0.771 |
| Apo test | 0.467 | 0.553 | 0.420 | 0.805 |
| Conformational test | 0.746 | 0.552 | 0.413 | 0.759 |

LQRBind contains 2,541,513 trainable parameters.

## Training

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
├── checkpoints/             # Default model and four supplementary-analysis checkpoints
├── configs/                 # Evaluation and retraining configurations
├── data/                    # Benchmark data, ligand metadata, and candidate-source inventory
├── docs/                    # Chinese README
├── expected_predictions/    # Seed497 reference labels and prediction probabilities
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
