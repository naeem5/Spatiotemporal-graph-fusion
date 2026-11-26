# Spatiotemporal graph fusion for image-sequence classification

Code for the paper *Spatiotemporal graph fusion for image-sequence classification*. An appearance expert (one frame, full z-stack) and a spatiotemporal graph expert
(all frames, image patches as graph nodes) are trained separately and combined by a class-wise gate that adds graph evidence to the appearance logits where the appearance
expert is uncertain. The code covers the A549 and NCM460 live-cell recordings and three public datasets (LCTP, PASTIS, HyperKvasir).

## Installation

```bash
git clone https://github.com/naeem5/Spatiotemporal-graph-fusion.git
cd Spatiotemporal-graph-fusion
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.3+. Training needs a GPU; the tests run on a CPU.

## Repository structure

- `src/pdgraph/`: library shared by all pipelines: appearance expert, graph expert, gate, temporal-only control experts, datasets, metrics, run bookkeeping
- `scripts/`: A549 pipeline (Stage 1, Stage 2, Stage 3, evaluation, comparison with the control expert); `configs/` holds the settings of the paper
- `external_datasets/`: LCTP, PASTIS and HyperKvasir
- `ncm460/`: second cell line
- `baselines/`: 3D ResNet-34 comparison network
- `data/`: split table and compound table of the A549 recordings
- `tests/`: smoke tests on synthetic data

## Usage

Check the installation (CPU, synthetic data, a few minutes):

```bash
python tests/run_smoke_test.py
```

Train and evaluate on A549 (settings of the paper in `configs/`, run folders are written to `runs/`):

```bash
python scripts/01_train_stage1_appearance.py --config configs/a549_stage1_appearance.yaml --data_root /path/to/recordings
python scripts/02_train_stage2_graph.py --config configs/a549_stage2_graph.yaml --data_root /path/to/recordings
python scripts/03_train_stage3_gate.py --config configs/a549_stage3_gate_seed1234.yaml --data_root /path/to/recordings \
    --app_ckpt runs/stage1_appearance_<time>/best_model.pth --graph_ckpt runs/stage2_graph_<time>/best_model.pth
python scripts/04_evaluate.py --data_root /path/to/recordings --fusion_ckpt runs/stage3_gate_<time>/best_model.pth --split test --save_npz results/stage3_seed1234_test.npz
```

Repeat Stage 3 with the configs of seeds 42 and 2025. `--graph_arch temporal` in scripts 02-04 trains a control expert without spatial edges, which
`scripts/05_compare_graph_vs_temporal.py` compares with the patch graph. All options are listed with `--help`.

Public datasets and the second cell line have their own instructions: [external_datasets/README.md](external_datasets/README.md) and [ncm460/README.md](ncm460/README.md).

## Data

- The A549 recordings (33,670 recordings) are available from the corresponding author on reasonable request. The scripts expect one `.npz` file per recording and channel
  (`mito`, `mmp`, `nuclear`) with an array `img` of shape (frames, z-planes, height, width); `data/split_table.csv` gives the split, compound and class of every recording and
  its paths relative to `--data_root`.
- The public datasets are not redistributed; download links and licences are in `external_datasets/README.md`.
