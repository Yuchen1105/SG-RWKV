# HS3R-Net

This is the anonymous review repository for **HS3R-Net**, a superpixel-guided
RWKV network for polyp segmentation.

The package contains the anonymized implementation, processed manuscript
figures and tables, clean YAML configs, and minimal train/evaluation entry
points. Raw local experiment paths, machine names, and author information are
not included.

## Repository Structure

```text
configs/              YAML configs for training and evaluation
src/                  anonymized model, data, metric, and Lightning code
tools/                clean command-line entry points
assets/figures/       processed manuscript figure assets
assets/tables/        processed tables and CSV summaries
docs/                 method and reproducibility notes
paper/                anonymous manuscript PDF
checkpoints/          placeholder for released weights
```

## Installation

```bash
conda create -n hs3r-review python=3.10 -y
conda activate hs3r-review
pip install -r requirements.txt
```

The CUDA WKV extension is compiled on first use. If compilation is unavailable,
use `--cpu` for a lightweight smoke test.

## Dataset Preparation

Place public polyp segmentation datasets under `data/`:

```text
data/Kvasir-SEG/images
data/Kvasir-SEG/masks
data/CVC-ColonDB/images
data/CVC-ColonDB/masks
```

See `docs/DATASETS.md` for the expected layout.

## Training

```bash
python tools/train_hs3r.py --config configs/hs3r_default.yaml --dataset Kvasir-SEG --devices 1
```

For the Kvasir-specific setting used in the sensitivity analysis:

```bash
python tools/train_hs3r.py --config configs/hs3r_kvasir.yaml --devices 1
```

## Evaluation

```bash
python tools/evaluate_hs3r.py --config configs/hs3r_kvasir.yaml --checkpoint checkpoints/hs3r_kvasir.ckpt
```

Cross-dataset evaluation with one checkpoint:

```bash
python tools/run_cross_dataset.py \
  --config configs/hs3r_default.yaml \
  --checkpoint checkpoints/hs3r_kvasir.ckpt \
  --targets CVC-ColonDB CVC-ClinicDB \
  --output runs/kvasir_to_colon_clinic.csv
```

## Manuscript Materials

- Figures: `assets/figures/`
- Tables and CSV summaries: `assets/tables/`
- Anonymous manuscript PDF: `paper/anonymous_manuscript.pdf`

## Code Map

- EDSC superpixel construction: `src/models/HS3R_Net_Block.py`
- SGRR receptance reweighting: `src/models/HS3R_Net_SGRC_Base.py`
- Final model wrapper: `src/models/HS3R_Net_V005.py`
- Energy ablations: `src/models/HS3R_Net_SPV005_EnergyVariants.py`
