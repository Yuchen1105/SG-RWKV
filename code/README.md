# SG-RWKV

This is an anonymous code package for the proposed SG-RWKV/HS3R-Net model.
It contains the core model implementation, a clean training entry point, a clean
evaluation entry point, and one dataset example configuration.

No author information, machine-specific paths, local checkpoint paths, or raw
experiment logs are included.

## Structure

```text
configs/
  hs3r_kvasir_example.yaml    Example config for one Kvasir-SEG-style dataset
src/
  data/                       Image-mask dataset and datamodule
  lightning/                  Lightning training module
  metrics/                    Segmentation metrics
  models/                     SG-RWKV/HS3R-Net model and WKV CUDA sources
  utils/                      Utility functions
tools/
  train_hs3r.py               Training entry point
  evaluate_hs3r.py            Evaluation entry point
checkpoints/
  README.md                   Where to place released checkpoints
```

## Environment

```bash
conda create -n sg-rwkv python=3.10 -y
conda activate sg-rwkv
pip install -r requirements.txt
```

The CUDA WKV extension is compiled on first use from
`src/models/wkv_op.cpp` and `src/models/wkv_cuda.cu`. For a lightweight CPU
smoke test, add `--cpu` to the commands below.

## Dataset Example

The example config expects a Kvasir-SEG-style folder:

```text
data/Kvasir-SEG/
  images/
  masks/
```

Image and mask files are matched by filename stem. To use another local path,
either edit `configs/hs3r_kvasir_example.yaml` or pass `--data-root`.

## Train

```bash
python tools/train_hs3r.py \
  --config configs/hs3r_kvasir_example.yaml \
  --dataset Kvasir-SEG \
  --devices 1
```

## Evaluate

Place the checkpoint under `checkpoints/`, then run:

```bash
python tools/evaluate_hs3r.py \
  --config configs/hs3r_kvasir_example.yaml \
  --checkpoint checkpoints/sg_rwkv_kvasir.ckpt \
  --dataset Kvasir-SEG \
  --devices 1
```

## Code Map

- Final model wrapper: `src/models/HS3R_Net_V005.py`
- Superpixel-guided receptance reweighting: `src/models/HS3R_Net_SGRC_Base.py`
- Energy-guided differentiable superpixel construction: `src/models/HS3R_Net_Block.py`
- Energy ablation variants: `src/models/HS3R_Net_SPV005_EnergyVariants.py`
