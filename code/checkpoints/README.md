# Checkpoints

Pretrained weights are not included in this anonymous review package to keep
the repository lightweight. Place released checkpoints in this directory, for
example:

```text
checkpoints/
  hs3r_kvasir.ckpt
  hs3r_colondb.ckpt
```

Then run evaluation with:

```bash
python tools/evaluate_hs3r.py --config configs/hs3r_kvasir.yaml --checkpoint checkpoints/hs3r_kvasir.ckpt
```
