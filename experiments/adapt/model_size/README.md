# Dimensione del modello: yolo26 n, s, m contro pico e pico con P2

Adapt su Svanstrom dei baseline AOD4 di yolo26n, s, m (dalla tesi, importati con `scripts/import_thesis_baselines.py`,
`train=thesis_baseline`) con la ricetta di `conf/adapt/default.yaml` (MuSGD lr0 1e-4, mosaic 1, warmup 2, 40 epoche),
uguale per tutte le dimensioni (nessuna conf per modello, non c'e' stato motivo di cambiarla); confronto con il `yolo26p`
e con il `yolo26p` con P2 (`../../train/p2`).

- `run.py` lancia i tre adapt in parallelo (log in `runs/`, gitignored); `eval.py` fa la misura omogenea (val e test,
  mAP a `conf` 0.001, F1 per classe alla `conf` del picco sul val).
- Costo: parametri e GFLOPs a 640 (nc 4) e latenza sul Pi (OpenVINO fp32, 4 core, 2.4 GHz, da `../../benchmark`).

## Esito

Da compilare.

**Stato**: in corso.
