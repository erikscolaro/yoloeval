# Dimensione del modello: yolo26 n, s, m contro pico e pico con P2

Adapt su Svanstrom dei baseline AOD4 di yolo26n, s, m (dalla tesi, importati con `scripts/import_thesis_baselines.py`,
`train=thesis_baseline`) con la ricetta di `conf/adapt/default.yaml` (MuSGD lr0 1e-4, mosaic 1, warmup 2, 40 epoche),
uguale per tutte le dimensioni (nessuna conf per modello, non c'e' stato motivo di cambiarla); confronto con il `yolo26p`
e con il `yolo26p` con P2 (`../../train/p2`).

- `run.py` lancia i tre adapt in parallelo (log in `runs/`, gitignored); `eval.py` fa la misura omogenea (val,
  mAP a `conf` 0.001, F1 per classe alla `conf` del picco sul val).
- Costo: parametri e GFLOPs a 640 (nc 4) e latenza sul Pi (OpenVINO fp32, 4 core, 2.4 GHz, da `../../benchmark`).

## Esito

Misura omogenea (`eval.py`): `best.pt` sul val di Svanström, mAP a `conf` 0.001; F1 per classe alla `conf`
del picco sul val. Adapt: ricetta di `conf/adapt/default.yaml`, patience 10, `adapt.workers=6`
per n, s, m; per p e P2 (stessa ricetta, patience 15) i run di `../../train/p2` e `../lr`.

| modello | parametri | GFLOPs | AOD4 puro (mAP50 / mAP50-95) | val tutte mAP50 / mAP50-95 | drone val mAP50-95 | drone val F1 | Pi fp32 (ms) |
|---|---|---|---|---|---|---|---|
| yolo26p (`c3b39818`) | 0.68 M | 1.7 | 0.960 / 0.649 | 0.861 / 0.470 | 0.390 | 0.861 | 27.95 |
| yolo26p con P2 (`ccb23f0f`) | 0.69 M | 2.5 | 0.964 / 0.654 | 0.874 / 0.476 | 0.392 | 0.853 | 43.14 |
| yolo26n (`53a25b61`) | 2.51 M | 5.9 | 0.969 / 0.684 | 0.882 / 0.517 | 0.441 | 0.865 | 57.02 |
| yolo26s (`9890e36b`) | 9.95 M | 22.8 | 0.968 / 0.689 | 0.891 / 0.530 | 0.470 | 0.903 | non misurato |
| yolo26m (`b32d2b1e`) | 21.78 M | 75.0 | 0.971 / 0.694 | 0.894 / 0.538 | 0.479 | 0.898 | non misurato |

Pi: OpenVINO fp32, 4 core, 2.4 GHz, misure del 01/10 (p, n) e del 06/10 (P2). L'AOD4 puro e' il mAP del training sul val di AOD4 (non e'
omogeneo con le colonne di Svanstrom). Adapt: n 17 epoche (best 7), s 14 (best 4), m 13 (best 3).

- **Il nano da solo chiude gran parte del divario:** 0.517 di mAP50-95 sul val contro 0.470 del pico (+0.047); s ed m aggiungono poco
  (0.530 e 0.538).
- **Sul drone** il vantaggio sale con la dimensione (mAP50-95 0.390 → 0.441 → 0.470 → 0.479 da p a m). L'F1 del drone resta intorno a 0.86-0.90 per tutti.
- **Il P2 non compensa la dimensione:** resta ai livelli del pico (0.476 contro 0.470 sul val) a +47% di
  FLOPs e +54% di latenza sul Pi.
- Il nano parte dai pesi COCO ufficiali (a loro volta da Objects365) e il pico da zero: il vantaggio include il pretrain, non solo la
  dimensione. Un solo seed per modello.
- Latenza di s ed m sul Pi non misurata: il m (75 GFLOPs) sarebbe molto sopra il nano.

**Stato**: chiuso (n, s, m adattati e valutati). Il pico distillato dal nano e' in `../../train/distill`.
