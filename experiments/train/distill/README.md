# Distillazione del pico dal nano

Il pico (`yolo26p`, 678,176 parametri, 1.7 GFLOPs) allenato con il nano come teacher (`yolo26n`, 2,505,360 parametri,
5.9 GFLOPs) migliora, su AOD4 e poi su Svanström dopo l'adapt? Esperimento preliminare: un solo seed.

- Teacher: il nano della tesi su AOD4, `artifacts/weights/yolo26n_aod4_e250_00a6209e/best.pt` (`train=thesis_baseline`,
  importato con `scripts/import_thesis_baselines.py`).
- Student: `yolo26p` da zero con la ricetta di `train=scratch_musgd` (MuSGD lr0 0.01, 250 epoche, batch 64, mosaic 1.0,
  `close_mosaic` 10); cambia solo `+train.distill_model=<teacher>`, passato tale e quale a `model.train()` (nessuna
  modifica a `conf/` o `src/`). Il path del teacher entra nel `training_key`: va scritto sempre come in `run.py`.
- Distillazione di Ultralytics 8.4 (`ultralytics/nn/distill_model.py`): sulle tre feature del neck che entrano nella
  Detect (layer 16, 19, 22, P3-P5, uguali per teacher e student) un proiettore 1x1 (Conv-ReLU-Conv, 129,920
  parametri, solo per il training) porta i canali del pico (32, 64, 128) a quelli del nano (64, 128, 256); perdita L2
  pesata per pixel con lo score massimo del teacher (media delle teste one-to-many e one-to-one), peso `dis` 6.0 (default).
  Il `best.pt` salvato contiene solo lo student: export, adapt e benchmark lo vedono come un pico normale.
- Adapt su Svanström con `conf/adapt/default.yaml` (MuSGD lr0 1e-4, mosaic 1, warmup 2, 40 epoche) e `workers` 6,
  `patience` 15 come il pico adattato di riferimento (`c3b39818`); nessuna distillazione nell'adapt. `stage=adapt`
  riceve lo stesso override `+train.distill_model` solo per trovare i pesi (`training_key`).

## Comandi

```bash
.venv/bin/python experiments/train/distill/run.py --smoke   # controllo: 2 epoche di train, 1 di adapt
.venv/bin/python experiments/train/distill/run.py           # train 250 epoche, poi adapt
.venv/bin/python experiments/adapt/model_size/eval.py p=artifacts/adapt/yolo26p_svanstrom_all_c3b39818/best.pt \
    p_distill=artifacts/adapt/yolo26p_svanstrom_all_<slug>/best.pt   # confronto omogeneo, slug nel log dell'adapt
```

Log in `runs/` (gitignored): `smoke_train.log`, `smoke_adapt.log`, `train.log`, `adapt.log`. Cartelle:

- train: `artifacts/weights/yolo26p_aod4_e250_2fa35a2d` (smoke: `yolo26p_aod4_e2_ec78304b`);
- adapt: `artifacts/adapt/yolo26p_svanstrom_all_<slug>`, nel log (`adattamento completato: ...`).

Gli artefatti dello smoke (`e2_ec78304b` e il suo adapt di 1 epoca) restano in cache con `epochs` 2 e 1 nella
`meta.json` e non si confondono con il run vero: si possono cancellare a mano (con le loro voci in `index.json`).

Lo smoke va guardato per: che la DistillationModel si costruisca (nel log la colonna `dis_loss` accanto a
`box_loss`/`cls_loss`), che la `dis_loss` scenda e non sia NaN, che la validazione finale e l'adapt partano dal `best.pt`
distillato, e il tempo per epoca contro il pico senza teacher (in `run/results.csv`, colonna `time`) per stimare il
run intero.

## Come si legge

Su AOD4 si confronta la mAP del training (val di AOD4, `meta.json`) con `e5930f9c`; su Svanström la misura di `eval.py`
(val e test, mAP a `conf` 0.001, F1 per classe alla `conf` del picco F1 sul val). Un solo seed: differenze sotto circa
0.005 di mAP50-95 sono nel rumore.

## Esito

Training su AOD4 (val AOD4), 250 epoche:

| modello | mAP50 / mAP50-95 | tempo |
|---|---|---|
| yolo26p (`e5930f9c`) | 0.960 / 0.649 | 276 min |
| yolo26p distillato (`2fa35a2d`) | | |
| yolo26n teacher (`00a6209e`) | 0.969 / 0.684 | |

Adapt su Svanström:

| modello | val tutte mAP50 / mAP50-95 | val drone F1 / mAP50-95 | test tutte mAP50 / mAP50-95 | test drone F1 / mAP50-95 |
|---|---|---|---|---|
| yolo26p adattato (`c3b39818`) | 0.861 / 0.470 | | 0.825 / 0.465 | |
| yolo26p distillato adattato | | | | |

**Stato**: da lanciare.
