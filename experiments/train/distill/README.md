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
(val, mAP a `conf` 0.001, F1 per classe alla `conf` del picco F1 sul val). Un solo seed: differenze sotto circa
0.005 di mAP50-95 sono nel rumore.

## Esito

Due run con teacher nano, stessa ricetta, cambia solo il peso `dis` della perdita di distillazione (`--dis`, default
Ultralytics 6.0): `dis` 6.0 (`26a1fa7b`) e `dis` 1.0 (`8f90ac31`). Un solo seed per run, quindi differenze sotto circa 0.005
di mAP50-95 sono dentro il rumore.

Training su AOD4 (val AOD4), 250 epoche:

| modello | mAP50 / mAP50-95 | tempo |
|---|---|---|
| yolo26p (`e5930f9c`) | 0.960 / 0.649 | 276 min (altro carico) |
| yolo26p distillato `dis` 6 (`26a1fa7b`) | 0.955 / 0.643 | 149 min |
| yolo26p distillato `dis` 1 (`8f90ac31`) | 0.961 / 0.652 | 148 min |
| yolo26n teacher (`00a6209e`) | 0.969 / 0.684 |  |

Con `dis` 6 il distillato parte indietro (-0.06 di mAP50-95 alle epoche 5-6) e resta di circa -0.005 / -0.007 per tutto il training.
Il teacher non allunga il tempo per epoca (circa 35 s: il training e' limitato dai dati).

Adapt su Svanström (stessa ricetta di `conf/adapt/default.yaml`, patience 15; misura omogenea: `best.pt`, mAP a `conf` 0.001,
`conf` per classe al picco F1 sul val):

| modello | val tutte mAP50 / mAP50-95 | drone val F1 / mAP50-95 |
|---|---|---|
| yolo26p adattato (`c3b39818`) | 0.861 / 0.470 | 0.861 / 0.390 |
| distillato `dis` 6 adattato (`c1f78224`) | 0.852 / 0.458 | 0.843 / 0.395 |
| **distillato `dis` 1 adattato** (`8f4e1ad7`) | **0.884 / 0.481** | 0.852 / **0.409** |

- **`dis` 6 non aiuta:** -0.012 di mAP50-95 sul val rispetto al pico normale.
- **`dis` 1 aiuta poco sul totale e di piu' sul drone:** val +0.011 (oltre la soglia di rumore); sul drone mAP50-95 +0.019 sul val.
  Per il criterio del progetto (migliore in assoluto) vince il distillato `dis` 1. Resta lontano dal nano adattato
  (0.517 sul val) e davanti di poco al P2 (0.476 sul val).
- Bird e airplane: airplane migliora sul val (0.608 contro 0.585 di mAP50-95); bird e' pari.
- Il peso della distillazione conta piu' del fatto di distillare: sono stati provati solo due valori, 6 e 1; non so se 0.5 o 2
  sarebbero meglio. L'effetto resta piccolo e da confermare con piu' seed.

**Stato**: chiuso (due pesi di distillazione provati).
