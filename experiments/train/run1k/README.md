# run1k: pico da zero a 1000 epoche, poi adapt e pruning

Un miglior modello di partenza migliora alla fine anche il modello potato e finetunato? Il pico (`yolo26p`, ~0.68 M
parametri) allenato da zero su AOD4 per 1000 epoche invece di 250: migliora l'adapt su Svanström, e migliora il pico
potato (search + finetune su AOD4, poi adapt)? Esperimento preliminare: un solo seed.

- **Fase 0**: `yolo26p` da zero su AOD4 con `conf/train/run1k.yaml`: `scratch_musgd` (MuSGD lr0 0.01, batch 64,
  seed 42, warmup 3, mosaic 1, `scale` 0.5, mixup 0) con 1000 epoche, coseno con `lrf` 0.001 e `close_mosaic` 100
  come la run a 1000 epoche della tesi, `patience` 1000 (niente early stopping). Della tesi non si riprendono
  optimizer (SGD), batch (128) e seed (0).
- **Ramo 1**: `stage=adapt` su Svanström con `conf/adapt/default.yaml`, `workers` 6 e `patience` 15 come il pico di
  riferimento `c3b39818`.
- **Ramo 2**: search PIT/DUCCIO `strategy=pit_duccio_pico` su AOD4 (60 epoche, -50% di MAC e parametri, lr dei pesi
  0.001 costante, batch 64) con N fissato a 8 (`strategy.search.pit.n=8`) come la search del potato di riferimento
  `5b5558e3`: con `n: auto` il probe int8 piu' recente del Pi (01/10) darebbe N=4. Poi finetune del potato su AOD4
  dai pesi ereditati con `conf/finetune/musgd_500.yaml` (gli argomenti del finetune di riferimento, dal checkpoint
  `artifacts/complexity/_onnx/670b7067_640/best.pt`: 500 epoche, MuSGD lr0 0.01, `close_mosaic` 50, warmup 3, `pretrained`,
  augmentation di default). Infine `stage=adapt` del potato su Svanström con `conf/adapt/default.yaml` e `workers` 6
  (patience 10): per le strategie pit l'adapt parte dal `best.pt` del finetune e usa `yolopit.PrunedTrainer`, come
  `../../adapt/pruned_direct`.

## Search, finetune e adapt con la pipeline

La catena delle strategie pit e' `train (AOD4) -> search -> finetune (AOD4) -> adapt (Svanström)`: `stage=search`
parte dal `best.pt` del training e lavora su `pretrain_dataset` (AOD4), `stage=finetune` dal `pruned.pt` della search,
sempre su AOD4, e `stage=adapt` porta il potato finetunato su `dataset` (Svanström). Tutto in cache, nessun
trainer scritto a mano in `run.py`.

Gli stessi override (`strategy=pit_duccio_pico strategy.search.pit.n=8 finetune=musgd_500 adapt.workers=6`) vanno a
search, finetune e adapt: le chiavi si concatenano (`adapt.*` non entra in search e finetune). Cartelle (dalle
chiavi, con i dati di oggi):

| fase | cartella | smoke |
|---|---|---|
| train | `artifacts/weights/yolo26p_aod4_e1000_487d6cd9` | `yolo26p_aod4_e2_37eb60a2` |
| ramo 1, adapt | `artifacts/adapt/yolo26p_svanstrom_all_13ae6c4c` | `yolo26p_svanstrom_all_e8c07b04` |
| ramo 2, search | `artifacts/search/yolo26p_aod4_pit_duccio_pico_c4f54039` | `..._30cb5b0e` |
| ramo 2, finetune | `artifacts/finetune/yolo26p_aod4_pit_duccio_pico_e0262fe5` | `..._6de447ee` |
| ramo 2, adapt | `artifacts/adapt/yolo26p-pit_duccio_pico_svanstrom_all_55ddd518` | `..._c7b70e40` |

Le chiavi cambiano se cambiano i dati (hash del dataset) o yolopit; i nomi veri sono nei log.

## Comandi

```bash
.venv/bin/python experiments/train/run1k/run.py --smoke         # controllo: 2 epoche di train, 1 per ogni fase dopo
.venv/bin/python experiments/train/run1k/run.py                 # tutto: train, ramo1, ramo2
.venv/bin/python experiments/train/run1k/run.py train           # solo la fase 0
.venv/bin/python experiments/train/run1k/run.py ramo1           # solo il ramo 1 (dopo la fase 0)
.venv/bin/python experiments/train/run1k/run.py ramo2           # solo il ramo 2 (dopo la fase 0)
.venv/bin/python experiments/adapt/model_size/eval.py p=artifacts/adapt/yolo26p_svanstrom_all_c3b39818/best.pt \
    p1000=artifacts/adapt/yolo26p_svanstrom_all_311cab66/best.pt \
    run1k=artifacts/adapt/yolo26p_svanstrom_all_13ae6c4c/best.pt \
    potato_run1k=artifacts/adapt/yolo26p-pit_duccio_pico_svanstrom_all_55ddd518/best.pt   # misura omogenea, solo val
```

Gli stadi gia' in cache si saltano: rilanciare dopo un errore riparte dal primo stadio mancante. Log in `runs/`
(gitignored): `train.log`, `adapt.log`, `search.log`, `finetune.log`, `adapt_potato.log` (smoke: con prefisso
`smoke_`). Gli artefatti dello smoke restano in cache (epoche 2 e 1 nelle `meta.json`) e si possono cancellare a mano
con le loro voci in `index.json`.

Tempi stimati: fase 0 circa 10-12 ore (la tesi: 8.8 h, 31.6 s/epoca, con batch 128 e cache in RAM); search circa
1.2 h (la precedente: 60 epoche in 73 min); finetune circa 5 h; ogni adapt 15-35 min.

Confronti, tutti sul val di Svanström (mAP50 / mAP50-95): pico run1k adattato contro `c3b39818` (pico a 250 epoche,
0.861 / 0.470) e `311cab66` (pico a 1000 epoche della tesi, 0.872 / 0.490); potato da run1k adattato contro il potato
di riferimento (`../../adapt/pruned_direct`, 0.858 / 0.467).

## Esito

| modello | AOD4 val mAP50 / mAP50-95 | Svanström val mAP50 / mAP50-95 | drone mAP50-95 | drone F1 |
|---|---|---|---|---|
| pico 250 epoche, adattato (`c3b39818`) | | | | |
| pico 1000 epoche tesi, adattato (`311cab66`) | | | | |
| **pico run1k, adattato** | | | | |
| potato di riferimento, adattato (`pruned_direct`) | | | | |
| **potato da run1k, adattato** | | | | |

| fase | epoche (best) | tempo |
|---|---|---|
| train | | |
| search | | |
| finetune | | |
| adapt pico | | |
| adapt potato | | |

**Stato**: da lanciare.
