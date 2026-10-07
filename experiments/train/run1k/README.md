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
  augmentation di default). Infine adapt su Svanström con `conf/adapt/default.yaml` e `workers` 6 (patience 10),
  con `yolopit.PrunedTrainer` come `../../adapt/pruned_direct` (`stage=adapt` non carica pesi potati); risultati in
  `runs/adapt_potato/`, fuori dalla cache.

## Search e finetune su AOD4 con la pipeline

`stage=search` parte dal `best.pt` di un adapt (`adapt_key`) e prende i dati da `dataset`; `stage=finetune` usa lo
stesso `dataset`. Con `dataset=aod4` entrambi girano su AOD4, ma serve un adapt su AOD4: lo fa
`conf/adapt/identity.yaml`, un adapt che non cambia i pesi (un'epoca sull'1% del train, SGD con lr 0, tutti i layer
congelati e le BatchNorm in eval, tranne l'ultima conv della testa one2one che non ha BN: Ultralytics vuole almeno
un parametro allenabile). La sua validazione finale riporta la mAP del training; `run.py` controlla che il suo `best.pt`
sia identico, tensore per tensore, a quello della fase 0 prima di lanciare la search. Nessuna modifica a `src/`.

Gli stessi override (`dataset=aod4 adapt=identity strategy=pit_duccio_pico strategy.search.pit.n=8
finetune=musgd_500`) vanno a identita', search e finetune: le chiavi si concatenano. Cartelle (dalle chiavi, con
i dati di oggi):

| fase | cartella | smoke |
|---|---|---|
| train | `artifacts/weights/yolo26p_aod4_e1000_487d6cd9` | `yolo26p_aod4_e2_37eb60a2` |
| ramo 1, adapt | `artifacts/adapt/yolo26p_svanstrom_all_13ae6c4c` | `yolo26p_svanstrom_all_ab896339` |
| ramo 2, identita' | `artifacts/adapt/yolo26p_aod4_e0ef8a1b` | `yolo26p_aod4_21420157` |
| ramo 2, search | `artifacts/search/yolo26p_aod4_pit_duccio_pico_b02d3252` | `..._c68f2c1f` |
| ramo 2, finetune | `artifacts/finetune/yolo26p_aod4_pit_duccio_pico_6fa91ab2` | `..._bcf90e8a` |
| ramo 2, adapt | `experiments/train/run1k/runs/adapt_potato/` | `runs/smoke_adapt_potato/` |

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
    run1k=artifacts/adapt/yolo26p_svanstrom_all_<slug>/best.pt \
    potato_run1k=experiments/train/run1k/runs/adapt_potato/weights/best.pt   # misura omogenea, solo val
```

Gli stadi gia' in cache si saltano: rilanciare dopo un errore riparte dal primo stadio mancante (l'adapt del potato,
fuori dalla cache, si rifa' sempre). Log in `runs/` (gitignored): `train.log`, `adapt.log`, `identity.log`,
`search.log`, `finetune.log` (smoke: con prefisso `smoke_`); l'adapt del potato scrive su stdout e in
`runs/adapt_potato/`. Gli artefatti dello smoke restano in cache (epoche 2 e 1 nelle `meta.json`) e si possono
cancellare a mano con le loro voci in `index.json`.

Tempi stimati: fase 0 circa 10-12 ore (la tesi: 8.8 h, 31.6 s/epoca, con batch 128 e cache in RAM); identita' pochi
minuti; search circa 1.2 h (la precedente: 60 epoche in 73 min); finetune circa 5 h; ogni adapt 15-35 min.

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
