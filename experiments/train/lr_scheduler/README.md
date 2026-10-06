# Ablation: optimizer del training da zero

Training da zero di yolo26p su AOD4 con tre optimizer, stessi altri parametri (default di
Ultralytics): 250 epoche, batch 64, `lrf` 0.01, warmup 3, `close_mosaic` 10, niente early
stopping, stessa augmentation.

- Dataset: `pretrain_dataset` (AOD4); i pesi si ritrovano in cache con `find_weights`. Se
  `train/close_mosaic` e' stato lanciato, il suo `results.csv` resta quello a 250 epoche.
- Conf: `conf/train/scratch_{adam,musgd,sgd}.yaml`, copiate in `config/` con gli argomenti
  completi di Ultralytics.
- Script: `run.py` lancia i tre training in parallelo con `stage=train` della
  pipeline (cache vera, `artifacts/weights/`), poi grafici (`figures/`) e tabella; `--report` rifà solo questi. Log in `runs/`.

## Esito storico (AOD4)
MuSGD davanti dall'inizio: 0.960 / 0.649 (mAP50 / mAP50-95 alla 250), SGD 0.959 / 0.643, Adam 0.959 / 0.637; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
