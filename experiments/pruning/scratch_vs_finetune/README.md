# Ablation: finetune contro training da zero sull'architettura potata

Stessa architettura potata: cambia l'inizializzazione (pesi ereditati o da zero), con le ricette
della pipeline.
Serve a capire se conta l'architettura trovata dalla search o i pesi ereditati.

- Lo script lancia da solo `stage=train` (AOD4), `adapt` e `search` (`pit_duccio_pico`, 50% di
  MAC e params, N dal probe) per yolo26p `scratch_musgd`; gli stadi gia' in cache sono saltati.
- **finetune**: pipeline normale, `stage=finetune` (recipe `finetune`) su Svanstrom dal
  `pruned.pt` della search.
- **scratch**: stesso `pruned.pt` con pesi reinizializzati (init di PyTorch, BN e bias della
  testa come Ultralytics); training con la recipe `train` (scratch_musgd) su AOD4, poi adattamento
  su Svanstrom con la recipe `adapt`, con `PrunedTrainer`.
- Il confronto e' sulle curve di mAP su Svanstrom: epoche di finetune contro epoche di adapt.
  Le due varianti non hanno lo stesso budget (scratch fa anche il training su AOD4).
- Script: `run.py` (tutto), `--report` per i soli grafici. Log, modelli e run di
  Ultralytics in `runs/`, dati della search in `results/search.json`, grafici in `figures/`.

## Esito storico (AOD4)
A 500 epoche finetune 0.959 / 0.645 contro scratch 0.954 / 0.639 (mAP50 / mAP50-95); il distacco scende da +0.034 (epoca 50) a +0.006: conta l'architettura, il finetune arriva prima; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
