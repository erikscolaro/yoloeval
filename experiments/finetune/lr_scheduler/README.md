# Ablation: optimizer del finetune

Finetune di yolo26p potato con tre optimizer, stesso punto di partenza e stessi altri parametri
(100 epoche, batch 64, `close_mosaic` 25, stessa augmentation).

- Prima: `run.py stage=train`, `adapt` e `search` con gli override di `COMMON` in `run.py`
  (yolo26p `scratch_musgd`, `pit_duccio_pico`, N dal probe a 4 core (`strategy.n_source=cpu_4`)); lo script esce se la search manca.
- Partenza: `pruned.pt` della search in cache (parte dall'adattamento a Svanstrom); finetune su
  Svanstrom.
- Script: `run.py <adam|musgd|sgd>` (run in `runs/`), grafici con `plot_finetune.py` (in `figures/`).
- Configurazioni in `config/`: la conf di yoloeval e gli argomenti completi di Ultralytics.

## Esito storico (AOD4)
MuSGD davanti: 0.923 / 0.580 (mAP50 / mAP50-95 a 100 epoche), SGD 0.917 / 0.573, Adam 0.910 / 0.564; adottato `finetune=musgd`; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
