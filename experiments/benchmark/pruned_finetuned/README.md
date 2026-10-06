# yolo26p potato su Raspberry Pi 5 (OpenVINO)

Latenza, mAP e RAM su Raspberry Pi 5 del yolo26p potato e finetunato con MuSGD
(`experiments/finetune/lr_scheduler`, mAP50 0.923 in training).

- Prima: `run.py stage=train`, `adapt`, `search` e `finetune` con gli override di `COMMON` in `run.py`; i pesi sono quelli della
  pipeline (`final_weights`).
- Modello: yolo26p `scratch_musgd` (AOD4), adattato a Svanstrom, search `pit_duccio_pico`
  (N dal probe a 4 core (`strategy.n_source=cpu_4`)), finetune MuSGD 100 epoche su Svanstrom.
- Export e benchmark dalla pipeline (`run.py`): OpenVINO in fp32, fp16 e int8; Pi a 2.4 GHz con
  governor performance (`freq_target=max`), 4 core (`cpu_4`), scheduling FIFO 80, attesa
  termica fino a 50 °C (max 5 min).
- RAM: RAM usata dal Pi campionata ogni secondo, prima del lancio, durante la latenza
  (`benchmark_app`) e durante la mAP (`eval_map.py`); indicazione, non misura del solo modello.
- Script: `run.py` (tutto), `--report` per la sola tabella; RAM in `results/ram.json`.

## Esito storico (AOD4)
OpenVINO fp32 22.4 ms (44.7 FPS), fp16 uguale, int8 piu' lento (32.9 ms, +47%); dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
