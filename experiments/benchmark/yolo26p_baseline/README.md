# yolo26p non potato su Raspberry Pi 5 (OpenVINO)

Stesso setup di `experiments/benchmark/pruned_finetuned`, ma con yolo26p non potato: serve a vedere
quanto il pruning (ops ~49%, params ~40%) migliora la latenza sul Pi.

- Prima: `run.py stage=train model=yolo26p train=scratch_musgd` (AOD4) e `stage=adapt`
  (Svanstrom).
- Modello: yolo26p `scratch_musgd` adattato a Svanstrom (baseline: pesi di `stage=adapt`).
- OpenVINO fp32, fp16, int8; Pi a 2.4 GHz governor performance (`freq_target=max`), 4 core
  (`cpu_4`), scheduling FIFO 80, attesa termica fino a 50 °C (max 5 min); RAM del Pi campionata
  prima e durante la latenza (`results/ram.json`).
- Script: `run.py` (`--after PID` per partire alla fine dell'esperimento sul
  potato, `--report` per la sola tabella).

## Esito storico (AOD4)
Il pruning rende 1.25× piu' veloce (fp32 27.95 → 22.36 ms), int8 piu' lento di fp32; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
