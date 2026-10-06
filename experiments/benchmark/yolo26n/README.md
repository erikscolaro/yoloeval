# yolo26n su Raspberry Pi 5 (OpenVINO fp32)

Riferimento per i modelli pico (`experiments/benchmark/pruned_finetuned`,
`experiments/benchmark/yolo26p_baseline`): yolo26n della pipeline, stesso setup.

- Prima: `run.py stage=train model=yolo26n train=default` (AOD4) e `stage=adapt` (Svanstrom).
- Modello: yolo26n adattato a Svanstrom (baseline: pesi di `stage=adapt`).
- OpenVINO fp32, Pi a 2.4 GHz governor performance (`freq_target=max`), 4 core (`cpu_4`),
  scheduling FIFO 80, attesa termica fino a 50 °C (max 5 min); RAM del Pi in `results/ram.json`.
- Script: `run.py`, `--report` per la sola tabella.

## Esito storico (AOD4)
Fp32 57.0 ms (17.5 FPS); il pico potato e' 2.55× piu' veloce; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
