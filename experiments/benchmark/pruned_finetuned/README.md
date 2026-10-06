# yolo26p potato su Raspberry Pi 5 (OpenVINO)

Latenza, mAP e RAM su Raspberry Pi 5 del yolo26p potato e finetunato con MuSGD
(`experiments/ablation/finetune/lr_scheduler`, mAP50 0.923 in training).

- Modello: search `pit_duccio_pico` interrotta all'epoca 58 (N=8, ops 48.7%, params 39.8%),
  finetune MuSGD 100 epoche; registrato nella cache come strategia `pit_duccio_pico_ep58`.
- Export e benchmark dalla pipeline (`run.py`): OpenVINO in fp32, fp16 e int8; Pi a 2.4 GHz con
  governor performance (`freq_target=max`), 4 core (`cpu_4`), scheduling FIFO 80, attesa
  termica fino a 50 °C (max 5 min).
- RAM: RAM usata dal Pi campionata ogni secondo, prima del lancio, durante la latenza
  (`benchmark_app`) e durante la mAP (`eval_map.py`); indicazione, non misura del solo modello.
- Script: `run_experiment.py` (tutto), `--report` per la sola tabella; RAM in `ram.json`.

## Risultati

| precisione | latenza mediana (ms) | FPS | mAP50 | mAP50-95 | RAM del Pi prima → latenza (MB) | temp. inizio→fine (°C) |
|---|---|---|---|---|---|---|
| fp32 | 22.36 | 44.7 | 0.900 | 0.561 | 280 → 411 | 48→58 |
| fp16 | 22.55 | 44.3 | 0.900 | 0.561 | 274 → 419 | 53→61 |
| int8 | 32.91 | 30.4 | 0.883 | 0.526 | 275 → 480 | 55→62 |

fp32 e fp16 coincidono (OpenVINO sulla CPU del Pi esegue probabilmente l'IR fp16 in fp32);
l'int8 e' **piu' lento** di fp32 (+47%) e perde 0.017 di mAP50: su questa CPU ARM l'INT8 di
OpenVINO non conviene. Il picco "durante la mAP" non e' riportato: il campionatore lo contava
male in questa run (corretto nello script). p90 assente: benchmark_app non riporta percentili.
