# yolo26n su Raspberry Pi 5 (OpenVINO fp32)

Riferimento per i modelli pico (`experiments/rpi5_finetuned_musgd`,
`experiments/rpi5_yolo26p_baseline`): yolo26n allenato su AOD4, stesso setup.

- Modello: run della tesi `baseline-yolo26n` (pesi COCO, 250 epoche su AOD4, mAP50 0.969,
  mAP50-95 0.684 nel val del training); in cache con una chiave sua (`+train.source=thesis_baseline`).
- OpenVINO fp32, Pi a 2.4 GHz governor performance (`freq_target=max`), 4 core (`cpu_4`),
  scheduling FIFO 80, attesa termica fino a 50 °C (max 5 min); RAM del Pi in `ram.json`.
- Script: `run_experiment.py`, `--report` per la sola tabella.

## Risultati

| precisione | latenza mediana (ms) | media (ms) | max (ms) | FPS medi | mAP50 | mAP50-95 | RAM del Pi prima → latenza (MB) | temp. inizio→fine (°C) |
|---|---|---|---|---|---|---|---|---|
| fp32 | 57.02 | 57.44 | 68.65 | 17.4 | 0.967 | 0.673 | 272 → 406 | 48→63 |

Confronto in fp32 (latenza mediana, stesso setup):

| modello | parametri | MAC | latenza (ms) | FPS | speedup su yolo26n |
|---|---|---|---|---|---|
| yolo26n | 2.45 M | 2.74 G | 57.02 | 17.5 | 1× |
| yolo26p (snapshot ep. 36) | 0.66 M | 0.73 G | 27.95 | 35.8 | 2.04× |
| yolo26p potato | 0.29 M | 0.36 G | 22.36 | 44.7 | **2.55×** |

yolo26n e' molto piu' veloce dei 104.6 ms OpenVINO riportati da Ultralytics per il Pi 5: qui si
misura la sola inferenza con benchmark_app, 4 thread a 2.4 GHz. Con 3.7× le MAC del pico
non potato e' solo 2× piu' lento, a conferma che i modelli piccoli sono limitati dalla memoria.
