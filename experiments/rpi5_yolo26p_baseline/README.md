# yolo26p non potato su Raspberry Pi 5 (OpenVINO)

Stesso setup di `experiments/rpi5_finetuned_musgd`, ma con yolo26p non potato: serve a vedere
quanto il pruning (N=8, ops ~49%, params ~40%) migliora la latenza sul Pi.

- Modello: `last.pt` del training da zero con MuSGD (`experiments/ablation/train/lr_scheduler`)
  preso a meta' training (epoca in `snapshot.json`); in cache con una chiave sua
  (`+train.snapshot=epoch<N>`), separata da quella del training vero. La mAP non e'
  quella finale: il confronto e' sulla latenza.
- OpenVINO fp32, fp16, int8; Pi a 2.4 GHz governor performance (`freq_target=max`), 4 core
  (`cpu_4`), scheduling FIFO 80, attesa termica fino a 50 °C (max 5 min); RAM del Pi campionata
  prima e durante la latenza (`ram.json`).
- Script: `run_experiment.py` (`--after PID` per partire alla fine dell'esperimento sul
  potato, `--report` per la sola tabella).

## Risultati

Snapshot all'epoca 36 (mAP50 0.925 nel val del training).

| precisione | latenza mediana (ms) | FPS | mAP50 | mAP50-95 | RAM del Pi prima → latenza (MB) | temp. inizio→fine (°C) |
|---|---|---|---|---|---|---|
| fp32 | 27.95 | 35.8 | 0.890 | 0.552 | 271 → 381 | 50→60 |
| fp16 | 28.10 | 35.6 | 0.891 | 0.552 | 276 → 413 | 53→63 |
| int8 | 41.32 | 24.2 | 0.886 | 0.537 | 272 → 479 | 54→64 |

Confronto con il potato (`experiments/rpi5_finetuned_musgd`), latenza mediana:

| precisione | non potato (ms) | potato (ms) | speedup |
|---|---|---|---|
| fp32 | 27.95 | 22.36 | 1.25× |
| fp16 | 28.10 | 22.55 | 1.25× |
| int8 | 41.32 | 32.91 | 1.26× |

Il pruning toglie il 51% delle MAC (0.73 → 0.36 G) e il 56% dei parametri (662K → 290K), ma
la latenza scende solo del 20%: il modello e' in buona parte limitato dagli accessi in memoria.
Come per il potato, fp16 coincide con fp32 e l'int8 e' piu' lento su questa CPU.
