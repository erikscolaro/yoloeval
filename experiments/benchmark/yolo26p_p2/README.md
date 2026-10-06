# yolo26p con uscita P2 su Raspberry Pi 5 (OpenVINO fp32)

Latenza del `yolo26p_p2` non potato (adattato a Svanstrom con la ricetta di `conf/adapt/default.yaml`), stesso setup
degli altri benchmark: OpenVINO fp32, Pi a 2.4 GHz governor performance (`freq_target=max`), 4 core (`cpu_4`), scheduling
FIFO 80, attesa termica fino a 50 °C; RAM del Pi in `results/ram.json`.

- Prima: `run.py stage=train model=yolo26p_p2 train=scratch_musgd` (AOD4) e `stage=adapt` (Svanstrom); la pipeline esporta
  il `best.pt` dell'adapt, ma la latenza fp32 non dipende dai pesi.
- Script: `run.py`, `--report` per la sola tabella.
- Riferimenti sullo stesso Pi e setup (`results/`): yolo26p non potato 27.95 ms, yolo26n 57.02 ms (fp32, OpenVINO, cpu_4,
  max, misure del 01/10). Stima prima della misura: circa 41 ms (FLOPs 2.5 contro 1.7), intervallo 38-50 ms.

## Esito

| modello | latenza mediana (ms) | media / max (ms) | FPS | mAP50 / mAP50-95 (Pi) | RAM Pi prima / picco (MB) | temp. (°C) |
|---|---|---|---|---|---|---|
| **yolo26p con P2** (`ccb23f0f`) | **43.14** | 43.44 / 58.69 | 22.9 | 0.878 / 0.469 | 250 / 390 | 48→60 |
| yolo26p non potato (01/10) | 27.95 | | 35.8 | | | |
| yolo26n (01/10) | 57.02 | | 17.5 | | | |

Il P2 costa +54% di latenza sul `yolo26p` (+47% di FLOPs: la stima di circa 41 ms era giusta, un po' sotto) e resta 1.32x piu'
veloce del nano. Con l'adapt il guadagno di accuratezza del P2 sul p non e' chiaro (vedi `../../train/p2`).
Il Pi non riporta i percentili (`benchmark_app`): in analisi resta solo il valore centrale. Log: `runs/bench.log` (gitignored).

**Stato**: misurato (fp32, OpenVINO, 4 core, 2.4 GHz).
