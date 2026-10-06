# Pre e post-processing su Raspberry Pi 5 (+ ablation su max_det)

Quanto pesano pre-processing e post-processing rispetto all'inferenza, per il yolo26p potato e
finetunato (`experiments/benchmark/pruned_finetuned`), e quanto conta `max_det` (numero di box in
uscita, fissato nel TopK del grafo all'export).

- Prima: `run.py stage=train`, `adapt`, `search` e `finetune` con gli override di `COMMON` in `run.py`.
- Modello: OpenVINO fp32 640x640, due IR dallo stesso `best.pt`: `max_det` 300 (default di
  Ultralytics) e 50 (il val ha al massimo 36 oggetti per immagine, 1.4 in media).
- Pi a 2.4 GHz governor performance, 4 thread, FIFO 80, attesa termica fino a 50 °C.
- 300 immagini del val di Svanstrom (seed 0), lette e decodificate prima delle misure, come i frame di una
  camera; 3 giri alternati tra le configurazioni, 900 frame per configurazione.
- Frame 640x640 (camera a 640; i frame nativi di Svanstrom sono 640x512, ridimensionati una volta prima delle misure).
- Pre-processing: BGR→RGB, HWC→CHW, /255. Post-processing: soglia di
  confidenza 0.25 e box riportate sull'immagine originale. Niente NMS: la testa one-to-one di
  YOLO26 la evita.
- OpenVINO diretto (API Python) e `model.predict()` di Ultralytics; per controllo,
  `benchmark_app` sui due IR (`bench_app.sh`, `results/benchmark_app.txt`).
- Script: `run.py` (export, immagini, misure sul Pi con `pi_bench.py` e `bench.sh`,
  grafici), `--report` per i soli grafici. Tempi per frame in `results/timings.csv`, grafici in `figures/`, IR e immagini in `runs/`.

## Esito storico (AOD4)
End-to-end 27.0 ms (37 FPS) con frame 640 e `max_det` 50; pre-processing 2.4 ms, post-processing 0.1 ms; `max_det` 50 toglie ~0.55 ms (benchmark_app); dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
