# Pre e post-processing su Raspberry Pi 5 (+ ablation su max_det)

Quanto pesano pre-processing e post-processing rispetto all'inferenza, per il yolo26p potato e
finetunato (`experiments/rpi5_finetuned_musgd`), e quanto conta `max_det` (numero di box in
uscita, fissato nel TopK del grafo all'export).

- Modello: OpenVINO fp32 640x640, due IR dallo stesso `best.pt`: `max_det` 300 (default di
  Ultralytics) e 50 (il val ha al massimo 36 oggetti per immagine, 1.4 in media).
- Pi a 2.4 GHz governor performance, 4 thread, FIFO 80, attesa termica fino a 50 °C.
- 300 immagini del val (seed 0), lette e decodificate prima delle misure, come i frame di una
  camera; 3 giri alternati tra le configurazioni, 900 frame per configurazione.
- Frame 640x640 (camera a 640) e 1024x1024 (le immagini originali, resize nel pre-processing).
- Pre-processing: resize (solo 1024), BGR→RGB, HWC→CHW, /255. Post-processing: soglia di
  confidenza 0.25 e box riportate sull'immagine originale. Niente NMS: la testa one-to-one di
  YOLO26 la evita.
- OpenVINO diretto (API Python) e `model.predict()` di Ultralytics; per controllo,
  `benchmark_app` sui due IR (`bench_app.sh`, `results/benchmark_app.txt`).
- Script: `run_experiment.py` (export, immagini, misure sul Pi con `pi_bench.py` e `bench.sh`,
  grafici), `--report` per i soli grafici. Tempi per frame in `results/timings.csv`.

## Risultati

| configurazione | pre (ms) | inferenza (ms) | post (ms) | end-to-end mediana / p90 / max (ms) | FPS |
|---|---|---|---|---|---|
| OpenVINO, frame 640, max_det 300 | 2.39 | 25.62 | 0.11 | 28.13 / 28.38 / 29.52 | 35.5 |
| OpenVINO, frame 640, max_det 50 | 2.41 | 24.47 | 0.11 | **26.99** / 27.09 / 27.77 | **37.1** |
| OpenVINO, frame 1024, max_det 300 | 3.24 | 25.56 | 0.11 | 28.92 / 29.14 / 33.97 | 34.6 |
| OpenVINO, frame 1024, max_det 50 | 3.25 | 24.39 | 0.11 | 27.74 / 27.81 / 28.83 | 36.0 |
| Ultralytics predict, frame 1024, max_det 300 | 4.32 | 29.63 | 0.42 | 34.70 / 34.98 / 37.29 | 28.8 |
| Ultralytics predict, frame 1024, max_det 50 | 4.31 | 28.50 | 0.42 | 33.55 / 33.66 / 34.07 | 29.8 |

`benchmark_app` (mediana, due passate): max_det 300 22.45 ms, max_det 50 21.90 ms.

- Il post-processing e' trascurabile (0.1 ms; 0.4 ms con Ultralytics). Il pre-processing costa
  2.4 ms con frame a 640 e 3.2 ms con il resize da 1024: circa il 10% del totale.
- L'inferenza dall'API Python costa ~3 ms piu' di `benchmark_app` (25.6 contro 22.4 ms):
  copia dell'input e chiamata sincrona. `model.predict()` aggiunge altri ~4 ms (34.7 contro
  28.9 ms con lo stesso frame).
- `max_det` 50 accorcia l'inferenza di 0.55 ms con `benchmark_app` (-2.5%) e di ~1.1 ms
  dall'API Python, con le stesse detection (1.38 box per frame in tutte le configurazioni): il
  TopK a 300 sulla CPU del Pi pesa piu' del previsto. Anche la coda si stringe (max 27.8
  contro 29.5 ms).
- imread di un jpg 1024x1024 (fuori dalle misure): 4.7 ms di mediana. Temperatura 51 → 69 °C.

![tempi per fase](breakdown.png)
![end-to-end](end_to_end.png)
