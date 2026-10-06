# yolo26p potato (500 epoche) su Raspberry Pi 5: onnxruntime e OpenVINO a tre frequenze

Benchmark del migliore (mAP50-95) tra finetune e scratch di
`experiments/ablation/pruned/scratch_vs_finetune` (search `pit_duccio_pico`, 50% di MAC e
params, N=8).

- Modello in cache come strategia `pit_duccio_pico_500ep`, N=8 fisso; scelta in `chosen.json`.
- Export e benchmark dalla pipeline: onnxruntime e OpenVINO, fp32 e fp16, frequenze `max`
  (2.4 GHz, performance), `mid` (1.5 GHz) e `low` (1.0 GHz), 4 core (`cpu_4`), FIFO 80, attesa
  termica fino a 50 °C (max 5 min), mAP sul Pi per ogni cella.
- Script: `run_experiment.py` (`--after PID` per partire a fine ablation, `--report` per la
  tabella). Log in `outputs/rpi5_pruned_500ep/`.

## Risultati

Modello scelto: **finetune** (mAP50-95 0.645 nel val del training, contro 0.639 di scratch).
Frequenze effettive verificate in ogni cella: max 2.4 GHz (performance), mid 2.0 GHz e low
1.5 GHz (userspace).

| backend | precisione | frequenza | latenza mediana (ms) | FPS | max (ms) | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| onnxruntime | fp32 | 2.4 GHz | 27.90 | 35.8 | 28.59 | 0.950 | 0.629 |
| onnxruntime | fp32 | 2.0 GHz | 29.88 | 33.5 | 30.41 | 0.950 | 0.629 |
| onnxruntime | fp32 | 1.5 GHz | 35.07 | 28.5 | 35.88 | 0.950 | 0.629 |
| onnxruntime | fp16 | 2.4 GHz | 60.32 | 16.6 | 61.22 | 0.950 | 0.629 |
| onnxruntime | fp16 | 2.0 GHz | 67.41 | 14.8 | 68.20 | 0.950 | 0.629 |
| onnxruntime | fp16 | 1.5 GHz | 82.69 | 12.1 | 83.72 | 0.950 | 0.629 |
| openvino | fp32 | 2.4 GHz | **22.38** | **44.7** | 22.79 | 0.950 | 0.630 |
| openvino | fp32 | 2.0 GHz | 25.87 | 38.7 | 26.80 | 0.950 | 0.630 |
| openvino | fp32 | 1.5 GHz | 33.41 | 29.9 | 33.97 | 0.950 | 0.630 |
| openvino | fp16 | 2.4 GHz | 22.47 | 44.5 | 23.42 | 0.950 | 0.630 |
| openvino | fp16 | 2.0 GHz | 26.00 | 38.5 | 26.48 | 0.950 | 0.630 |
| openvino | fp16 | 1.5 GHz | 33.24 | 30.1 | 33.84 | 0.950 | 0.630 |

- OpenVINO resta il migliore: 22.4 ms a 2.4 GHz (44.7 FPS), fp16 uguale a fp32.
- La latenza scala meno della frequenza: da 2.4 a 1.5 GHz (-37.5% di clock) OpenVINO passa da
  22.4 a 33.4 ms (+49%, contro il +60% atteso se fosse tutto calcolo); con onnxruntime fp32 da
  27.9 a 35.1 ms (+26%): conferma che il modello e' in buona parte limitato dalla memoria.
- onnxruntime fp16 e' piu' del doppio piu' lento di fp32 (60 contro 28 ms): su questa CPU
  onnxruntime non ha kernel fp16 efficienti a 4 core (come nel probe).
- Stessa latenza del potato ep58 a 2.4 GHz (22.4 ms) con mAP molto piu' alta: 0.950 / 0.630 sul
  Pi contro 0.900 / 0.561. La mAP sul Pi e' piu' bassa di quella del training (0.645) perche'
  sul Pi e' a batch 1 con letterbox 640x640.
- Prima esecuzione (2026-10-02, 3:45) interrotta: il Pi si e' riavviato durante onnxruntime fp16
  a 2.4 GHz e `low` era 1.0 GHz, sotto il minimo del Pi 5. Rifatta alle 11:15 con le frequenze
  corrette, senza riavvii (log della prima in `outputs/rpi5_pruned_500ep/experiment_interrotto.log`).

### Con `max_det` 64 (default degli export dal 2026-10-02)

OpenVINO, 2.4 GHz, 4 core; uscita [1, 64, 6] invece di [1, 300, 6]:

| precisione | latenza mediana (ms) | FPS | max (ms) | mAP50 | mAP50-95 | con max_det 300 (ms) |
|---|---|---|---|---|---|---|
| fp32 | **21.90** | **45.7** | 22.43 | 0.949 | 0.630 | 22.38 |
| fp16 | **21.83** | **45.8** | 22.22 | 0.949 | 0.629 | 22.47 |

-0.5 / -0.6 ms come nell'ablation `experiments/ablation/deploy/max_det`, mAP invariata.

