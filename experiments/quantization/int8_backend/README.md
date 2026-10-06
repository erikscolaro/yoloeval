# int8 sul Raspberry Pi 5: backend o quantizzazione?

Sul Pi l'int8 di OpenVINO era piu' lento di fp32 (32.9 contro 22.4 ms). Qui si separa
l'effetto del backend da quello di quanti nodi sono quantizzati, sullo stesso modello di
`experiments/rpi5_finetuned_musgd` (yolo26p potato ep58, finetune MuSGD).

- Pi a 2.4 GHz governor performance, 4 core (`cpu_4`), FIFO 80, attesa termica fino a 50 °C,
  mAP sul Pi per ogni cella (val AOD4).
- Quantizzazioni (stesso calibration set, 512 immagini):
  - `int8`: QDQ di onnxruntime, quantizzati solo Conv, Gemm e MatMul: le SiLU e il resto in
    fp32, con Quantize/Dequantize attorno a ogni conv;
  - `int8_full`: tutti i nodi in int8 tranne la Softmax dell'attention e la coda della testa
    (decode, ultima conv di box e score, sigmoide finale, TopK);
  - OpenVINO `int8` (backend `openvino`): NNCF via Ultralytics, nodi scelti da NNCF.
- Backend: `onnxruntime`, `openvino` e `openvino_qdq` (lo stesso ONNX QDQ di onnxruntime
  convertito in IR: stessa quantizzazione, cambia solo il backend).
- Script: `run_experiment.py` (export e benchmark dalla pipeline), `--report` per tabella e
  grafici. Log in `outputs/rpi5_int8_backend/`.

## Risultati

| backend | precisione | latenza mediana (ms) | FPS | max (ms) | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| onnxruntime | fp32 | 28.04 | 35.7 | 31.25 | 0.900 | 0.561 |
| onnxruntime | int8 (solo conv) | 44.16 | 22.6 | 47.42 | 0.796 | 0.423 |
| onnxruntime | **int8_full** | **22.02** | **45.4** | 24.53 | 0.823 | 0.463 |
| openvino | fp32 | 22.36 | 44.7 | 39.13 | 0.900 | 0.561 |
| openvino | int8 (NNCF) | 32.91 | 30.4 | 50.20 | 0.883 | 0.526 |
| openvino_qdq | int8 (solo conv) | 36.68 | 27.3 | 39.21 | 0.797 | 0.425 |
| openvino_qdq | int8_full | 48.97 | 20.4 | 51.21 | 0.823 | 0.464 |

- **Quanti nodi in int8 conta piu' del backend**: con onnxruntime quantizzare tutto (SiLU, Add,
  Concat, Resize) porta l'int8 da 44.2 a 22.0 ms, -21% rispetto al suo fp32 e alla pari con
  OpenVINO fp32. Con le sole conv in int8 i Quantize/Dequantize attorno a ogni conv costano piu'
  di quanto fanno risparmiare.
- **A parita' di quantizzazione il backend conta**: con le sole conv OpenVINO e' piu' veloce
  (36.7 contro 44.2 ms), con int8_full molto piu' lento (49.0 contro 22.0 ms). Probabilmente il
  plugin CPU ARM di OpenVINO non ha kernel int8 per le operazioni elementwise e le esegue in
  float con conversioni (non verificato). L'int8 di NNCF (32.9 ms) resta il migliore dei suoi.
- **mAP**: la quantizzazione e' la stessa su onnxruntime e openvino_qdq (0.423/0.425,
  0.463/0.464). La calibrazione MinMax globale perde molto (-0.10 / -0.14 di mAP50-95 rispetto a
  fp32) contro NNCF (-0.035); va migliorata la calibrazione (vedi TODO.md).
- In sintesi: sul Pi oggi conviene OpenVINO fp32 (22.4 ms, nessuna perdita); onnxruntime
  int8_full e' altrettanto veloce ma, con questa calibrazione, perde 0.10 di mAP50-95.

![latenza](latency.png)
![mAP](accuracy.png)
