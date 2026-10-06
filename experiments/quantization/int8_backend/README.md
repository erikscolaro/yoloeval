# int8 sul Raspberry Pi 5: backend o quantizzazione?

Sul Pi l'int8 di OpenVINO era piu' lento di fp32 (32.9 contro 22.4 ms). Qui si separa
l'effetto del backend da quello di quanti nodi sono quantizzati, sullo stesso modello di
`experiments/benchmark/pruned_finetuned` (yolo26p potato, finetune MuSGD).
Prima: `run.py stage=train`, `adapt`, `search` e `finetune` con gli override di `COMMON` in `run.py`.

- Pi a 2.4 GHz governor performance, 4 core (`cpu_4`), FIFO 80, attesa termica fino a 50 °C,
  mAP sul Pi per ogni cella (val Svanstrom).
- Quantizzazioni (stesso calibration set, 512 immagini):
  - `int8`: QDQ di onnxruntime, quantizzati solo Conv, Gemm e MatMul: le SiLU e il resto in
    fp32, con Quantize/Dequantize attorno a ogni conv;
  - `int8_full`: tutti i nodi in int8 tranne la Softmax dell'attention e la coda della testa
    (decode, ultima conv di box e score, sigmoide finale, TopK);
  - OpenVINO `int8` (backend `openvino`): NNCF via Ultralytics, nodi scelti da NNCF.
- Backend: `onnxruntime`, `openvino` e `openvino_qdq` (lo stesso ONNX QDQ di onnxruntime
  convertito in IR: stessa quantizzazione, cambia solo il backend).
- Script: `run.py` (export e benchmark dalla pipeline), `--report` per tabella e
  grafici. Log in `runs/`, grafici in `figures/`.

## Esito storico (AOD4)
Conta quanti nodi sono int8: onnxruntime int8_full 22.0 ms contro 44.2 dell'int8 solo conv e 28.0 di fp32; OpenVINO e' lento sul QDQ (49.0 ms con int8_full); mAP50-95 0.46 con MinMax contro 0.53 di NNCF; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
