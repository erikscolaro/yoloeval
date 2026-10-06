# Ablation: metodo di calibrazione dell'int8

Con `int8_full` (QDQ di onnxruntime, tutti i nodi in int8 tranne Softmax e coda della testa)
l'int8 e' veloce sul Pi (22.0 ms contro 28.0 di fp32, `experiments/quantization/int8_backend`), ma
con la calibrazione MinMax globale perde 0.10 di mAP50-95; NNCF (OpenVINO) solo 0.035. Qui si
confrontano i metodi di calibrazione di onnxruntime sullo stesso grafo.

- Prima: `run.py stage=train`, `adapt`, `search` e `finetune` con gli override di `COMMON` in `run.py`.
- Modello: yolo26p potato finetunato (`experiments/benchmark/pruned_finetuned`); mAP sul val di Svanstrom.
- Metodi (`conf/quantization/int8_full*.yaml`, parametri `calibration_method` e
  `calibration_options`): MinMax globale (`int8_full_minmax`, allora `int8_full`), MinMax con media mobile
  (`int8_full_ma`, la piu' vicina al MEAN_MINMAX di NNCF), Percentile 99.999
  (`int8_full_pct`), Entropy/KL (`int8_full_entropy`). Stesse 512 immagini di calibrazione.
- Solo accuratezza, sulla workstation: la calibrazione cambia le scale, non il grafo, quindi
  la latenza resta quella di int8_full. mAP con `model.val()` dell'ONNX su CPU, stesse soglie
  della mAP sul Pi (conf 0.001, iou 0.7, max_det 300). Riferimenti: fp32 e OpenVINO int8 (NNCF,
  misurato sul Pi).
- Script: `run.py`, `--report` per i soli grafici. Risultati in `results/results.json`, grafici in `figures/`.

## Esito storico (AOD4)
Media mobile e Percentile 99.999 recuperano 0.056 di mAP50-95 sul MinMax (0.519 / 0.518 contro 0.463, fp32 0.561, NNCF 0.526); Entropy di `quantize_static` degenere (= MinMax); la media mobile e' diventata il default di `int8` e `int8_full`; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
