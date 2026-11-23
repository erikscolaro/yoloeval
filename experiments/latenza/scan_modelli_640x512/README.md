# SCAN a 640×512 (camera Svanström nativa) sul Raspberry Pi 5

**Scopo**: latenza di P- e PN- con ingresso 640×512 (`conf/deploy/640x512.yaml`), cioè la camera Svanström senza resize.
Lo stesso IR si porta a questa forma all'avvio (`model.reshape` di OpenVINO): non serve un secondo modello.
**Stato**: misurato il 23/11/2026.

- `rifai.sh`: pesi casuali, export, benchmark OpenVINO fp16, 4 core, governor performance, 3 passate, lock del Pi.

```bash
experiments/latenza/scan_modelli_640x512/rifai.sh
```

## Risultati

| modello | mediana | massimo delle 3 passate | mAP50-95 (val 640) |
|---|---|---|---|
| P- | 20,5 ms | 25,3 ms | 0,570 |
| PN- | 32,8 ms | 37,3 ms | 0,587 |

- **Budget** (stima): 33,3 ms − pre a 640×512 (circa 2,1 ms, fra 640×384 e 640² di `../modelli_budget_per_ingresso`) −
  post − Kalman con un track (0,21 ms) = circa **31,1 ms**.
- **P- sta nel budget** con circa 6 ms di margine. **PN- no**: già la mediana (32,8 ms) supera il budget.
- Le stime del 23/11 (interpolazione fra 640×384 e 640²) erano giuste: P- circa 21 / 30 ms, PN- circa 33 / 39 ms.

## Limiti

- Il pre a 640×512 non è misurato, è interpolato.
- `benchmark_app` non dà il p99: si usa il massimo di ogni passata.
