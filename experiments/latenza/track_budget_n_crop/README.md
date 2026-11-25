# Costo fuori dal modello di un fotogramma TRACK sul Raspberry Pi 5

Quanto costa sul Pi tutto quello che il crop tracker fa in un fotogramma TRACK oltre all'inferenza: il ritaglio dei
crop, il Kalman, il pre-processing e il post-processing. Serve per il budget del modello a 30 fps in `../track_modelli_batch_crop256`:
`budget(N) = 33,3 ms − costo fuori dal modello con N crop`.

- `misura.py`: fotogramma 1920×1080 uint8, N oggetti di 60×40 px che oscillano (±60 px, periodo 3 s). Usa
  `src/crop_tracker.py` e `src/tracking.py` così come sono nel repo. Il modello è finto: per ogni crop restituisce
  la box vera, così i track restano confermati e si misura il percorso TRACK vero (100% dei fotogrammi in TRACK).
  Un core, 2000 fotogrammi dopo 200 di riscaldamento, N = 1, 2, 4, 8.
- `rifai.sh`: copia gli script sul Pi, governor performance, 3 passate, `results/crop_kalman.jsonl` (una riga per N e
  passata); rimette il governor com'era. Va lanciato con il lock del Pi:

```bash
flock runs/.pi.lock experiments/latenza/track_budget_n_crop/rifai.sh   # circa 2 minuti
```

Le parti misurate:

| parte | cosa contiene |
|---|---|
| ritaglio | ritaglio e resize dei N crop a 256×256 (già dentro "tracker") |
| tracker | tutto `CropTracker.step` tranne il modello: ritaglio, Kalman, associazione, riproiezione |
| pre | N crop uint8 HWC → float32 NCHW / 255, l'ingresso dell'export a batch N |
| post | N uscite (64, 6) → soglia di confidenza |
| totale | tracker + pre + post |

## Risultati (20/11/2026, mediana delle 3 passate)

| crop (N) | ritaglio | tracker | pre | post | totale | p99 del totale | budget del modello |
|---|---|---|---|---|---|---|---|
| 1 | 0,27 ms | 0,67 ms | 0,30 ms | 0,01 ms | 0,99 ms | 1,08 ms | 32,3 ms |
| 2 | 0,55 ms | 1,25 ms | 0,63 ms | 0,02 ms | 1,90 ms | 1,95 ms | 31,4 ms |
| 4 | 1,12 ms | 2,38 ms | 1,30 ms | 0,04 ms | 3,72 ms | 3,77 ms | 29,6 ms |
| 8 | 2,31 ms | 4,68 ms | 2,63 ms | 0,08 ms | 7,39 ms | 7,59 ms | 25,9 ms |

- Il costo fuori dal modello è lineare: circa 0,93 ms per crop. Le passate differiscono di meno di 0,15 ms.
- Le stime usate prima (circa 1 ms di ritaglio e 0,1–0,2 ms di Kalman in tutto) erano sotto il vero da N = 2 in su.
- Il fotogramma SCAN non è qui. Con la camera già a 640 il budget è 30,3 ms (`conf/target/rpi5_640.yaml`, pre in
  `../modelli_budget_per_ingresso`). Con una camera 720p, ridotta a 640×360 con bordo fino a 640×384, il budget è 30,7 ms
  (`../scan_budget_720p`: resize, bordo, pre e post costano 2,6 ms). Con 1080p il resize a 640 costa 11,5 ms con
  INTER_AREA e 2,9 ms con INTER_LINEAR (`../track_costo_ritaglio`).
- Il costo del ritaglio per lato del crop (da 0,3 a 3,8 ms) è in `../track_costo_ritaglio`; qui il crop è vicino al
  minimo (128 → 256).

## Limiti

- Il costo del Kalman non dipende dai valori dei parametri (Q, R): vale anche prima della taratura.
- Il ritaglio dipende un poco dalla grandezza della ROI (resize da un lato più grande costa di più). Qui gli oggetti
  sono di 60×40 px e la ROI è vicina al minimo (128 px); con oggetti grandi il ritaglio costa qualcosa in più.
- Un core, come il resto della pipeline in Python. L'inferenza usa i 4 core, ma i due passi non si sovrappongono.
