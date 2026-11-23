# Costo del resize del crop adattativo sul Raspberry Pi 5

Quanto costa sul Pi preparare l'ingresso del modello con il crop adattativo (`src/crop_tracker.py`): taglio del
quadrato, resize a 256×256 e pre-processing (uint8 HWC → float32 NCHW / 255). Confronto con lo SCAN, cioè il fit del
fotogramma intero a 640 più il pre-processing.

- `misura.py`: OpenCV a un thread su un core, 500 iterazioni dopo 50 di riscaldamento, fotogrammi 1280×720 e
  1920×1080, lato del crop 128–720 px, INTER_AREA e INTER_LINEAR per ridurre (per ingrandire sempre LINEAR).
- `rifai.sh`: lock del Pi, governor performance, 3 passate, `results/crop_resize.jsonl`; rimette il governor.

## Risultati (19/11/2026, mediana, le 3 passate differiscono di meno di 0,05 ms)

| caso | 1280×720 AREA | 1280×720 LINEAR | 1920×1080 AREA | 1920×1080 LINEAR |
|---|---|---|---|---|
| SCAN: fit a 640 + pre | 6,7 ms | 2,6 ms | 11,5 ms | 2,9 ms |
| crop 128 → 256 (zoom 2×) | 0,56 ms | 0,56 ms | 0,56 ms | 0,56 ms |
| crop 256 → 256 (niente resize) | 0,32 ms | 0,32 ms | 0,32 ms | 0,32 ms |
| crop 384 → 256 | 1,3 ms | 0,50 ms | 1,3 ms | 0,51 ms |
| crop 512 → 256 | 0,49 ms | 0,60 ms | 0,50 ms | 0,61 ms |
| crop 720 → 256 | 3,7 ms | 0,84 ms | 3,8 ms | 0,91 ms |

## Commento

- Un crop costa da 0,3 a 0,9 ms con LINEAR. Il crop non dipende dalla risoluzione del video.
- INTER_AREA è lento quando il fattore di riduzione non è intero: 384 e 720 → 256 costano 1,3 e 3,7 ms. Con un
  fattore intero (512 → 256) AREA è veloce come LINEAR.
- Il costo più alto è lo SCAN con AREA: 11,5 ms a 1080p. Con LINEAR lo SCAN costa 2,9 ms.
- Rischio non misurato: la mAP di `experiments/tracking/crop_map` usa AREA per ridurre. LINEAR con un fattore alto
  può perdere dettagli degli oggetti piccoli; prima di cambiare l'interpolazione, misurare di nuovo la mAP.
