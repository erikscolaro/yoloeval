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
- Rischio non misurato: la mAP di `experiments/tracking/crop_zoom_map` usa AREA per ridurre. LINEAR con un fattore alto
  può perdere dettagli degli oggetti piccoli; prima di cambiare l'interpolazione, misurare di nuovo la mAP.

## Crop grandi e potenze di 2 (23/11/2026)

`rifai_grandi.sh` (3 passate, mediana, 1280×720; con 1080p i valori differiscono di meno di 0,2 ms, tranne 1000 px):

| crop → 256 | fattore | AREA | LINEAR |
|---|---|---|---|
| 384 | 1,5 | 1,3 ms | 0,50 ms |
| **512** | **2** | **0,49 ms** | 0,61 ms |
| 640 | 2,5 | 2,5 ms | 0,73 ms |
| 720 | 2,8 | 3,8 ms | 0,85 ms |
| 768 | 3 | 2,9 ms | 0,92 ms |
| 1000 | 3,9 | 6,4 ms | 1,7 ms |
| **1024** | **4** | **1,5 ms** | 1,7 ms |

- AREA è veloce con le potenze di 2 (512 e 1024): costa come LINEAR o meno. Con 1000 px costa 6,4 ms, con 1024 px 1,5 ms.
- Un fattore intero non basta: 768 (fattore 3) costa 2,9 ms.
- LINEAR cresce piano con il lato (da 0,5 a 1,7 ms).
- Nel tracker il lato del crop non è una potenza di 2 (dipende dall'oggetto). Si potrebbe arrotondare il lato a 384/512/1024; non l'abbiamo provato.

## Perché AREA costa di più con alcuni fattori (23/11/2026)

`rifai_area.sh`: solo `cv2.resize` (OpenCV 5.0.0) da un quadrato di lato S a 256, Pi 5, un core, mediana di 2 passate.
`due1` = AREA con fattore 2 esatto, poi LINEAR a 256 (solo S pari ≥ 512).

| S | fattore | AREA | LINEAR | due1 |
|---|---|---|---|---|
| 300 | 1,17 | 1,03 ms | 0,12 ms | — |
| 384 | 1,5 | 0,91 ms | 0,12 ms | — |
| 448 | 1,75 | 1,36 ms | 0,12 ms | — |
| **512** | **2** | **0,06 ms** | 0,16 ms | 0,07 ms |
| 640 | 2,5 | 1,93 ms | 0,16 ms | 0,21 ms |
| 720 | 2,8 | 3,01 ms | 0,19 ms | 0,25 ms |
| 768 | 3 | 2,40 ms | 0,20 ms | 0,29 ms |
| 1000 | 3,9 | 5,26 ms | 0,63 ms | 0,61 ms |
| **1024** | **4** | **0,39 ms** | 0,65 ms | 0,66 ms |
| 1280 | 5 | 4,54 ms | 0,65 ms | 0,93 ms |
| 2048 | 8 | 8,81 ms | 0,69 ms | 2,55 ms |

**Causa** (`modules/imgproc/src/resize.cpp`, OpenCV 5.0.0):
- Fattore 2 su immagine a 1, 3 o 4 canali: percorso SIMD dedicato (`ResizeAreaFastVec`), il più veloce.
- Altri fattori interi: percorso "area fast" senza SIMD, che somma `fattore²` pixel per ogni pixel di uscita con una
  tabella di offset.
- Fattori non interi: percorso generale con tabelle di pesi in virgola mobile, il più lento.
- LINEAR legge solo 4 pixel per uscita: il suo costo cresce piano con S.
- I thread non servono: `parallel_for_` usa `dst.total() / 65536` strisce, cioè 1 per un'uscita 256×256 (la variante
  `area4` costa come `area1`).
- Non spiegato dal sorgente: perché il fattore 4 (0,39 ms) è molto più veloce dei fattori 3 e 5.

**Come attenuare**
1. Lato del crop arrotondato a 256, 512 o 1024: AREA costa 0,06 ms (512) o 0,39 ms (1024).
2. Due passi: AREA 2× esatto, poi LINEAR. Costa 0,2–0,3 ms fino a 768 px, come LINEAR. La qualità non è misurata.
3. Un crop tra 257 e 511 px si porta a 512: da circa 1 ms a 0,06 ms, con l'oggetto più piccolo nel crop.
