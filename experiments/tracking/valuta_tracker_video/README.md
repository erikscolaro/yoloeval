# Valutazione del tracker sui video

**Scopo**: misurare sugli stessi fotogrammi dei video di `../video_val_test` tre modi: detector da solo (P-), Kalman
(`src/tracking.py`) e tracker con crop (`src/crop_tracker.py`). Risultati separati per Svanström, Anti-UAV e totale,
e per fotogrammi con 0–1 oggetti (`1ogg`, il caso del deploy) e con 2+ oggetti (`2+ogg`).

**Comando**: `python experiments/tracking/valuta_tracker_video/run.py [--split val] [--tag nome] [--p ...] [--ct ...]`.
Le opzioni sono nell'help in testa a `run.py`. Non ha un `rifai.sh` suo: lo lanciano i `rifai.sh` di
`../taratura_kalman` e `../taratura_kalman_crop` con i parametri tarati.

**Metriche**: mAP50 e mAP50-95 per classe e per fascia (con `../crop_zoom_map/run.py`). Al punto di lavoro: recall,
falsi allarmi per fotogramma, fotogrammi senza uscita, quota di fotogrammi in TRACK. Associazione senza classe,
IoU >= 0,5.

## Risultati

In `results/val_tarato_{svanstrom,anti_uav,tutte,crop}.json`. Le tabelle sono nei README delle due tarature
(`../taratura_kalman`, `../taratura_kalman_crop`). Le prove del 19–23/11, fatte prima della taratura, sono state
cancellate il 25/11/2026.
