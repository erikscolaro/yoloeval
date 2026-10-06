# yolo26p MuSGD: 15 epoche in piu' senza mosaic

Il training da zero con MuSGD (`train=scratch_musgd`, `experiments/train/lr_scheduler`)
ha `close_mosaic: 10` su 250 epoche; nella tesi il pico aveva 100 su 1000. Qui si aggiungono 15
epoche senza mosaic (25 in tutto, il 10% finale) per vedere se la mAP sale ancora.

- Prima: `run.py stage=train model=yolo26p train=scratch_musgd` (AOD4); la run si ritrova in
  cache con `find_weights`. Si allena e si valida su AOD4 (`pretrain_dataset`).
- Partenza: `last.pt` all'epoca 250. Training nuovo di 15 epoche (a fine training last.pt non ha
  lo stato dell'optimizer, quindi niente resume): lr costante al valore finale del run
  (lr0 * lrf = 1e-4), niente warmup, mosaic spento, il resto come `scratch_musgd`.
- Fuori dalla cache: pesi e risultati in `runs/run/`; il best.pt del training resta quello a 250.
- Il `results.csv` del training in cache non viene toccato: le 15 epoche sono solo in
  `runs/run/results.csv` (epoche 1-15, 251-265 nei grafici).
- Script: `run.py` (`--after PID` per partire a fine ablation, `--report` per i soli
  grafici).

## Esito storico (AOD4)
Con 15 epoche senza mosaic 0.960 / 0.649 → 0.962 / 0.652 (mAP50 / mAP50-95, epoca 250 → 265); la tesi (1000 epoche) arrivava a 0.662; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
