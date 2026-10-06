# Ablation: risoluzione di training e di validazione

Le immagini di AOD4 sono 1024x1024 e finora si allena e si valuta a 640 (resize 0.625 nel
dataloader): gli oggetti piccoli perdono dettaglio. Qui: conviene allenare e/o valutare a 1024?

- Prima: `run.py stage=train model=yolo26p train=scratch_musgd` e `experiments/train/close_mosaic/run.py`
  (il confronto a 640+15 usa il suo best.pt). Training e validazione su AOD4 (`pretrain_dataset`).
- Training a 1024: yolo26p da zero, `conf/train/scratch_musgd_1024.yaml` (come `scratch_musgd`:
  MuSGD, lr0 0.01, batch 64, 250 epoche; `imgsz` 1024 e `close_mosaic` 25).
- Riferimenti a 640 gia' allenati: `scratch_musgd` (250 epoche, `close_mosaic` 10) e la sua
  estensione con 15 epoche senza mosaic (`experiments/train/close_mosaic`, 25 senza
  mosaic in tutto come il training a 1024).
- Validazione incrociata: `model.val()` del best.pt di ogni training a 640 e a 1024 (GPU,
  default di Ultralytics: batch 32, rect).
- Script: `run.py`, `--report` per i soli grafici. Log in `runs/`,
  risultati in `results/results.json`, grafici in `figures/`: mAP50 e mAP50-95 per epoca, matrice train x val.

## Esito storico (AOD4)
Nessun risultato: l'esperimento era ancora in corso; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
