# Ablation: risoluzione di training e di validazione

Le immagini di AOD4 sono 1024x1024 e finora si allena e si valuta a 640 (resize 0.625 nel
dataloader): gli oggetti piccoli perdono dettaglio. Qui: conviene allenare e/o valutare a 1024?

- Training a 1024: yolo26p da zero, `conf/train/scratch_musgd_1024.yaml` (come `scratch_musgd`:
  MuSGD, lr0 0.01, batch 64, 250 epoche; `imgsz` 1024 e `close_mosaic` 25).
- Riferimenti a 640 gia' allenati: `scratch_musgd` (250 epoche, `close_mosaic` 10) e la sua
  estensione con 15 epoche senza mosaic (`experiments/scratch_musgd_close_mosaic`, 25 senza
  mosaic in tutto come il training a 1024).
- Validazione incrociata: `model.val()` del best.pt di ogni training a 640 e a 1024 (GPU,
  default di Ultralytics: batch 32, rect).
- Script: `run_experiment.py`, `--report` per i soli grafici. Log in `outputs/ablation_imgsz/`,
  risultati in `results.json`. Grafici: mAP50 e mAP50-95 per epoca, matrice train x val.

## Risultati

In corso.
