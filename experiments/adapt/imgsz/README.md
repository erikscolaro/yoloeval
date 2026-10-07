# Ablation: risoluzione dell'adattamento su Svanström

Le immagini di Svanström sono 640x512 e gli oggetti hanno lato mediano 35 px (5° percentile 22 px). Adattare a
`imgsz` 1024 (oggetti circa 1.6x piu' grandi, vicini alla scala di AOD4 a 640) migliora la mAP rispetto a 640?

- Run: yolo26p `scratch_musgd`, MuSGD lr0 2e-4, warmup 2, 40 epoche, patience 15, rete intera, `scale: 0`, `mosaic: 0`
  (come `577b097f` di `../lr`), cambia solo `+adapt.imgsz=1024`: slug `f172c7c3`. Log: [`runs/imgsz1024.log`](runs/imgsz1024.log)
  (gitignored); curve in `artifacts/adapt/yolo26p_svanstrom_all_f172c7c3/run/results.csv`.
- Validazione a 1024 (come il training) per il run a 1024, a 640 per il riferimento; stessa misura degli altri
  esperimenti (mAP a `conf` 0.001, drone con F1 alla `conf` del picco sul val).

| imgsz | val tutte mAP50 / mAP50-95 | val drone F1 / mAP50-95 |
|---|---|---|
| 640 (`577b097f`) | 0.851 / 0.462 | 0.868 / 0.386 |
| 1024 (`f172c7c3`) | 0.866 / 0.462 | 0.828 / 0.396 |

## Esito

A 1024 il mAP50-95 sul val e' identico (0.462), il mAP50 e' piu' alto (+0.015), il drone ha F1 piu' basso (0.828 contro 0.868)
e mAP50-95 simile (0.396 contro 0.386); il costo e' circa 2.5x di tempo e di GPU. Non conviene.

**Stato**: chiuso.
