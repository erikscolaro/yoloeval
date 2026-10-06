# Ablation: lr0 dell'adattamento su Svanström

Adattamento (`stage=adapt`) dei pesi AOD4 di yolo26p (`train=scratch_musgd`) su Svanström, rete intera, MuSGD.
Quanto pesa `lr0` sul sovrallenamento? Run lanciati con
`run.py stage=adapt model=yolo26p train=scratch_musgd adapt.lr0=<valore> ...`; curve in
`artifacts/adapt/<slug>/run/results.csv`, log in `runs/`.

Misura omogenea di tutti i run (script fuori repo): `best.pt` sul val e sul test di Svanström, mAP a `conf` 0.001;
per il drone P/R/F1 alla `conf` che massimizza l'F1 sul val (stessa soglia sul test).

| lr0 | altro | epoche (patience) | slug | val tutte mAP50 / mAP50-95 | val drone F1 / mAP50-95 | test tutte mAP50 / mAP50-95 | test drone F1 / mAP50-95 |
|---|---|---|---|---|---|---|---|
| (AOD4, senza adapt) | | | `e5930f9c` | 0.651 / 0.360 | 0.783 / 0.324 | 0.573 / 0.312 | 0.702 / 0.227 |
| 0.005 | mosaic 1, close_mosaic 25 (vecchio default) | 100 (100) | `0f26a427` | 0.838 / 0.386 | 0.816 / 0.346 | 0.849 / 0.431 | 0.809 / 0.369 |
| 0.005 | `scale` 0.562, mosaic 0 | 100 (100) | `729c74b4` | 0.820 / 0.370 | 0.821 / 0.347 | 0.788 / 0.366 | 0.747 / 0.327 |
| 0.005 | `scale` 0, mosaic 0 | 17 su 100 (15) | `72cb37e5` | 0.830 / 0.390 | 0.856 / 0.391 | 0.779 / 0.381 | 0.762 / 0.314 |
| 0.001 | mosaic 0 | 17 su 100 (15) | `6ddfcba2` | 0.847 / 0.442 | 0.883 / 0.397 | 0.883 / 0.474 | 0.805 / 0.353 |
| 0.0002 | warmup 2, mosaic 0 | 17 su 40 (15) | `577b097f` | 0.851 / 0.462 | 0.868 / 0.386 | 0.853 / 0.471 | 0.814 / 0.335 |
| **0.0001** | warmup 2, **mosaic 1**, `scale` 0 | 21 su 40 (15) | `c3b39818` | **0.861 / 0.470** | 0.861 / 0.390 | 0.825 / 0.465 | 0.772 / 0.326 |

In train (senza `conf`, `results.csv`) tutti i run con mosaic 0 hanno il best all'epoca 2-3 (mAP50-95 0.390 / 0.443 /
0.462 con lr0 0.005 / 0.001 / 0.0002); il run con mosaic arriva a 0.470 all'epoca 6 e poi resta piatto (0.464-0.470).

## Note

- Con lr0 alto il best arriva nelle prime epoche e la val loss poi sale (box 1.90 → 2.14 a 100 epoche): sovrallenamento,
  non un problema di `scale` (il picco c'e' anche con `scale: 0`). Abbassare l'lr alza il picco (+0.05 e +0.02 di
  mAP50-95 train) ma non lo sposta: la causa non e' solo l'lr.
- Sul val vince `c3b39818` (0.470), con 0.007 di margine sul secondo (0.463, vedi `../optimizer`); sul test i run a
  lr 0.001 e 0.0002 sono avanti (0.471-0.474 contro 0.465), differenze nel rumore di un solo seed. Ricetta adottata
  in `conf/adapt/default.yaml`: lr0 1e-4, mosaic 1, warmup 2, 40 epoche.
- Sul drone il migliore per val e' lr 0.001 (F1 0.883, mAP50-95 0.397), ma a pari merito sul complessivo conta poco.
- AOD4 crolla dopo l'adapt (bird quasi azzerata, mAP50 0.002): oblio/cambio di dominio, la dimensione degli oggetti da
  sola non lo spiega (il drone ha scala simile).
- `0f26a427` e' il vecchio default (mosaic 1, `scale` 0.5): il suo test e' alto ma e' un singolo run con un altro setup.
- `lr0` effettivo: con MuSGD i gruppi di parametri hanno moltiplicatori (`lr/pg0` nel csv ~3 volte `lr0`), non verificato.

**Stato**: chiuso.
