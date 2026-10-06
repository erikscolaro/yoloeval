# Ablation: optimizer dell'adattamento su Svanström

Adattamento di yolo26p (`train=scratch_musgd`, pesi AOD4) su Svanström con tre optimizer, a parita' di resto
(40 epoche, patience 15, warmup 2, `scale: 0`, `mosaic: 0`, batch 64). Due serie: rete intera e backbone
congelato (`freeze=11`: layer 0-10, fino a C2PSA).

lr0: MuSGD e SGD 2e-4 (migliore dell'ablation su lr), Adam 2e-5 (10x piu' basso come nelle conf di finetune: scelta mia,
non ottimizzata).

- Script: `run.py` lancia i 6 adapt in sequenza (salta quelli in cache), `run.py --report` stampa la tabella dalle curve.
- Il run MuSGD 2e-4 a rete intera e' `577b097f` di `../lr`.
- Misura omogenea: `best.pt` sul val e sul test di Svanström, mAP a `conf` 0.001; drone con F1 alla `conf` del picco sul val.

| optimizer | lr0 | freeze | slug | val tutte mAP50 / mAP50-95 | val drone F1 / mAP50-95 | test tutte mAP50 / mAP50-95 | test drone F1 / mAP50-95 |
|---|---|---|---|---|---|---|---|
| MuSGD | 2e-4 | no | `577b097f` | 0.851 / 0.462 | 0.868 / 0.386 | 0.853 / 0.471 | 0.814 / 0.335 |
| SGD | 2e-4 | no | `cd3581bd` | 0.846 / 0.461 | 0.873 / 0.395 | 0.841 / 0.468 | 0.789 / 0.348 |
| Adam | 2e-5 | no | `bdff53f2` | 0.855 / 0.436 | 0.882 / 0.383 | 0.844 / 0.425 | 0.812 / 0.348 |
| MuSGD | 2e-4 | 11 | `21f04a25` | 0.847 / 0.461 | 0.854 / 0.392 | 0.794 / 0.441 | 0.717 / 0.295 |
| SGD | 2e-4 | 11 | `22e2e1fa` | 0.849 / 0.463 | 0.863 / 0.383 | 0.808 / 0.449 | 0.753 / 0.305 |
| Adam | 2e-5 | 11 | `8d4640de` | 0.845 / 0.458 | 0.873 / 0.377 | 0.820 / 0.446 | 0.770 / 0.314 |

## Esito

- L'optimizer non cambia la forma del problema: il picco arriva entro l'epoca 4 con tutti e tre; i valori stanno in
  0.01-0.03 di mAP50-95, dentro il rumore di un seed. MuSGD e SGD pari (0.462 / 0.461), Adam piu' basso in mAP50-95
  (0.436) ma con mAP50 e F1 del drone simili.
- Il backbone congelato non migliora sul val (0.458-0.463) e peggiora sul test (0.441-0.449 contro 0.468-0.471 della
  rete intera): non conviene.
- Non e' MuSGD a dare problemi: restano la scelta di `conf/adapt/default.yaml`.

**Stato**: chiuso.
