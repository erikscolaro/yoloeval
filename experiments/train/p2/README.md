# yolo26p con uscita P2

`yolo26p` (testa P3-P5) contro `yolo26p_p2` (testa P2-P5, `models/yolo26p_p2.yaml`): l'uscita a stride 4 serve
agli oggetti piccoli. Costo: 686,768 parametri (+1.3%) ma 2.5 GFLOPs contro 1.7 (+47%).

- Training su AOD4 con gli stessi parametri del pico: `run.py stage=train model=yolo26p_p2 train=scratch_musgd`
  (MuSGD lr0 0.01, 250 epoche, batch 64, mosaic 1.0, `close_mosaic` 10). Log: [`runs/train.log`](runs/train.log)
  (gitignored).
- Poi adapt su Svanström con la ricetta migliore (`conf/adapt/default.yaml`) e confronto, con lo stesso
  protocollo (mAP a `conf` 0.001, `conf` per classe al picco F1 scelta sul val), con il pico migliore e il nano.

## Esito (training su AOD4, val AOD4)

| modello | parametri | GFLOPs | mAP50 / mAP50-95 (250 epoche) |
|---|---|---|---|
| yolo26p (`e5930f9c`) | 678,176 | 1.7 | 0.960 / 0.649 |
| **yolo26p con P2** (`35b57861`, 173 min) | 686,768 | 2.5 | 0.964 / 0.654 |
| yolo26n (baseline tesi, dai pesi COCO) | 2,505,360 | 5.9 | 0.969 / 0.684 |

A parita' di epoca il P2 e' avanti al p di circa 0.004 di mAP50-95 dall'epoca 50 (dentro il rumore di un seed) e
resta indietro al nano di circa 0.03; su AOD4 gli oggetti sono grandi (mediana ~64 px a 640), quindi il P2 ha poco da
dare: il confronto che conta e' su Svanström dopo l'adapt.

Attivazioni (un'immagine 640x640, uscita di ogni layer): somma 9.33 M valori contro 5.65 M del p (35.6 contro 21.5 MiB
fp32), picco con i tensori vivi 3.64 M contro 1.79 M (13.9 contro 6.8 MiB fp32, 3.5 contro 1.7 MiB int8);
il tensore piu' grande e' la concat P2 (64x160x160 = 1.64 M valori); la Detect vede 34000 candidati contro 8400.

**Stato**: training su AOD4 concluso; adapt su Svanström, latenza sul Pi e confronto in corso.
