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

## Dopo l'adapt: il P2 non serve su Svanström

Adapt su Svanström (ricetta di `conf/adapt/default.yaml`, misura omogenea, vedi `../../adapt/model_size`): P2 contro p
val 0.874 / 0.476 contro 0.861 / 0.470 (+0.006), test 0.810 / 0.456 contro 0.825 / 0.465 (-0.009), drone mAP50-95
0.392 / 0.330 contro 0.390 / 0.326. Costo sul Pi 5 (OpenVINO fp32, 4 core): 43.14 ms contro 27.95 (+54%).

- Nessun guadagno dimostrato: differenze dentro il rumore di un seed, su tutte le classi e sul drone, a +47% di FLOPs e +54% di latenza.
- Probabile motivo (ipotesi, non provata con un test diretto): in Svanström nessun oggetto scende sotto circa 13 px (lato maggiore
  minimo 13.0, 0.27% sotto 16 px, 0% sotto 8 px, mediana 35 px; `scale: 0` nell'adapt), cioe' sopra il limite pratico dello stride 8 della
  testa P3, e il livello P2 (stride 4) serve per oggetti sotto circa 8-16 px.
- In AOD4 il training vede oggetti molto piu' piccoli: dopo il resize 1024 → 640 (×0.625) e con `scale` 0.5 (×0.5 al minimo) il lato
  maggiore minimo e' 1.6 px, 0.1° percentile 4.1 px, 1° 7.0 px, e nel caso peggiore il 15.7% delle box sta sotto 16 px. Eppure su
  AOD4 il P2 e' solo +0.005 di mAP50-95 sul p (dentro il rumore).
- Conclusione: P2 non adottato per Svanström; avrebbe senso solo con oggetti sotto 16 px in modo sostanziale (o a risoluzione piu' bassa).

**Stato**: chiuso (training su AOD4, adapt su Svanström e latenza sul Pi misurati).
