# yolo26p con uscita P2

`yolo26p` (testa P3-P5) contro `yolo26p_p2` (testa P2-P5, `models/yolo26p_p2.yaml`): l'uscita a stride 4 serve
agli oggetti piccoli. Costo: 686,768 parametri (+1.3%) ma 2.5 GFLOPs contro 1.7 (+47%).

- Training su AOD4 con gli stessi parametri del pico: `run.py stage=train model=yolo26p_p2 train=scratch_musgd`
  (MuSGD lr0 0.01, 250 epoche, batch 64, mosaic 1.0, `close_mosaic` 10). Log: [`runs/train.log`](runs/train.log)
  (gitignored).
- Poi adapt su Svanström con la ricetta migliore (`conf/adapt/default.yaml`) e confronto, con lo stesso
  protocollo (mAP a `conf` 0.001, `conf` per classe al picco F1 scelta sul val), con il pico migliore e il nano.

## Esito

Da compilare.

**Stato**: training in corso.
