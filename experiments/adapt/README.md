# adapt

Esperimenti sul finetune di adattamento su Svanström (stadio adapt, dopo il training su AOD4).
Vuoto per ora.

Promemoria: con 100 epoche l'adattamento di yolo26p sovrallena (val loss in salita dall'epoca ~10); il best (mAP50-95 0.370, mAP50 0.820) è all'epoca 7, a fine run 0.329. Per questo `conf/adapt/default.yaml` usa `epochs: 50`, `patience: 15`.
