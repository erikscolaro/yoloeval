# Scoperte

Una riga (max due) per scoperta, in elenco puntato; i dettagli stanno nei README degli esperimenti. Prima di aggiungere una voce
si propone il testo e si chiede il permesso.

- L'adapt su Svanström porta il pico da 0.360 a 0.470 di mAP50-95 sul val (+0.11) e il picco in training arriva sempre nelle prime epoche (2-9).
- Con lr più basso nell'adapt il picco sale (mAP50-95 0.390 → 0.443 → 0.462 con lr0 0.005 → 0.001 → 0.0002) ma non si sposta: il sovrallenamento resta.
- Optimizer (MuSGD, SGD, Adam) e backbone congelato non cambiano l'adapt (val 0.436-0.463) e `imgsz` 1024 non aiuta: non sono la leva.
- Confrontare modelli con un `conf` fisso falsa il mAP: servono mAP a `conf` 0.001 e F1 alla `conf` ottimale per classe.
- Il P2 non migliora il pico su Svanström (val +0.006) e costa +54% di latenza sul Pi: gli oggetti (min 13 px) non scendono sotto lo stride 8.
- Il nano adattato batte il pico di +0.047 sul val (0.517 contro 0.470) ma parte da pesi COCO/Objects365; s ed m aggiungono poco (0.530, 0.538).
- La distillazione dal nano aiuta solo con peso basso: `dis` 6 peggiora (val -0.012), `dis` 1 dà +0.011 sul val e +0.019 sul drone.
- Il teacher non allunga il training distillato (circa 35 s per epoca): il training è limitato dai dati, non dalla GPU.
