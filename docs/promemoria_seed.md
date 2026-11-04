# Promemoria per i prossimi seed (2/11/2026)

## Fitness: mAP50-95 o mAP50

- Ultralytics 8.4 sceglie `best.pt` (e conta la patience) solo sul **mAP50-95**: `DetMetrics.fitness` ha pesi
  `[P, R, mAP50, mAP50-95] = [0, 0, 0, 1]` (`ultralytics/utils/metrics.py`). Non è più 0,1·mAP50 + 0,9·mAP50-95.
- Per questo dominio (bersagli di 20–40 px) la metrica principale più sensata è il **mAP50**: su un box di 24 px
  bastano 2 px di errore per scendere a IoU 0,85, quindi le soglie alte misurano soprattutto il rumore delle etichette.
- Sulla griglia larghezza × profondità (seed 0, 23 celle finite al 4/11) scegliere sul mAP50-95 costa in mediana
  0,0014 di mAP50 rispetto al miglior mAP50 visto nel training, al massimo 0,011 (0.375 D170); solo 3 celle superano
  0,005.

## Da decidere prima di lanciare i seed 1 e 2

- I seed 1 e 2 devono usare la **stessa** regola di scelta del seed 0, altrimenti la media ± deviazione standard
  mescola due criteri.
- Il seed 0 della griglia è già scelto sul mAP50-95 e sul disco ci sono solo `best.pt` e `last.pt`: il checkpoint
  migliore in mAP50 del seed 0 non si può recuperare.
- Quindi, o si tiene il mAP50-95 per tutti i seed (scelta coerente con la griglia), o si passa al mAP50 e si rifà
  anche il seed 0 dei modelli che entrano nei confronti finali.
- Per passare al mAP50 basta ridefinire `DetMetrics.fitness` in `src/stages/train.py` prima di `model.train`, e
  la scelta va nel `training_key`, così i pesi vecchi non vengono presi dalla cache.
- In ogni caso nella tesi si riportano tutte e due le metriche; il numero onesto è quello sul test, non il massimo
  sul val.
