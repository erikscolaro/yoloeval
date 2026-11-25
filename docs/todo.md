# Cose da fare

Stato al 25/11/2026 (le parti di potatura ed esame finale sono ferme al 19/11). Il piano di riferimento è `docs/NEW_MASTERPLAN.md`; i numeri vengono da `lab/reports/latest`
(`python -m tools.report`).

## Misure e confronti (blocco 2, fase 4b)

- [ ] **Potature con la temperatura dalle altre partenze.** La ricerca con `pit.temperature=0.1` c'è solo da PN-;
  da rifare da N-, PN, PN+ e P+ (ricerca, da zero 300 epoche, Pi). Circa 13-14 ore con 3 in parallelo, 16-18 con 4.
- [ ] **Più seed per la conclusione sulla potatura.** Oggi tutto è a seed 0. La migliore potata (PN- con
  temperatura, `d0105599`: 30,4 ms, mAP50-95 0,567) sta 0,006 sotto P (27,7 ms, 0,573); sulla mAP50 il distacco
  scende a 0,003-0,005, vicino al rumore fra seed (0,004 su P). Con la mAP misurata sul Pi (19/11) il distacco
  sparisce: 0,5683 contro 0,5687 di P. Servono 3 seed per la potata migliore, per P e per
  P- (masterplan 4b, "i confronti (3 seed)").
- [ ] **Forma o pesi ereditati** (domanda 2 del masterplan): la stessa architettura potata anche con il finetune
  dai pesi del modello grande, da confrontare con il training da zero (oggi il default).
- [ ] **Calcoli contro stimatore** (S4-01): potatura guidata dalle operazioni, alla stessa latenza. Rimandata.
- [ ] **Budget più stretto `B'`** (S4-03, scelta 4 del masterplan): da decidere se farlo.
- [ ] **Target alla pari con P-** (circa 26 ms invece di 30,3): da decidere.
- [ ] **Baseline di architetture casuali** alla stessa latenza, da zero 300 epoche, per il confronto delle
  pendenze. Rimandata.

## Tracker (Kalman e crop, `docs/tracker.md`)

Fatto al 24/11/2026: Kalman e tracker con crop in `src/`, tarati con Optuna sul val; risultati in
`experiments/tracking/` (`taratura_kalman`, `taratura_kalman_crop`, `confronto_p_s_crop`). Crop 256 con S-: F2 sui
droni 0,718 sul totale (Svanström 0,461, Anti-UAV 0,837), contro 0,666 di P- da solo.

- [ ] Latenza di S- sul Pi a crop 128 e 64, batch 1, 2, 4: oggi c'è solo il 128 a batch 1 (14/10, testa vecchia).
- [ ] R di S- e Optuna a crop 128 (servono `--crop` in `rs.py` e `optuna_crop.py`): con i parametri del 256 il
  128 perde solo 0,007 di F2.
- [ ] Lato del crop scelto dalla grandezza del box: a 64 Svanström sale (0,476), Anti-UAV scende (0,786). Da provare.
- [ ] Da verificare: sui fotogrammi con 2+ oggetti l'F2 del crop è 0,654 a 256, 128 e 64.
- [ ] Valutazione sul test, una volta sola, con i parametri scelti sul val.

## Esame finale (fase 5, alla fine)

- [ ] Soglia di confidenza di deploy sul val (massimo F1) per i modelli scelti.
- [ ] E5-1 ed E5-2: precisione sul test, per classe e per grandezza del box, una volta sola.
- [ ] B5-1 e B5-3: latenza finale e robustezza alla frequenza sostenibile del Pi.

## Decisioni in sospeso

- [ ] Rendere potabili i canali dentro C2PSA, PSABlock e Detect in yolopit: solo proposta, non autorizzata.

## Documentazione

- [ ] Figure di sintesi B e C: collegarle alla mAP vera (modello di precisione).

## Aperti minori

- [ ] Le misure int8 con onnxruntime su N- fanno riavviare il Pi dopo circa un minuto (due volte su due). Sospetto
  un watchdog, non verificato. Per ora lasciate fallite.
- [ ] La ricetta int8 del Pi scritta nel masterplan (X2: onnxruntime con testa, ramo di classe e attention in float,
  29,4 ms) non c'è in `conf/quantization/`: `int8` quantizza solo Conv, Gemm e MatMul (P- 57,7 ms, il doppio di
  `int8_full`, come in AOD4), `int8_full` tutto tranne la coda della testa (26,5 ms, mAP50-95 0,560). Da aggiungere
  se l'int8 sul Pi serve ancora.

## Seed e metrica di scelta del checkpoint (promemoria del 2/11/2026)

### Fitness: mAP50-95 o mAP50

- Ultralytics 8.4 sceglie `best.pt` (e conta la patience) solo sul **mAP50-95**: `DetMetrics.fitness` ha pesi
  `[P, R, mAP50, mAP50-95] = [0, 0, 0, 1]` (`ultralytics/utils/metrics.py`). Non è più 0,1·mAP50 + 0,9·mAP50-95.
- Per questo dominio (bersagli di 20–40 px) la metrica principale più sensata è il **mAP50**: su un box di 24 px
  bastano 2 px di errore per scendere a IoU 0,85, quindi le soglie alte misurano soprattutto il rumore delle etichette.
- Sulla griglia larghezza × profondità (seed 0, 23 celle finite al 4/11) scegliere sul mAP50-95 costa in mediana
  0,0014 di mAP50 rispetto al miglior mAP50 visto nel training, al massimo 0,011 (0.375 D170); solo 3 celle superano
  0,005.

### Da decidere prima di lanciare i seed 1 e 2

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
