# Cose da fare

Stato al 19/11/2026 (pomeriggio). Il piano di riferimento è `NEW_MASTERPLAN.md`; i numeri vengono da `lab/reports/latest`
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

## Esame finale (fase 5, alla fine)

- [ ] Soglia di confidenza di deploy sul val (massimo F1) per i modelli scelti.
- [ ] E5-1 ed E5-2: precisione sul test, per classe e per grandezza del box, una volta sola.
- [ ] B5-1 e B5-3: latenza finale e robustezza alla frequenza sostenibile del Pi.

## Decisioni in sospeso

- [ ] **Tracker con predittore di Kalman** e crop guidato: piano in `docs/tracker.md`, non avviato. Partenza proposta:
  livello B, 1–2 giorni.
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
