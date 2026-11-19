# Cose da fare

Stato al 19/11/2026 (pomeriggio). Il piano di riferimento è `NEW_MASTERPLAN.md`; i numeri vengono da `lab/reports/latest`
(`python -m tools.report`).

## Misure e confronti (blocco 2, fase 4b)

- [ ] **Potature con la temperatura dalle altre partenze.** La ricerca con `pit.temperature=0.1` c'è solo da PN-;
  da rifare da N-, PN, PN+ e P+ (ricerca, da zero 300 epoche, Pi). Circa 13-14 ore con 3 in parallelo, 16-18 con 4.
- [ ] **Più seed per la conclusione sulla potatura.** Oggi tutto è a seed 0. La migliore potata (PN- con
  temperatura, `d0105599`: 30,4 ms, mAP50-95 0,567) sta 0,006 sotto P (27,7 ms, 0,573); sulla mAP50 il distacco
  scende a 0,003-0,005, vicino al rumore fra seed (0,004 su P). Servono 3 seed per la potata migliore, per P e per
  P- (masterplan 4b, "i confronti (3 seed)").
- [ ] **mAP sul Pi anche per la griglia.** Il report confronta le mAP del training perché quella misurata sul Pi
  c'è solo per P-, N- e le potate. Per un confronto alla pari serve almeno per P, FP+ e le reti vicine al budget.
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

- [ ] Rendere potabili i canali dentro C2PSA, PSABlock e Detect in yolopit: solo proposta, non autorizzata.

## Documentazione

- [ ] Masterplan da aggiornare: i set di deploy a 480p, 720p e 1080p (dal 13/11 il deploy è a 640); in 4b la
  partenza dal nano con il finetune (oggi più partenze, training da zero); la temperatura nella ricetta della ricerca.
- [ ] README di `experiments/pruning/pendenze/livello1`.
- [ ] Figure di sintesi B e C: collegarle alla mAP vera (modello di precisione).

## Aperti minori

- [ ] Le misure int8 con onnxruntime su N- fanno riavviare il Pi dopo circa un minuto (due volte su due). Sospetto
  un watchdog, non verificato. Per ora lasciate fallite.
- [ ] P- int8 onnxruntime con le parti sensibili in float: 57,7 ms contro 29,4 della prova vecchia. Non indagato.
