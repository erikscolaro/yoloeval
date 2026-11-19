# Maschere stabilizzate: lr a coseno e rete dell'ultima epoca

**Scopo**: capire se la ricerca da PN- lascia 3 ms di budget inutilizzati (27,1 ms sul Pi contro 30,3) e taglia
troppo all'inizio della rete perché le maschere non si fermano mai, e quindi la rete scelta è quasi un'estrazione a
sorte.

**Stato**: chiuso il 18/11/2026. La temperatura è il default della ricerca dal 19/11/2026.

## Ipotesi (dai dati della ricerca di PN-, `46e210de`)

- **DUCCIO non riporta su i canali.** La penalità è `forza x max(0, costo - target)`: sotto il target vale zero, e
  i canali tornano solo se la loss del task li chiede. PN- scende sotto il target già all'epoca 4 (0,69 contro
  0,716 del costo iniziale), poi il regolarizzatore tace.
- **Le maschere non si fermano.** Il lr delle maschere è fisso a 0,01 per 30 epoche (`nas.lrf=1.0`). All'epoca 30
  il 66% dei blocchi sta a meno di 0,15 dalla soglia (|alpha| = 0,5), e la latenza oscilla di circa 2 ms tra
  un'epoca e l'altra.
- **La scelta del best amplifica il rumore.** best.pt è l'epoca con la mAP più alta dentro il budget, ma durante la
  ricerca la mAP salta tra 0,13 e 0,39: PN- ha preso l'epoca 20 (27,7 ms stimati), il last era a 27,1.
- Nelle ricerche da N a 28 ms (`pendenze/livello1/results/maschere_best.jsonl`) il collo di bottiglia del layer 2
  scende a 2 canali all'epoca 11 e non risale più; i layer 0 e 1 invece oscillano.

## Setup

Uguale alla ricerca di PN- in `scripts/catena_pn.sh` (SGD lr 0,01, forza 2, stimatore `best`, 30 epoche, target
30,3 ms), con tre differenze:

- `strategy.search.nas.lrf=0.01` e `nas.cos_lr=true`: lr delle maschere da 0,01 a 0,0001 a coseno;
- `strategy.search.pit.select=last`: la rete è quella dell'ultima epoca (opzione nuova di yolopit, branch
  `esplora/maschere-last` al momento della prova, in main dal 19/11/2026, commit 5176375);
- `save_period=1` nella ricerca e nel training da zero: un checkpoint per epoca, con le maschere (`pit_state_dict`)
  nella ricerca.

Poi da zero 300 epoche (`finetune=scratch`) e misura sul Pi (OpenVINO fp16).

```bash
systemd-run --user --unit=maschere-lr --working-directory=$PWD experiments/pruning/maschere_lr/rifai.sh
tail -f experiments/pruning/maschere_lr/runs/diario.log
```

## Risultati

### Ricerca (17/11, `c0d7cb3f`, 47 min)

Rete finale (epoca 30): 27,74 ms stimati, praticamente uguale a PN- (27,72). Il lr a coseno non ferma la rete:

| epoca | lr maschere | costo (ms) | variazione media di \|alpha\| | blocchi che cambiano stato |
|---|---|---|---|---|
| 22 | 0,0021 | 29,65 | 0,042 | 265/1031 |
| 26 | 0,0008 | 29,87 | 0,012 | 250/1031 |
| 28 | 0,0003 | 26,02 | 0,005 | 242/1031 |
| 30 | 0,0001 | 27,74 | 0,001 | 177/1031 |

Le maschere si muovono sempre meno, ma a ogni epoca cambia ancora un quarto dei blocchi, perché sono **sulla
soglia**: all'epoca 30, 625 blocchi su 1031 (61%) hanno \|alpha\| a meno di 0,005 da 0,5. La soglia fa da
attrattore (lo straight-through spinge su un blocco spento e giù uno acceso), e la rete finale dipende da
differenze di millesimi: né il lr a coseno né la scelta fra best e last la rendono stabile.

### Temperatura delle maschere (18/11, `rifai_temp.sh`, ricerca `51f642f1`)

Come sopra più `pit.temperature=0.1` (yolopit `esplora/maschere-last`, in main da 5176375): nel forward di training la maschera di un
blocco è `sigmoid((|alpha| - 0,5)/T)`, T da 0,1 a 0,001, ultime 3 epoche nette; costo, validazione ed export netti.

| | senza temperatura (`c0d7cb3f`) | con temperatura (`51f642f1`) |
|---|---|---|
| latenza stimata finale | 27,74 ms | **30,26 ms** (target 30,3) |
| latenza dall'epoca 10 alla 30 | oscilla 26,0–30,2 ms | ferma a 30,26 ms |
| blocchi a meno di 0,005 dalla soglia (ep. 30) | 625/1031 | **0** (nessuno nemmeno entro 0,05) |
| blocchi che cambiano stato fra le epoche 29 e 30 | 177 | **0** |
| mAP50-95 durante la ricerca (ultime epoche) | 0,09–0,44 | 0,571–0,577 |
| canali potabili tenuti | 1234 | 1372 |
| layer 0 / 1 / collo di bottiglia del layer 2 / cv2 | 8 / 14 / 4→2→4 / 16 (PN-) | 10 / 20 / 10→6→10 / 22 |

La temperatura toglie l'accumulo sulla soglia: dall'epoca 10 l'architettura non cambia più, la ricerca usa tutto il
budget e i primi strati restano più larghi.

### Da zero e Pi

Da zero 300 epoche (seed 0), poi Pi 5 con OpenVINO fp16 (mediana su 200 iterazioni); mAP50-95 sul validation.

| rete | ricerca | latenza stimata | latenza Pi | mAP training (best) | mAP Pi | risultati |
|---|---|---|---|---|---|---|
| PN- (senza temperatura, ultima epoca) | `c0d7cb3f` | 27,74 ms | 27,23 ms | 0,559 | 0,554 | `865e78f4a08d` |
| riempita (strada 3) | `8114da10` | 30,30 ms | 30,12 ms | 0,566 | 0,568 | `d2095054d01f` |
| con temperatura (strada 1) | `51f642f1` | 30,26 ms | 30,40 ms | 0,567 | 0,568 | `940f2b6bf838` |

Le due strade arrivano alla stessa mAP (differenza nel rumore di un solo seed), circa 0,014 sopra PN- sul Pi con 3 ms
in più: il budget lasciato libero valeva mAP. La temperatura ci arriva da sola, senza il riempimento a posteriori, ed
è diventata il default della ricerca (`conf/strategy/pit_latency.yaml`: `pit.temperature: 0.1`, `pit.select: last`).

Con lo yaml di oggi, che ha già `select` e `temperature`, gli script passano `pit.select` senza `+` e tolgono la
temperatura con `~strategy.search.pit.temperature` dove la prova non l'aveva. Le chiavi tornano quelle delle ricerche
originali (`c0d7cb3f`, `8114da10`, `51f642f1`), quindi rilanciandoli si riusano gli artefatti.
