# Livello 1: ricerca corta con la loss di detection

Secondo filtro del piano (`../PIANO.md`), dopo il livello 0 (solo costo, `../livello0`). Qui la ricerca PIT ha la loss
di detection completa, e cambia solo la pendenza dello stimatore di latenza (`strategy.search.pit.latency.estimator`).
Scopo: scegliere la pendenza da usare nella ricerca finale.

**Stato**: chiuso il 14/11/2026. La pendenza scelta è `best`, il default di `conf/strategy/pit_latency.yaml`.

## Setup

- **Rete e target**: nano D126 (`yolo26n`), target 28 ms (`target.model_budget_ms=28`), profilo `rpi5_640`.
- **Ricerca**: 30 epoche, batch 64, maschere con SGD a lr 0,01 e forza di DUCCIO 2. Seed della ricerca 42 e 43.
- **Pendenze**: `lut`, `smooth`, `best`. `secant` non è fra le pendenze di `rifai.sh`.
- **Dopo la ricerca** (`dopo.sh`): finetuning corto (10 epoche, `ft10`), finetuning intero (100 epoche, `ft100`) e
  training da zero (300 epoche, `scratch`), poi export OpenVINO fp16 e misura sul Pi.
- **Regola di scelta** (`scegli.py`): si scelgono sul training da zero, perché misura l'architettura senza i pesi
  usciti dalla ricerca.
  - Fuori la pendenza che sul Pi supera 28,85 ms: target più il margine di yolopit, il 5% dei 17 ms della parte PIT.
  - Vince la pendenza con la latenza media più vicina al target.
  - Se due pendenze stanno entro 0,5 ms, vince la mAP50-95 più alta.

Le ricerche del 13/11 con la forza di default non arrivavano a 28 ms (`results/forza1/LEGGIMI.txt`). Da qui nasce il
moltiplicatore della forza (`regularizer.strength`).

```bash
OPTS=sgd LR=0.01 FORZA=2 STIME="smooth best" experiments/pruning/pendenze/livello1/rifai.sh        # ricerca, seed 42
OPTS=sgd LR=0.01 FORZA=2 SEED=43 STIME="smooth best" experiments/pruning/pendenze/livello1/rifai.sh
TAG=scratch FT="finetune=scratch" experiments/pruning/pendenze/livello1/dopo.sh smooth best         # da zero e Pi
TAG=scratch .venv/bin/python experiments/pruning/pendenze/livello1/scegli.py                       # classifica
```

I passi dal secondo seed alla ricerca finale stanno in `scripts/catena_notte.sh`. `traccia_maschere.py` scrive i
canali tenuti a ogni epoca in `results/maschere_<pendenza>.jsonl`.

## Risultati

### Ricerca (seed 42, latenza stimata a fine epoca, `results/maschere_*.jsonl`)

| pendenza | primo arrivo sotto 28,85 ms | epoche 27–30 (ms) |
|---|---|---|
| `lut` | mai | 32,7 / 32,1 / 32,2 / 31,9 |
| `smooth` | epoca 28 | 28,7 / 27,8 / 27,2 (dall'epoca 28) |
| `best` | epoca 15 | 27,4 / 28,7 / 27,7 / 26,9 |

`lut` si ferma a 31,9 ms e manca il target, quindi è esclusa dai passi dopo. Con `best` la rete arriva al target a
metà ricerca, ma poi continua a oscillare di circa 2 ms fra un'epoca e l'altra. È l'oscillazione delle maschere sulla
soglia studiata poi in `../../maschere_lr`.

### Dopo la ricerca (mAP50-95 dell'epoca migliore del training, latenza mediana sul Pi)

| pendenza | seed | ft10 | ft100 | da zero | Pi (da zero) |
|---|---|---|---|---|---|
| `best` | 42 | 0,464 | 0,522 | 0,536 | 27,91 ms |
| `best` | 43 | – | – | 0,535 | 27,93 ms |
| `smooth` | 42 | 0,459 | 0,523 | 0,525 | 26,74 ms |
| `smooth` | 43 | – | – | 0,530 | 26,89 ms |

Tutte e quattro le reti da zero stanno sotto 28,85 ms. `best` sta più vicino al target: 27,9 ms di media contro
26,8, cioè più di 0,5 ms di differenza, quindi vince `best` già con la seconda regola. Ha anche la mAP più alta
(0,535 contro 0,528 di media). La differenza di mAP è piccola, ma i due seed danno lo stesso ordine. Al finetuning
le due pendenze sono pari.
