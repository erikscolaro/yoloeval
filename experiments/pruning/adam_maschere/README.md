# Maschere con Adam invece di SGD

**Scopo**: capire se Adam con lr 5e-4 sulle maschere di PIT, come nel notebook di PLiNIO, porta la ricerca da N-
a un'architettura migliore di quella che si ottiene con SGD a lr 0,01. Si prova anche una ricerca da 100 epoche
invece di 30.

**Stato**: chiuso il 19/11/2026. Le maschere restano su SGD.

## Setup

È la ricerca finale da N- (w0.25, D101, target 30,3 ms, stimatore `best`, forza 2, testa chiusa), con
`strategy.search.nas.optimizer=Adam` e `nas.lr0=0.0005`. Poi da zero 300 epoche (seed 0) e Pi 5 con OpenVINO fp16.
I piloti sono del 16/11, quindi senza temperatura e con `select` di allora.
I mAP50-95 di training sono in `artifacts/finetune/index.json` (ID `663f95e5`, `b4110602`, `8453b834`).

```bash
scripts/pilota_adam.sh          # ricerca, 30 epoche
scripts/pilota_adam100.sh       # ricerca, 100 epoche
scripts/pilota_adam_scratch.sh  # da zero 300 epoche e Pi, tutte e due
```

## Risultati

| maschere | epoche di ricerca | rete | latenza Pi | mAP50-95 (training) |
|---|---|---|---|---|
| SGD 0,01 | 30 | `663f95e5` | 29,15 ms | 0,555 |
| Adam 5e-4 | 30 | `b4110602` | 30,48 ms | 0,553 |
| Adam 5e-4 | 100 | `8453b834` | 29,52 ms | 0,554 |

Nessuno dei due piloti supera SGD di almeno 0,005 di mAP50-95, la soglia di rumore che usiamo. Nemmeno la ricerca
più lunga ci arriva. Con un solo seed le tre reti risultano uguali.
