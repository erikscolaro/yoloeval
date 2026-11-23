# Latenza a batch 1, 2, 4, 8 a 256×256 sul Raspberry Pi 5

Quanto costa sul Pi un'inferenza con N crop 256×256 insieme, come nel fotogramma TRACK del crop tracker
(`src/crop_tracker.py`), per i 27 modelli della griglia (larghezze F…M × D101/D126/D170). Pesi casuali, OpenVINO fp16,
4 core, governor performance, 3 passate.

```bash
experiments/latenza/track_modelli_batch_crop256/rifai.sh        # train (pesi casuali), export, benchmark con il lock del Pi, figure
.venv/bin/python experiments/latenza/track_modelli_batch_crop256/figure.py   # solo le figure
```

Il budget del modello a 30 fps dipende da N: `33,3 ms − costo fuori dal modello`, misurato in `../track_budget_n_crop`
(32,3 / 31,4 / 29,6 / 25,9 ms per N = 1 / 2 / 4 / 8). Le figure lo disegnano con una fascia di ±5%.

## Figure (`report/`)

| file | cosa mostra |
|---|---|
| `A_batch_serie` | un pannello per profondità; x = batch, y = latenza dell'inferenza (0–40 ms), una serie per larghezza |
| `A_crop_serie` | lo stesso con la latenza per immagine (latenza / N); il budget diventa budget(N) / N |
| `A_pareto` | un pannello per batch (2×2), legenda unica; mAP50-95 contro latenza, solo i modelli entro il budget +5%, fronte di Pareto |
| `latenza_batch.csv` | i numeri: mediana, ms per crop, budget |

Le mAP50-95 del Pareto sono quelle del val a 640 dei training da 300 epoche (`artifacts/weights`, media fra i semi),
non sui crop. M non ha la mAP: non è addestrata, costa troppo per il deploy.

## Didascalie (da usare nella tesi)

- **Serie**: in legenda solo le larghezze visibili. Le larghezze più grandi escono dal grafico sopra i 40 ms, perché
  sono molto sopra il budget a ogni batch.
- **Pareto**: nel grafico ci sono solo i modelli entro il budget del modello +5% a quel batch. Gli altri modelli sono
  esclusi perché non stanno nel budget di tempo. M non c'è perché non è addestrata.

## Risultati (20/11/2026)

- La latenza è quasi affine nel batch: `latenza(N) ≈ a + b·N`, errore massimo sotto l'8%. Il costo fisso `a` è
  1,5–3 ms (cresce con la profondità), il costo per immagine `b` va da 2,4 ms (F-) a 71 ms (M+).
- Per immagine il batch aiuta solo i modelli piccoli: per F–P il costo fisso è il 30–50% di un'inferenza, e con 8 crop
  il costo per immagine scende del 30–40%. Da NS in su il costo fisso è trascurabile e il batch non aiuta.
- Due riavvii del Pi durante la misura, tutti e due con batch 8 (SM- e SM+, non con M che è più pesante). Le celle
  sono state rimisurate e sono buone. La causa non è nota.
- `benchmark_app` non riporta i percentili: nei record c'è solo la mediana (record "non conformi" per `p99_ms`).
