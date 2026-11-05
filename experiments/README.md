# Esperimenti

Studi una tantum fuori dalla pipeline, raggruppati per stadio (`dataset`, `train`, `adapt`, `pruning`,
`finetune`, `quantization`, `deploy`, `benchmark`). Tenere aggiornato questo indice quando se ne
aggiunge o chiude uno. Gli esperimenti di prima del 12/10 sono in `.old/experiments/`.

Un esperimento corrisponde a una cartella con tutto quello che serve per rifarlo:

- `README.md`: scopo, setup, comandi, risultati e commento ai grafici;
- `run.py`: punto d'ingresso; gli script secondari hanno un nome descrittivo;
- `results/` e `figures/`: csv/json e grafici;
- `runs/`: output grezzi (log, modelli intermedi), ignorata da git (`experiments/**/runs/`).

Il codice comune sta in `_lib.py`: `ROOT` (root del repo, aggiunta a `sys.path`), più gli helper
della pipeline.

## pruning

| cartella | scopo | stato |
|---|---|---|
| [`pruning/separability`](pruning/separability) | latenza di una conv sul Pi 5 (OpenVINO fp16, 4 core): t(c_in, c_out) ≈ f(c_out)·g(c_in)? | chiuso il 05/11: non separabile (A conv 1×1: 71% dei punti entro 2× il rumore; B conv 3×3: 34%); salti a numeri fissi di canali, non spiegati da riscaldamento né dimensioni dei tensori |
| [`pruning/lut_pi_passo2`](pruning/lut_pi_passo2) | LUT 2D di latenza a passo 2 delle 13 famiglie di conv potabili del nano sul Pi 5 (OpenVINO fp16, 4 core), 69 989 punti × 3 | in corso dal 05/11 (stima 40–100 h); SiLU non fusa nella conv, come nel nano vero |
