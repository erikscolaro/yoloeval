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
| [`pruning/lut_pi_passo2`](pruning/lut_pi_passo2) | LUT 2D di latenza a passo 2 delle 13 famiglie di conv potabili del nano sul Pi 5 (OpenVINO fp16, 4 core), 69 989 punti × 3 | in corso dal 06/11 11:10 con la ventola al massimo (stima 39 h, fine ~08/11 02:00): I/O fp16 NHWC, warmup a tempo di 60 ms, niente reti identità (costo fisso in C); SiLU non fusa nella conv, come nel nano vero |
| [`pruning/b4r_composizione`](pruning/b4r_composizione) | composizione LUT → latenza del nano potato sul Pi 5: C e a_f su 150 nani potati a caso con i vincoli di yolopit, misurati interi | script pronti il 06/11; reti generate, misura dopo la LUT |
