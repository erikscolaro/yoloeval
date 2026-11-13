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
| [`pruning/lut_pi_passo2`](pruning/lut_pi_passo2) | LUT 2D di latenza a passo 2 delle 13 famiglie di conv potabili del nano sul Pi 5 (OpenVINO fp16, 4 core), 71 328 punti × 3 | **procedura**, chiusa il 09/11: LUT completa (220 722 misure in 74 h, nessun throttling); I/O fp16 NHWC, warmup a tempo di 60 ms, niente reti identità (costo fisso in C); SiLU non fusa nella conv, come nel nano vero. Studi chiusi dentro: warmup, transitorio, due regimi, fusione |
| [`pruning/b4r_composizione`](pruning/b4r_composizione) | composizione LUT → latenza del nano potato sul Pi 5: C per architettura e a_f su 765 reti (15 architetture fino al nano) con i vincoli di yolopit, misurate intere | **procedura**, chiusa il 09/11: reti misurate, fit fatto; architetture tenute fuori: errore p95 3,0% (somma LUT 10,2%, MAC 26,9%) |

Le due **procedure** (LUT e B4-R) non sono più codice dell'esperimento: stanno in `src/lut` e
`scripts/build_lut.sh <out_dir>` le rifà da capo (default `artifacts/lut/rpi5/`: `lut_<famiglia>.csv` e
`fit.json`, i file di `pit.latency`; `scripts/sync_yolopit_lut.sh` li confronta con `yolopit/data/rpi5`). Ogni
esperimento ha un `rifai.sh` che le richiama con i parametri di allora e scrive nella sua `results/` (o in una
cartella indicata). Gli **studi chiusi** (separability, warmup, transient, bimodal, check_fusion) restano script
dell'esperimento.
