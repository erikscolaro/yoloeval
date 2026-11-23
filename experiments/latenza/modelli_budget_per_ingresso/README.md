# Pre e post per ingresso di deploy sul Raspberry Pi 5

**Scopo**: il budget del modello a 30 fps per ogni ingresso: `33,3 ms − pre − post`. Prima la figura della griglia
(`../modelli_latenza_per_ingresso/report/A_budget*`) usava un budget unico di 28 ms (stima della fase 1) per tutti gli ingressi;
`../modelli_pre_post_640` misura solo 640×640.

- `misura.py`: pre (uint8 HWC → float32 NCHW / 255) e post one-to-one (64 box → soglia 0,25) come
  `../modelli_pre_post_640/misura.py`, agli ingressi di `conf/deploy` (128², 256², 512², 640², 480p, 720p, 1080p) e a
  640×384 (720p senza cambiare le proporzioni). Un core, 1000 iterazioni dopo 100 di riscaldamento.
- `rifai.sh`: 3 passate sul Pi con il lock (aspetta le altre misure), `results/pre_post.jsonl`; poi rifà
  `A_budget` e `A_budget_dettaglio` in `../modelli_latenza_per_ingresso/report` con il budget di ogni ingresso
  (`tools/figure_sintesi.py --budget`).

```bash
experiments/latenza/modelli_budget_per_ingresso/rifai.sh
```

**Legenda della figura**: verde = latenza sotto il budget del suo ingresso −5%; giallo = latenza nel budget ±5%;
grigio = oltre. Non c'è più una soglia unica in ms: il budget cambia con l'ingresso (tabella sotto).

## Risultati

Mediana di 3 passate (20/11/2026), un core:

| ingresso | pre | post | budget del modello |
|---|---|---|---|
| 128² | 0,11 ms | 0,008 ms | 33,2 ms |
| 256² | 0,42 ms | 0,009 ms | 32,9 ms |
| 512² | 1,91 ms | 0,009 ms | 31,4 ms |
| 640² | 3,02 ms | 0,008 ms | 30,3 ms |
| 640×384 | 1,08 ms | 0,009 ms | 32,2 ms |
| 480p (864×480) | 3,04 ms | 0,008 ms | 30,3 ms |
| 720p (1280×736) | 6,99 ms | 0,009 ms | 26,3 ms |
| 1080p (1920×1088) | 15,48 ms | 0,009 ms | 17,8 ms |

- Il pre è proporzionale ai pixel (circa 7,4 ms per megapixel), tranne sotto i 640² dove la cache aiuta.
- La stima di prima (28 ms per tutti) era troppo bassa fino a 640² e troppo alta a 1080p (17,8 ms).
- La figura `A_budget` della griglia è rifatta con questi budget; le latenze sono quelle di `results/` solo con
  `strategy=baseline`.

## Limiti

- Il budget non toglie il resize dalla camera all'ingresso (es. 1080p → 640: `../track_costo_ritaglio`,
  `../scan_budget_720p`), né il ritaglio e il Kalman (`../track_budget_n_crop`).
- Le latenze di 128², 256², 512², 480p, 720p e 1080p nella figura sono quelle del 4/11/2026 (testa a k=300). Per
  rimisurarle a k=64: `FORCE=1 DEPLOY=crop128,crop256,crop512,480p,720p,1080p ../modelli_latenza_per_ingresso/rifai.sh` (rimandato).
