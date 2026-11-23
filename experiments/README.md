# Esperimenti

Studi una tantum fuori dalla pipeline, raggruppati per stadio (`dataset`, `train`, `adapt`, `pruning`,
`finetune`, `quantization`, `deploy`, `benchmark`, `latenza`, `tracking`). Tenere aggiornato questo indice quando se ne
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

## deploy

| cartella | scopo | stato |
|---|---|---|
| [`deploy/ncnn_rpi5`](deploy/ncnn_rpi5) | backend NCNN sul Pi 5 contro OpenVINO, fp16 e int8, latenza e mAP sul Pi (P e N D126) | chiuso il 16/11: NCNN fp16 -8% su P, pari su N; int8 più lento dell'fp16 e -0,03 di mAP. Backend in `src/backends/ncnn.py`. Il 23/11 aggiunto fp32 (`rifai_fp32.sh`, `benchncnn_fp32` a fp16 spento): 2,2× più lento dell'fp16 su P e N (56,3 e 129,0 ms), stessa mAP |
| [`deploy/tiling_rpi5`](deploy/tiling_rpi5) | tiling dei primi strati (parallelismo sull'ingresso, 4 stream da 1 thread) contro OpenVINO a 4 thread | chiuso il 16/11: utile solo sullo strato 0 (fino a 1,14x, -0,4 ms), dallo strato 2 peggiora per l'halo; OpenVINO divide già ogni strato con efficienza 80-90% |

## benchmark

| cartella | scopo | stato |
|---|---|---|
| [`benchmark/overclock_rpi5`](benchmark/overclock_rpi5) | overclock di CPU (2400-3000 MHz) e GPU sul Pi 5, OpenVINO fp16 e NCNN, con `tryboot` | chiuso il 16/11: 3000 MHz stabile, -16% di latenza; la GPU non conta (<1%); NCNN scala peggio e a 3000 MHz non conviene più |

## latenza

Tutte sul Pi 5 (OpenVINO fp16, 4 core, governor performance, lock `runs/.pi.lock`), per il budget a 30 fps (33,3 ms).

| cartella | scopo | stato |
|---|---|---|
| [`latenza/modelli_latenza_per_ingresso`](latenza/modelli_latenza_per_ingresso) | latenza delle 27 reti della griglia (larghezza × profondità) a ogni ingresso di deploy; figura `A_budget` | figura rifatta il 20/11 con il budget di ogni ingresso e solo le reti `baseline` (la lettura mescolava le potate: N a 640 era 27,9 ms, vale 56,5) |
| [`latenza/modelli_pre_post_640`](latenza/modelli_pre_post_640) | pre (uint8 HWC → float32 NCHW / 255) e post one-to-one a 640×640 | chiuso; sostituito per gli altri ingressi da `modelli_budget_per_ingresso` |
| [`latenza/modelli_budget_per_ingresso`](latenza/modelli_budget_per_ingresso) | pre e post per ogni ingresso: budget del modello = 33,3 ms − pre − post | chiuso il 20/11: da 33,2 ms (128²) a 17,8 ms (1080p); 640×384 32,2 ms. k=300 delle altre risoluzioni da rimisurare a k=64 (rimandato) |
| [`latenza/scan_modelli_640x384`](latenza/scan_modelli_640x384) | latenza dei candidati SCAN a 640×384 (720p senza cambiare le proporzioni) | chiuso il 20/11: scelta PN- (24,3 ms); N- (31,4 ms) sta nel +5% ma guadagna meno di 0,01 di mAP. Senza mAP a 640×384 |
| [`latenza/scan_modelli_per_camera_e_ingresso`](latenza/scan_modelli_per_camera_e_ingresso) | miglior modello dello SCAN nel budget per ogni coppia camera (640, HD, FHD) × ingresso del modello, con la mAP del val 640 | chiuso il 23/11: a 640×384 regge ogni camera (PN-, P+ o P-); con ingresso HD o FHD nessun modello della griglia sta nel budget |
| [`latenza/scan_modelli_640x512`](latenza/scan_modelli_640x512) | latenza di P- e PN- con ingresso 640×512 (camera Svanström nativa) | chiuso il 23/11: P- 20,5 ms (max 25,3) nel budget di circa 31 ms; PN- 32,8 ms (max 37,3) fuori |
| [`latenza/scan_budget_720p`](latenza/scan_budget_720p) | costo fuori dal modello dello SCAN a 720p (resize, bordo, pre, post) | chiuso il 20/11: 2,6 ms, budget 30,7 ms (i 6,7 ms di prima stiravano l'immagine) |
| [`latenza/track_costo_ritaglio`](latenza/track_costo_ritaglio) | costo del ritaglio e del resize del crop, INTER_AREA contro INTER_LINEAR | chiuso il 19/11: da 1080p lo SCAN costa 11,5 ms con AREA e 2,9 ms con LINEAR |
| [`latenza/track_costo_kalman`](latenza/track_costo_kalman) | costo di `KalmanTracker.step` con N oggetti | chiuso il 19/11 |
| [`latenza/track_budget_n_crop`](latenza/track_budget_n_crop) | costo fuori dal modello di un fotogramma TRACK con N crop (ritaglio, Kalman, pre, post) | chiuso il 20/11: circa 0,93 ms per crop; budget 32,3 / 31,4 / 29,6 / 25,9 ms per N = 1 / 2 / 4 / 8 |
| [`latenza/track_modelli_batch_crop256`](latenza/track_modelli_batch_crop256) | latenza a batch 1, 2, 4, 8 con crop 256×256 per i 27 modelli, con Pareto contro la mAP | chiuso il 20/11; M non addestrata. Cause dei riavvii del Pi a batch 8 non note |
| [`latenza/modelli_carico_e_cambio`](latenza/modelli_carico_e_cambio) | tempo di caricamento di PN- (SCAN) e S- (TRACK) e costo del cambio fra i due, API Python di OpenVINO | chiuso il 23/11: cambio 0,1–0,2 ms con i due modelli compilati; caricamento 0,3–2,3 s; prima inferenza di S- +20 ms; memoria 212 MB |
| [`latenza/rete_una_uscita`](latenza/rete_una_uscita) | nano contro 3 reti a una sola uscita del backbone: latenza, ops, parametri (senza training) | lasciato a metà il 20/11: nano 56,5 ms; L3 32,2; L4 29,4; L5 34,9. Senza mAP |

## tracking

| cartella | scopo | stato |
|---|---|---|
| [`tracking/video_val_test`](tracking/video_val_test) | video completi di val e test (Svanström, Anti-UAV300) per tarare e provare il tracker | dati pronti; taratura non ancora fatta |
| [`tracking/crop_zoom_map`](tracking/crop_zoom_map) | zoom del crop adattativo che rende più mAP, per classe e fascia (P-, crop 256) | chiuso il 19/11: la regola di oggi è vicina all'ottimo da 16 px in su. Il lato del crop (128, 256, 384) non è stato confrontato |
| [`tracking/scan_map_per_ingresso`](tracking/scan_map_per_ingresso) | mAP di P- sul val con ingresso nativo 640×512, schiacciato a 640×384 e letterbox a 640×384 | chiuso il 23/11 (P- e PN-): schiacciato −0,064/−0,074 di mAP50-95, letterbox −0,029, persi soprattutto gli oggetti da 8 a 32 px; Svanström va a 640×512 |
| [`tracking/crop_interpolazione_map`](tracking/crop_interpolazione_map) | INTER_AREA o INTER_LINEAR nel crop tracker, in mAP | misurato il 20/11; scelta del 23/11: AREA ovunque (conservativa) |
