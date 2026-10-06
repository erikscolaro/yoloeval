# Esperimenti

Studi una tantum fuori dalla pipeline, raggruppati per stadio: `train`, `adapt`, `pruning`,
`finetune`, `quantization`, `deploy`, `benchmark`. Tenere aggiornato questo indice quando se ne
aggiunge o chiude uno.

Struttura uniforme di ogni esperimento:

- `README.md`: scopo, setup ed esito storico;
- `run.py`: unico punto d'ingresso (`--report` per i soli grafici, dove c'e'); gli script secondari hanno nome descrittivo;
- `config/`: file di configurazione in input o copie delle conf usate, se ce ne sono;
- `results/` e `figures/`: json/csv e grafici scritti da `run.py` (create al primo uso, vuote nel repo);
- `runs/`: output grezzo (run di Ultralytics, log, IR, modelli intermedi), ignorata da git (`experiments/**/runs/`).

Il codice comune sta in `_lib.py`: `ROOT` (root del repo, aggiunta a `sys.path`) e `PiRam`, il
campionatore della RAM del Pi usato dai benchmark. Ogni `run.py` lo importa con
`sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` seguito da `from _lib import ROOT`.
Gli script per il Pi (`pi_bench.py`, `bench.sh`) restano in `deploy/max_det` e
`deploy/pre_post_processing`: le due copie differiscono (modelli, passate, output).

I risultati AOD4 dei vecchi esperimenti sono nel tag git `aod4-results`; tutti vanno rifatti con
la pipeline AOD4 → Svanström.

## train

| cartella | scopo | stato |
|---|---|---|
| [`train/lr_scheduler`](train/lr_scheduler) | optimizer del training da zero di yolo26p (Adam, MuSGD, SGD) | da rifare (esito AOD4: MuSGD 0.960 / 0.649 alla 250, davanti dall'inizio; SGD 0.643, Adam 0.637) |
| [`train/imgsz`](train/imgsz) | risoluzione di training e validazione: yolo26p da zero a 1024 contro i training a 640, validati a 640 e 1024 | da rifare (esito AOD4: in corso, nessun risultato) |
| [`train/distill`](train/distill) | distillazione del pico dal nano (teacher n su AOD4, student p da zero, stessa ricetta), poi adapt | da lanciare (smoke test prima) |
| [`train/p2`](train/p2) | yolo26p con uscita P2/4 in piu' (686 k parametri, 2.5 GFLOPs contro 1.7), da zero su AOD4 e poi adapt | in corso (AOD4: 0.964 / 0.654, p 0.960 / 0.649, nano 0.969 / 0.684) |
| [`train/close_mosaic`](train/close_mosaic) | 15 epoche in piu' senza mosaic sul yolo26p MuSGD da zero | da rifare (esito AOD4: 0.960 / 0.649 → 0.962 / 0.652) |

## adapt

| cartella | scopo | stato |
|---|---|---|
| [`adapt/model_size`](adapt/model_size) | adapt di yolo26 n, s, m con la ricetta scelta, confronto con pico e pico con P2 | chiuso (val mAP50-95: p 0.470, P2 0.476, n 0.517, s 0.530, m 0.538) |
| [`adapt/optimizer`](adapt/optimizer) | optimizer dell'adattamento su Svanström (Adam, SGD, MuSGD), rete intera e backbone congelato | chiuso (6 run: MuSGD e SGD pari 0.462, Adam 0.436; il backbone congelato non aiuta) |
| [`adapt/imgsz`](adapt/imgsz) | risoluzione dell'adattamento su Svanström (1024 contro 640) | chiuso (nessun vantaggio: val 0.462 / 0.462, test 0.445 contro 0.471) |
| [`adapt/lr`](adapt/lr) | lr0 del finetune di adattamento su Svanström (0.005 contro 0.001) | chiuso (val mAP50-95 0.442 / 0.462 / 0.470 con lr0 0.001 / 0.0002 / 0.0001 + mosaic; ricetta in `conf/adapt/default.yaml`) |

## pruning

| cartella | scopo | stato |
|---|---|---|
| [`pruning/scratch_vs_finetune`](pruning/scratch_vs_finetune) | sull'architettura potata dalla search (50% MAC e params, N dal probe): finetune dai pesi ereditati contro training da zero, 500 epoche | da rifare (esito AOD4: finetune 0.959 / 0.645, scratch 0.954 / 0.639; distacco da +0.034 a +0.006) |
| [`pruning/probe_n_vs_cores`](pruning/probe_n_vs_cores) | N del probe (`probe_edge`, onnxruntime) sul Pi per int8, fp16 e fp32 su 1, 2 e 4 core | da rifare (esito AOD4: N scende con i core, int8 4/4/2, fp16 8/4/2, fp32 N=1) |

## finetune

| cartella | scopo | stato |
|---|---|---|
| [`finetune/lr_scheduler`](finetune/lr_scheduler) | optimizer del finetune di yolo26p potato (Adam, MuSGD, SGD) | da rifare (esito AOD4: MuSGD 0.923 mAP50, adottato in `finetune=musgd`) |

## quantization

| cartella | scopo | stato |
|---|---|---|
| [`quantization/calibration`](quantization/calibration) | metodo di calibrazione dell'int8 `int8_full` di onnxruntime (MinMax, media mobile, Percentile, Entropy) | da rifare (esito AOD4: media mobile e Percentile 0.519 / 0.518 di mAP50-95, MinMax 0.463, fp32 0.561) |
| [`quantization/int8_backend`](quantization/int8_backend) | int8 piu' lento di fp32 sul Pi: backend o nodi quantizzati? | da rifare (esito AOD4: conta quanti nodi sono int8; onnxruntime int8_full 22.0 ms, OpenVINO lento sul QDQ) |

## deploy

| cartella | scopo | stato |
|---|---|---|
| [`deploy/max_det`](deploy/max_det) | `max_det` (k del TopK nel grafo) da 1 a 300 sul yolo26p potato: latenza sul Pi e mAP | da rifare (esito AOD4: latenza piatta fino a 64, a 300 +0.5 ms; per AOD4 64) |
| [`deploy/pre_post_processing`](deploy/pre_post_processing) | pre/post-processing sul Pi (OpenVINO fp32 640x640, API Python e Ultralytics predict) e `max_det` 300 contro 50 | da rifare (esito AOD4: end-to-end 27.0 ms con frame 640 e max_det 50; pre 2.4 ms, post 0.1 ms) |

## benchmark

| cartella | scopo | stato |
|---|---|---|
| [`benchmark/pruned_finetuned`](benchmark/pruned_finetuned) | latenza, mAP e RAM sul Pi del yolo26p potato e finetunato (OpenVINO fp32/fp16/int8, 4 core) | da rifare (esito AOD4: fp32 22.4 ms, int8 piu' lento, 32.9 ms) |
| [`benchmark/yolo26p_baseline`](benchmark/yolo26p_baseline) | stesso benchmark con yolo26p non potato, per il guadagno di latenza del pruning | da rifare (esito AOD4: pruning 1.25× piu' veloce, 27.9 → 22.4 ms) |
| [`benchmark/yolo26p_p2`](benchmark/yolo26p_p2) | stesso benchmark con yolo26p con uscita P2 non potato | misurato (fp32 43.14 ms: +54% sul p, 1.32x piu' veloce del nano) |
| [`benchmark/yolo26n`](benchmark/yolo26n) | stesso benchmark con yolo26n (fp32), riferimento per i modelli pico | da rifare (esito AOD4: 57.0 ms; pico potato 2.55× piu' veloce) |
