# yolo-bench

Misura latenza, accuratezza ed energia di YOLO26 su hardware edge. Confronta i modelli
Ultralytics (baseline) con quelli potati da [yolopit](https://github.com/erikscolaro/yolopit)
(ricerca PIT di PLiNIO; con `pit_latency` guidata dalla latenza del Pi 5 stimata da una LUT). Classi: airplane, bird, drone, helicopter.
Dataset: uno solo per training, pruning, fine-tuning e misure, **VEDETTA**, costruito e congelato nella repo
`~/Desktop/vedetta` (sei fonti pubbliche, 640×512, split per gruppi di sessioni; regole e procedura nel suo
README). Piano e documentazione in [`docs/`](#documentazione).

Un solo entry point, `run.py`, con configurazione [Hydra](https://hydra.cc): ogni asse è un
gruppo in `conf/`, ogni campo è documentato in [`docs/config/`](docs/config/).

## Installazione

Workstation (Python 3.11–3.13):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements/x86_64.txt       # include yolopit e PLiNIO
sudo apt install pandoc                              # solo per l'HTML del report
```

Board: alias SSH in `~/.ssh/config` (nessuna credenziale nel repo) e `sudo` senza password
per i comandi di tuning (`nvpmodel`, `jetson_clocks`, `cpufreq`). Poi:

```bash
python run.py stage=provision hardware=jetson_orin
python run.py stage=provision hardware=rpi5          # Pi senza scheda Axelera
python run.py stage=provision hardware=rpi5_axelera  # Pi con Metis: driver e container SDK
```

Board nuova, con solo un utente con sudo (es. `admin`): la prima volta, da terminale,

```bash
python run.py stage=provision hardware=rpi5 stage.bootstrap=admin@pi.local
```

crea l'utente `bench` senza password, con sudo senza password e la chiave SSH di questo PC,
e aggiunge l'alias `rpi5` a `~/.ssh/config`.

Il dataset viene copiato sulle board da solo al primo uso.

## Assi

| gruppo / chiave | valori |
|---|---|
| `model` | `yolo26f`, `yolo26p`, `yolo26n`, `yolo26s`, `yolo26m`; `model.depth=D101\|D126\|D170`; larghezze intermedie con `model=yolo26 model.width=0.1875` |
| `deploy` | `640` (il deploy di riferimento), `640x384`, `640x512`, `480p`, `720p`, `1080p` (ingressi 640×640, 640×384, 640×512, 864×480, 1280×736, 1920×1088); `crop128`, `crop256`, `crop512` (ritagli quadrati del tracker); `crop256_b2`, `_b4`, `_b8` (2, 4, 8 crop in un batch) |
| `rep` | passata della misura: `rep=0,1,2` per la mediana di 3 passate |
| `strategy` | `baseline`, `pit_latency` (DUCCIO sulla latenza stimata del Pi), `pit_standard`, `pit_duccio`, `pit_auto` |
| `quantization` | `fp32`, `fp16`, `int8` (PTQ di Conv, Gemm e MatMul), `int8_full` (quasi tutti i nodi) |
| `backend` | `onnxruntime`, `onnxruntime_py`, `tensorrt`, `openvino`, `openvino_qdq`, `ncnn`, `axelera` |
| `hardware` | `wks4_rtx6000`, `jetson_orin`, `rpi5`, `rpi5_axelera` |
| `freq_target` | profili della board (`maxn`, `w30`, `w15` sulla Jetson; `max`, `mid`, `low` sul Pi) |
| `compute_target` | `gpu`, `cpu_1`, `cpu_2`, …, `axelera` (quelli dichiarati dalla board; vuoto = tutti) |
| `train`, `finetune` | argomenti di training di Ultralytics: `default` è la ricetta unica, `random` salva i pesi casuali senza allenare; `finetune=scratch` allena il potato da zero |
| `target` | profilo di deploy: condizioni di misura, budget e LUT (`rpi5_640`), letto da `select` e `pit_latency` |

`freq_target` e `compute_target` dipendono dalla board: un valore che la board non dichiara è
un errore immediato. `freq_target` va sempre indicato sulle board.

## Stadi

Si lanciano in quest'ordine; ognuno salta quello che trova già in cache.

| stadio | dove | cosa produce |
|---|---|---|
| `provision` | board | driver, runtime e dataset sulla board |
| `train` | workstation | pesi allenati da zero (`pretrain_dataset`) in `artifacts/weights/` |
| `select` | workstation | modello di partenza per il profilo `target`, in `results/select/` (non misura niente) |
| `search` | workstation | solo strategie pit: ricerca dai pesi di `train`, modello potato in `artifacts/search/` |
| `finetune` | workstation | solo strategie pit: fine-tuning del potato in `artifacts/finetune/` |
| `probe` / `probe_edge` / `probe_p1` / `layers_b1` | board | N che premia l'hardware, in `results/probe/`; P1-1 e catene di strati B1-1 |
| `roofline` / `roofline_edge` | board | picco di calcolo e banda, in `results/roofline/` |
| `export` | workstation (ONNX) o board (TensorRT, Axelera) | `artifacts/exports/`, con MAC e byte del modello |
| `benchmark` | board | un JSON per cella in `results/` |

I preset `_edge` usano reti più piccole e meno iterazioni, adatti a Raspberry Pi e CPU della
Jetson.

## Flussi

**Latenza con pesi casuali (fasi 1 e 2a)**

```bash
python run.py -m stage=train train=random model=yolo26f,yolo26p,yolo26n,yolo26s,yolo26m
python run.py -m stage=export train=random model=yolo26n deploy=480p,720p,1080p \
  quantization=fp16 backend=openvino
python run.py -m stage=benchmark rep=0,1,2 train=random model=yolo26n deploy=480p,720p,1080p \
  quantization=fp16 backend=openvino hardware=rpi5 freq_target=max compute_target=cpu_4
```

**Training di mappatura (2b)**

```bash
python run.py -m stage=train train=default model=yolo26f,yolo26p,yolo26n,yolo26s,yolo26m
python run.py -m stage=train train=default model=yolo26n,yolo26s model.depth=D101
python run.py stage=train train=default model=yolo26n model.depth=D170
```

**Baseline contro potati**

```bash
python run.py stage=train model=yolo26n
python run.py -m stage=search   strategy=pit_standard,pit_duccio
python run.py -m stage=finetune strategy=pit_standard,pit_duccio
python run.py -m stage=export    strategy=baseline,pit_standard,pit_duccio quantization=fp32,int8
python run.py -m stage=benchmark strategy=baseline,pit_standard,pit_duccio quantization=fp32,int8 \
  hardware=rpi5 freq_target=max compute_target=cpu_1,cpu_4
```

**N misurata sulla board (`pit_auto`)**

```bash
python run.py stage=probe_edge hardware=rpi5 freq_target=max quantization=int8 compute_target=cpu_1
python run.py stage=search   strategy=pit_auto hardware=rpi5 quantization=int8
python run.py stage=finetune strategy=pit_auto hardware=rpi5 quantization=int8
```

`pit_auto` prende N dal probe onnxruntime con stessa board e precisione, per qualunque backend
(un export tensorrt int8 usa l'N del probe onnxruntime int8), sul compute target
`strategy.n_source` (default `cpu_1`). Nella chiave della ricerca entra il valore di N, non la
board. Se il probe manca, la ricerca lo lancia da sola (preset e profilo in `strategy.probe`),
lo scrive nel log e poi riparte: il primo comando qui sopra serve solo a scegliere il preset.

**Più target di DUCCIO**

```bash
python run.py -m stage=search strategy=pit_duccio strategy.search.regularizer.target.ops=30%,40%,50%
```

Stessi override per `finetune`, `export` e `benchmark`.

**Roofline**

```bash
python run.py -m stage=roofline_edge hardware=rpi5 freq_target=max quantization=fp32,int8
# ... export e benchmark dei modelli ...
python -m tools.roofline                     # lab/reports/roofline/: roofline.csv + un PNG per gruppo
```

Per ogni board × compute target × precisione × backend: intensità aritmetica (MAC/byte),
GMAC/s ottenuti, tetto raggiungibile, efficienza, memory/compute bound, speedup e rapporto di
MAC rispetto alla baseline. I tetti da datasheet, se scritti in `hardware.peaks`, compaiono
tratteggiati.

## Strategie

Una strategia (`conf/strategy/`) contiene la configurazione della ricerca PIT: argomenti di
Ultralytics più i gruppi di yolopit.

```yaml
name: pit_duccio
method: pit                  # ultralytics = baseline
search:
  epochs: 30
  lr0: 0.001
  pit:  {n: 16, remainder: true, warmup_epochs: 0, ema: false}    # n: auto -> dal probe
  nas:  {optimizer: AdamW, lr0: 0.01, lrf: 1.0, cos_lr: false}
  regularizer: {mode: duccio, target: {ops: 50%}}                 # oppure
  # regularizer: {mode: standard, lambda: {ops: 1.0, params: 0.0}}
```

- Target DUCCIO sul modello intero, assoluti (`1.2G`, `800M`) o in % del modello di partenza;
  sotto il minimo raggiungibile la ricerca si ferma con un errore.
- Ricerca e fine-tuning girano su `pretrain_dataset`; all'export vanno i pesi di
  `train` (baseline) o di `finetune` (pit).
- Le MAC della ricerca sono a `deploy.imgsz`: yolopit calcola i costi solo su ingressi
  quadrati, per 480p/720p/1080p va indicato `pit.trace_imgsz`.
- Il fine-tuning ha il suo gruppo `conf/finetune/`.

## Opzioni

| flag | effetto |
|---|---|
| `+dry_run=true` | mostra cosa verrebbe fatto, senza eseguire |
| `--cfg job --resolve` | stampa la config risolta |
| `+force=true` | rifà benchmark, probe e roofline già fatti |
| `+retry_failed=true` | riprova le celle fallite |
| `+force_retrain=true` / `+force_search=true` / `+force_finetune=true` / `+force_reexport=true` | rifà quello stadio |
| `+allow_reboot=true` | permette il riavvio della board per i cambi di profilo |
| `skip_invalid=false` | ferma lo sweep su una cella non valida invece di saltarla |
| `continue_on_error=false` | ferma lo sweep alla prima cella fallita |

`hydra/launcher=joblib` va bene per training ed export, **mai** per benchmark, probe o
roofline: misure in parallelo sullo stesso hardware non valgono niente.

## Launcher

`lab/launcher` (bash, dalla root) compone righe di comando di `run.py` e degli script senza modificare nessuno YAML:
tutto è un override Hydra.

- **ricette** (`lab/ricette/<nome>.ricetta`): step e override pronti. Prima riga `# descrizione`, poi `step a b c`
  e una riga `set chiave=valore` per override. Si salvano anche dal launcher.
- **step, modello, profilo, parametri**: una catena costruita a mano, con l'help di ogni campo da `docs/config/`.
- **predict**: detector, Kalman o tracker con crop dal vivo su video o sequenze (`tools/predict_video.py`); i modelli
  sono i `best.pt` in `artifacts/weights/` e `artifacts/finetune/`, più i pesi esterni in `artifacts/weights/test-*/`.
- **storico**: i run lanciati, con log, esito, rilancio e stop. I log vanno in `runs/`.
- **strumenti**: `scripts/*.sh`, `tools/*.py` e i `rifai.sh` degli esperimenti.
- **utilità**: stato della macchina e del Pi, test, misure per esito, report, artefatti, split del dataset,
  validazione di un modello, pulizia dei log di Hydra.

Le misure sul Pi (lanciate dal launcher o dagli script) prendono il lock `runs/.pi.lock`: una alla volta.

## Risultati

```bash
lab/launcher                                        # launcher: ricette, step, run, utilità
python -m tools.report                              # lab/reports/<data>/ e lab/reports/latest
python -m tools.results riepilogo | falliti         # misure per esito, fallite e perché
python -m tools.aggregate                           # lab/reports/data.parquet con tutte le misure
python -m tools.roofline                            # roofline e confronto con la baseline
python -m tools.artifacts ls | prune [--apply]      # cache degli artefatti
```

Opzioni e output di tutti i tool sono in [`docs/tools.md`](docs/tools.md).

Ogni cella registra latenza (media, mediana, p90/p95/p99, sempre batch 1, con il tool nativo
del backend), mAP (una volta per artefatto), energia dove ci sono sensori, temperatura,
throttling, frequenze, versioni, tempo macchina, validazione numerica contro l'FP32, strategia,
N e complessità del modello.

## Da sapere

- **Teste di YOLO26.** L'export può ricadere sulla testa one-to-many (con NMS): il tool
  registra quella usata davvero (`val_actual_e2e`).
- **Engine non portabili.** TensorRT e Axelera si compilano sulla board; la workstation
  produce solo `.pt` e `.onnx`.
- **Axelera.** Quantizza da sé in INT8: le celle `axelera + fp32` vengono saltate. Per i
  modelli potati il pacchetto yolopit viene copiato sulla board e messo nel PYTHONPATH del
  container, senza installarlo.
- **Profili di potenza.** Certi cambi di profilo vogliono il riavvio: raggruppa gli sweep per
  profilo. MAXN va letto insieme a `throttled`.
- **Caldo.** Il tool aspetta il raffreddamento prima di ogni misura e registra l'ordine di
  esecuzione, per vedere in analisi se la temperatura ha influito.

## Tracker

Per il deploy a 30 fps sul Pi c'è un tracker a due modi: SCAN cerca sul fotogramma intero con P-, TRACK guarda con S-
solo un ritaglio attorno alla predizione del filtro di Kalman (`src/tracking.py`, `src/crop_tracker.py`). Prova dal
vivo con `python -m tools.predict_video` o dal launcher. Progetto, latenza e risultati in
[`docs/tracker.md`](docs/tracker.md); taratura in `experiments/tracking/`.

Demo sul Pi 5 (`scripts/demo_pi.sh`, o dal launcher: utilita' -> demo-pi): lo stesso tracker tarato con gli export
OpenVINO fp16 di P- (640) e S- (crop 256), 4 core, sui video del val, a schermo intero sul monitor del Pi; `fine` spegne
il desktop e libera il Pi per i benchmark.

## Struttura

```
conf/         un gruppo per asse (stage, model, deploy, train, strategy, finetune, target, dataset,
              quantization, backend, hardware, eval, logging)
src/models.py architetture YOLO26 per larghezza e profondità
src/stages/   train, select, search, finetune, probe, roofline, export, benchmark
src/lut/      LUT di latenza del Pi 5, reti B4-R e fit dello stimatore (scripts/build_lut.sh)
src/tracking.py, src/crop_tracker.py   Kalman e tracker con crop
src/probe/, src/backends/, src/remote/, src/measure/, src/validation/
tools/        report, aggregate, roofline, artifacts, figure, predict_video
scripts/      provision delle board, requirements, catene e code di run lunghi
lab/          launcher (lab/launcher), ricette, template del report (report.md.j2)
experiments/  un esperimento per cartella (README, rifai.sh, risultati); indice in experiments/README.md
docs/         piano e documentazione (sotto)
```

## Documentazione

| file | cosa contiene |
|---|---|
| [`docs/NEW_MASTERPLAN.md`](docs/NEW_MASTERPLAN.md) | il piano della tesi: domande, regole fisse, fasi, decisioni |
| [`docs/todo.md`](docs/todo.md) | cose da fare e decisioni in sospeso |
| [`docs/stato-lavori-2026-11-16.md`](docs/stato-lavori-2026-11-16.md) | resoconto dei risultati al 16/11/2026 |
| [`docs/stimatore-latenza.md`](docs/stimatore-latenza.md) | lo stimatore di latenza del Pi (LUT e fit), con una sintesi divulgativa in fondo |
| [`docs/tracker.md`](docs/tracker.md) | il tracker con Kalman e crop |
| [`docs/tools.md`](docs/tools.md) | opzioni e output dei tool in `tools/` |
| [`docs/config/`](docs/config/) | riferimento di ogni campo di configurazione, per gruppo (usato anche dal launcher) |

## Test

```bash
pytest
```

Verificano soprattutto gli errori silenziosi: chiavi di cache (training e ricerca non
dipendono dall'hardware), parser sui veri output dei tool, probe, roofline, config
documentata.

## Limiti

- Probe e roofline solo con ONNX Runtime (`onnxruntime`, `onnxruntime_py`); TensorRT,
  OpenVINO e Axelera da aggiungere.
- Nessun isolamento dei core (`isolcpus`): richiederebbe modifiche permanenti al boot.
- Solo PTQ, niente QAT.
- mAP degli export solo a ingresso quadrato: il `val` di Ultralytics non valuta un export
  a forma fissa 864×480; la mAP ai set di deploy si ha per ora solo dai `.pt`.
- Nessun backend per la DX-M1 (DX-COM, DX-RT): si aggiunge con X1.
