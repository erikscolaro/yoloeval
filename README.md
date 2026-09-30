# yolo-bench

Misura latenza, accuratezza ed energia di YOLO26 su hardware edge. Confronta i modelli
Ultralytics (baseline) con quelli potati da [yolopit](https://github.com/erikscolaro/yolopit)
(ricerca PIT di PLiNIO, canali a blocchi di N). Dataset: AOD4 (airplane, bird, drone,
helicopter).

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
| `model` | `yolo26n`, `yolo26s`, `yolo26m`, `custom_pruned` |
| `strategy` | `baseline`, `pit_standard`, `pit_duccio`, `pit_auto` |
| `quantization` | `fp32`, `fp16`, `int8` (PTQ) |
| `backend` | `onnxruntime`, `onnxruntime_py`, `tensorrt`, `openvino`, `executorch`, `axelera` |
| `hardware` | `wks4_rtx6000`, `jetson_orin`, `rpi5`, `rpi5_axelera` |
| `freq_target` | profili della board (`maxn`, `w30`, `w15` sulla Jetson; `max`, `mid`, `low` sul Pi) |
| `compute_target` | `gpu`, `cpu_1`, `cpu_2`, …, `axelera` (quelli dichiarati dalla board; vuoto = tutti) |
| `train`, `finetune` | argomenti di training di Ultralytics |

`freq_target` e `compute_target` dipendono dalla board: un valore che la board non dichiara è
un errore immediato. `freq_target` va sempre indicato sulle board.

## Stadi

Si lanciano in quest'ordine; ognuno salta quello che trova già in cache.

| stadio | dove | cosa produce |
|---|---|---|
| `train` | workstation | pesi in `artifacts/weights/` |
| `search` | workstation | solo strategie pit: modello potato in `artifacts/search/` |
| `finetune` | workstation | solo strategie pit: fine-tuning in `artifacts/finetune/` |
| `probe` / `probe_edge` | board | N che premia l'hardware, in `results/probe/` |
| `roofline` / `roofline_edge` | board | picco di calcolo e banda, in `results/roofline/` |
| `export` | workstation (ONNX) o board (TensorRT, Axelera) | `artifacts/exports/`, con MAC e byte del modello |
| `benchmark` | board | un JSON per cella in `results/` |

I preset `_edge` usano reti più piccole e meno iterazioni, adatti a Raspberry Pi e CPU della
Jetson.

## Flussi

**Solo baseline**

```bash
python run.py -m stage=train model=yolo26n,yolo26s
python run.py -m stage=export model=yolo26n,yolo26s quantization=fp32,int8 backend=onnxruntime
python run.py -m stage=benchmark model=yolo26n,yolo26s quantization=fp32,int8 \
  backend=onnxruntime hardware=rpi5 freq_target=max compute_target=cpu_1,cpu_4
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
python -m tools.roofline                     # reports/roofline/: roofline.csv + un PNG per gruppo
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
- Le MAC sono calcolate alla risoluzione di deploy (`model.imgsz`).
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

## Risultati

```bash
python -m tools.aggregate --out data.parquet        # un DataFrame con tutte le celle
python -m tools.roofline                            # roofline e confronto con la baseline
jupyter lab notebooks/01_analysis.ipynb             # tabelle, probe, roofline
jupyter lab notebooks/02_figures.ipynb              # figure PNG + PDF per la tesi
python -m tools.report                              # reports/<data>/ e reports/latest
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

## Struttura

```
conf/        un gruppo per asse (stage, model, train, strategy, finetune, dataset,
             quantization, backend, hardware, eval, logging)
docs/config/ riferimento di ogni campo, per gruppo
src/stages/  train, search, finetune, probe, roofline, export, benchmark
src/probe/   reti dummy e scelta di N
src/backends/, src/remote/, src/measure/, src/validation/
tools/       aggregate, roofline, report, artifacts
notebooks/   analisi e figure
```

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
- Mai girato su una board vera: parser testati su output reali salvati, flusso completo
  provato solo in locale sulla workstation.
