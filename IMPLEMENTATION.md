# yolo-bench — Specifica implementativa

Documento destinato a chi (umano o modello) deve implementare il progetto.
Contiene decisioni già prese, vincoli non negoziabili e i dettagli che, se
ignorati, producono un tool che gira ma restituisce numeri sbagliati.

Leggere prima `README.md` per il contesto generale.

---

## 0. Decisioni già prese — non riaprire

| Tema | Decisione |
|---|---|
| Framework di configurazione | Hydra, gruppi per asse, multirun per la matrice |
| Training | API standard Ultralytics. Nessuna pipeline custom |
| Quantizzazione | Solo post-training (PTQ). QAT fuori scope |
| Misura della latenza | Solo tool nativi dei vendor (C++). Mai timer in Python |
| Confronto con PLiNIO | Fuori scope. Progetto separato |
| Formato per cella | Un file JSON |
| Formato aggregato | Parquet, prodotto da uno script di aggregazione |
| Credenziali SSH | Mai nel repo. Solo alias in `~/.ssh/config` |
| Modifiche di sistema | Sempre temporanee, ripristinate per cella |
| Isolamento core (`isolcpus`, cpuset) | **Non implementare.** Valutazione rimandata |
| NMS / end-to-end | Rilevato e loggato, **non** un asse della matrice |
| W&B | Opzionale, mirror. Il JSON locale è la fonte di verità |

---

## 1. Modello di esecuzione

### 1.1 Tre stadi, cardinalità decrescente in costo unitario

```
stage=train      model                                     → artifacts/weights/
stage=export     model × quantization × backend × arch     → artifacts/exports/
stage=benchmark  export × hardware × freq_target × compute_target → results/
```

Ogni stadio verifica la cache del precedente e la popola se manca. Uno stadio
non rigenera mai ciò che esiste già, salvo override esplicito.

### 1.2 Dove gira cosa

| Operazione | Dove |
|---|---|
| Training | Workstation |
| Export ONNX | Workstation (artefatto portabile) |
| Build engine TensorRT | **Board di destinazione** |
| Compilazione modello Axelera | **Board di destinazione** (richiede la scheda Metis presente) |
| Calcolo mAP | Workstation per ONNX; sulla board per artefatti hardware-specifici |
| Benchmark di latenza | Board di destinazione (workstation inclusa come target) |

Motivo del vincolo: un engine TensorRT è legato a GPU, architettura e versione
della libreria; un modello Axelera richiede l'hardware fisicamente presente in
fase di export ed è legato alla versione dell'SDK.

---

## 2. Layout della configurazione

```
conf/
├── config.yaml
├── stage/{train,export,benchmark,provision}.yaml
├── model/{yolo26n,yolo26s,yolo26m,custom_pruned}.yaml
├── train/{default,short}.yaml
├── dataset/aod4.yaml
├── quantization/{fp32,fp16,int8}.yaml
├── backend/{onnxruntime,onnxruntime_py,tensorrt,openvino,executorch,axelera}.yaml
├── hardware/{wks4_rtx6000,jetson_orin,rpi5}.yaml
├── eval/default.yaml
└── logging/{local,wandb}.yaml
```

**Un solo file per board.** `freq` e `compute` non sono gruppi Hydra separati:
sono dizionari dentro il file dell'hardware, selezionati per chiave con
`freq_target` e `compute_target`.

Motivo: con gruppi paralleli è sempre esprimibile un incrocio illegale
(`hardware=rpi5 compute=jetson_orin/gpu`, `compute_target=cpu_80`), che va poi
intercettato con controlli scritti a mano. Con i dizionari annidati quelle
combinazioni **non sono rappresentabili**: una chiave assente solleva
`KeyError` al momento della risoluzione, senza che nessuno abbia dovuto
scrivere una regola. Il costo è la duplicazione di poche righe tra board, che
in cambio rende ogni file leggibile da solo — aprendo `jetson_orin.yaml` si
vede subito cosa può girarci sopra.

### 2.1 `conf/config.yaml`

```yaml
defaults:
  - stage: benchmark
  - model: yolo26n
  - train: default
  - dataset: aod4
  - quantization: fp32
  - backend: onnxruntime
  - hardware: wks4_rtx6000
  - eval: default
  - logging: local
  - _self_

freq_target: default            # chiave dentro hardware.freq
compute_target: null            # chiave dentro hardware.compute; null = tutte

seed: 42
project_root: ${hydra:runtime.cwd}
artifacts_dir: ${project_root}/artifacts
results_dir:   ${project_root}/results

training_key_fields: [model, train, dataset]
cell_key_fields: [model, train, dataset, quantization, backend,
                  hardware, freq_target, compute_target, eval]

skip_invalid: true
continue_on_error: true
force: false
retry_failed: false
force_retrain: false
force_reexport: false
dry_run: false
allow_reboot: false             # riavvio automatico della board per cambio profilo

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
  job:
    chdir: true
  job_logging:
    formatters:
      simple:
        format: '[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
    handlers:
      console: {level: INFO}
      file:    {level: DEBUG}
    loggers:
      paramiko:    {level: WARNING}
      ultralytics: {level: WARNING}
```

### 2.2 Esempi di gruppo

**`conf/model/yolo26n.yaml`**

```yaml
name: yolo26n
source: pretrained        # pretrained | local
weights: yolo26n.pt
cfg: null
scale: n
imgsz: 640
nc: 4
```

**`conf/model/custom_pruned.yaml`**

```yaml
name: custom_pruned
source: local
weights: null                                      # oppure .pt di partenza
cfg: ${project_root}/models/custom_pruned.yaml     # formato Ultralytics nativo
scale: null
imgsz: 640
nc: 4
```

`train.py` sceglie `YOLO(cfg.model.weights)` o `YOLO(cfg.model.cfg)` in base a
`source`.

**`conf/train/default.yaml`** — i nomi delle chiavi devono coincidere 1:1 con
gli argomenti di `model.train()` di Ultralytics, così lo splat funziona senza
mapping.

```yaml
epochs: 300
batch: 16
imgsz: 640
optimizer: auto
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
patience: 50
seed: 42
mosaic: 1.0
mixup: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

**`conf/quantization/int8.yaml`**

```yaml
name: int8
precision: int8
method: ptq

calibration:
  n_samples: 512          # Axelera: 100–400, oltre non porta benefici
  source: val
  shuffle_seed: 42        # FISSO: calibration set identico per ogni artefatto
  cache: true

validation:
  max_abs_diff: 0.05
  map50_drop_tolerance: 0.02

backend_args:
  tensorrt:    ["--int8"]
  onnxruntime: {quant_format: QDQ, per_channel: true, reduce_range: false}
  openvino:    {preset: performance}
```

**`conf/quantization/fp16.yaml`**

```yaml
name: fp16
precision: fp16
method: cast                   # nessuna calibrazione

calibration: null

validation:
  max_abs_diff: 0.01
  map50_drop_tolerance: 0.005

backend_args:
  tensorrt:    ["--fp16"]
  onnxruntime: {}              # ORT usa fp16 via graph optimization
  openvino:    {preset: performance}
```

**`conf/quantization/fp32.yaml`**

```yaml
name: fp32
precision: fp32
method: null

calibration: null

validation: null               # è il riferimento, non c'è nulla da validare

backend_args:
  tensorrt:    []
  onnxruntime: {}
  openvino:    {}
```

**`conf/dataset/aod4.yaml`**

```yaml
name: aod4
local_path: /data/aod4                # path sulla workstation
yaml: /data/aod4/aod4.yaml            # data yaml in formato Ultralytics
nc: 4
names: [airplane, bird, drone, helicopter]
```

**`conf/backend/onnxruntime.yaml`**

```yaml
name: onnxruntime
export_format: onnx
requires:
  arch: [x86_64, aarch64]

build:
  on_target: false             # l'ONNX è portabile, si builda una volta
  opset: 17

benchmark:
  tool: onnxruntime_perf_test
  cmd: >
    onnxruntime_perf_test
      -m {model}
      -e {ep}
      -I
      -r {iters}
      -c {warmup_iters}
      -i "intra_op_num_threads|{threads}"
  iters: 200
  warmup_iters: 50
  parse: ort_perf_test
  metrics: [mean_ms, median_ms, p90_ms, p99_ms, throughput_qps]
```

`{ep}` viene risolto dal compute target: `cpu` → `cpu`, `cuda` → `cuda`.
`{threads}` viene risolto da `compute.n_cores`.
`-I` abilita la modalità di misurazione individuale (ogni iterazione misurata
separatamente, necessario per i percentili).

**`conf/backend/onnxruntime_py.yaml`** — opzionale, misura l'overhead Python

```yaml
name: onnxruntime_py
export_format: onnx
requires:
  arch: [x86_64, aarch64]

build:
  on_target: false
  opset: 17

benchmark:
  tool: python_timer           # timer interno, UNICA ECCEZIONE alla regola
  iters: 200
  warmup_iters: 50
  parse: python_timer
  metrics: [mean_ms, median_ms, p90_ms, p99_ms]
```

**`conf/backend/tensorrt.yaml`**

```yaml
name: tensorrt
export_format: engine
requires:
  trt: true
  arch: [x86_64, aarch64]

build:
  on_target: true
  from_format: onnx
  workspace_mb: 4096
  timeout_s: 1800

benchmark:
  tool: trtexec
  cmd: >
    trtexec --loadEngine={engine}
            --iterations={iters} --warmUp={warmup_ms}
            --avgRuns={iters} --duration=0
            --useSpinWait --noDataTransfers=false
  iters: 200
  warmup_ms: 2000
  parse: trtexec
  metrics: [mean_ms, median_ms, p90_ms, p99_ms, gpu_compute_ms, throughput_qps]
```

`--noDataTransfers=false` è obbligatorio: al default `trtexec` esclude i
trasferimenti host↔device e restituisce un tempo non confrontabile con quello
di `onnxruntime_perf_test`.

**`conf/backend/axelera.yaml`**

```yaml
name: axelera
export_format: axelera
requires:
  axelera: true
  arch: [aarch64, x86_64]
  os: [ubuntu22, ubuntu24]      # Raspberry Pi OS non supportato → container

build:
  on_target: true               # richiede la scheda Metis presente
  quantize: 8                   # forzato dall'AIPU, non è un asse
  calib_fraction: 400           # 100–400 raccomandato
  sdk_version: "1.8.0"          # ENTRA NELLA CHIAVE DI CACHE
  container: true
  image: yolo-bench-axelera:ubuntu22

benchmark:
  tool: ultralytics_predict     # nessun perf_test nativo equivalente
  iters: 200
  parse: ultralytics
  metrics: [mean_ms, median_ms, p90_ms, p99_ms]
```

**`conf/hardware/jetson_orin.yaml`**

```yaml
board: jetson_orin_nx
arch: aarch64
jetpack: "6.2"
soc: orin
n_cores_total: 8
trt_available: true
axelera_available: false
power_sensors: true            # tegrastats

remote:
  host: jetson-orin            # alias di ~/.ssh/config
  workdir: /home/<user>/bench
  python: /home/<user>/bench/.venv/bin/python

provision:
  script: scripts/provision/jetson_jp62.sh
  requirements: scripts/requirements/aarch64-jp62.txt

# ID e cores_online DA VERIFICARE sulla board con `sudo nvpmodel -q --verbose`:
# variano per variante di modulo (8GB / 16GB) e per versione di JetPack.
freq:
  maxn: {nvpmodel_id: 0, governor: performance, cores_online: 8}
  w30:  {nvpmodel_id: 1, governor: performance, cores_online: 8}
  w15:  {nvpmodel_id: 2, governor: performance, cores_online: 4}

compute:
  gpu:   {device: cuda}
  cpu_1: {device: cpu, n_cores: 1}
  cpu_2: {device: cpu, n_cores: 2}
  cpu_4: {device: cpu, n_cores: 4}
  cpu_8: {device: cpu, n_cores: 8, requires_freq: [maxn, w30]}
```

**`conf/hardware/rpi5.yaml`**

```yaml
board: rpi5
arch: aarch64
cpu: cortex_a76
n_cores_total: 4
shared_clock_domain: true      # i 4 core condividono il clock
trt_available: false
axelera_available: true
power_sensors: false

remote:
  host: rpi5
  workdir: /home/<user>/bench

provision:
  script: scripts/provision/rpi5_axelera.sh
  requirements: scripts/requirements/aarch64-rpi5.txt
  container: true

freq:
  max: {khz: 2400000, governor: performance}
  mid: {khz: 1500000, governor: userspace}
  low: {khz: 1000000, governor: userspace}

compute:                       # nessun requires_freq: i 4 core sono sempre online
  cpu_1:   {device: cpu, n_cores: 1}
  cpu_2:   {device: cpu, n_cores: 2}
  cpu_4:   {device: cpu, n_cores: 4}
  axelera: {device: metis}
```

**`conf/hardware/wks4_rtx6000.yaml`**

```yaml
board: wks4_rtx6000
arch: x86_64
n_cores_total: 24
trt_available: true
axelera_available: false
power_sensors: false
# nessuna chiave `remote`: esecuzione locale

freq:
  default: {governor: performance, restore: true}

compute:
  gpu:     {device: cuda}
  cpu_1:   {device: cpu, n_cores: 1}
  cpu_all: {device: cpu, n_cores: 24}
```

### 2.3 Risoluzione di `freq` e `compute`

```python
fq = cfg.hardware.freq[cfg.freq_target]          # KeyError se inesistente
ct = cfg.hardware.compute[cfg.compute_target]    # KeyError se inesistente
```

`compute_target: null` significa "tutti i target dichiarati dalla board":

```python
targets = ([cfg.compute_target] if cfg.compute_target
           else list(cfg.hardware.compute.keys()))
```

È un default, non una validazione: `compute_target=cpu_8` su `rpi5` non produce
una cella saltata, produce un errore immediato, perché quella chiave non esiste
nel file.

`requires_freq` è l'unica eccezione, e serve per un solo caso: su Jetson il
profilo di potenza determina quanti core sono online, quindi `cpu_8` esiste
come target della board ma non è disponibile in tutti i profili.

```python
req = ct.get("requires_freq")
if req and cfg.freq_target not in req:
    return False, f"{cfg.compute_target} non disponibile in {cfg.freq_target}"
```

È una dichiarazione locale nella riga del target, non una tabella di
compatibilità globale. `cores_online` nel blocco `freq` la rende
autodocumentante.

**`conf/eval/default.yaml`**

```yaml
map_conf: 0.001        # basso: la mAP integra su tutte le soglie
map_iou: 0.7
bench_conf: 0.25       # operativo, IDENTICO in tutte le celle di latenza
bench_iou: 0.45
max_det: 300
```

La distinzione è sostanziale. Per la mAP la soglia va tenuta bassa, altrimenti
la coda della curva precision-recall viene troncata e la metrica scende
artificialmente. Per la latenza la soglia va tenuta fissa e operativa, perché
il costo dell'NMS (quando presente) dipende da quante box superano il filtro:
soglie diverse fra celle significano carichi di post-processing diversi.

### 2.4 `wks4_rtx6000.yaml` — assenza di `remote`

Il file della workstation **non** contiene la chiave `remote`. La presenza o
assenza di quella chiave è l'unico discriminante fra esecuzione locale e
remota, e permette allo stesso codice di benchmark di funzionare in entrambi i
casi senza rami dedicati.

---

## 3. Chiavi, cache e resume

### 3.1 Tre chiavi distinte

```python
def training_key(cfg) -> str:
    payload = {k: OmegaConf.to_container(cfg[k], resolve=True)
               for k in cfg.training_key_fields}          # model, train, dataset
    payload["dataset_hash"] = read_dataset_hash(cfg.dataset.path)
    return _sha(payload)[:8]

def export_key(cfg) -> str:
    return (f"{cfg.model.name}_{training_key(cfg)}"
            f"_{cfg.quantization.name}_{cfg.backend.name}_{cfg.hardware.arch}"
            + (f"_sdk{cfg.backend.build.sdk_version}"
               if cfg.backend.name == "axelera" else ""))

def cell_key(cfg) -> str:
    payload = {k: OmegaConf.to_container(cfg[k], resolve=True)
               for k in cfg.cell_key_fields}
    return _sha(payload)[:12]
```

`training_key` **deve** dipendere solo da `model`, `train`, `dataset` (più
l'hash del dataset). Backend, quantizzazione, hardware, compute e power profile
non devono influenzarlo, altrimenti lo stesso modello verrebbe riallenato per
ogni cella della matrice.

`dataset_hash`: hash della lista ordinata dei file con le loro dimensioni, non
del contenuto. Salvato in `.dataset_hash` accanto ai dati. È la protezione
contro il riuso silenzioso di pesi allenati su un dataset diverso.

`export_key` include la versione dell'SDK per Axelera: un aggiornamento SDK
invalida i modelli `.axm` compilati in precedenza, che vengono rifiutati al
caricamento.

### 3.2 Naming degli artefatti

```
artifacts/
├── weights/
│   ├── yolo26n_aod4_e300_a3f2c891/
│   │   ├── best.pt
│   │   └── meta.json          # config completa, timestamp, metriche, git sha
│   └── index.json             # hash → config, per il listing leggibile
└── exports/
    ├── yolo26n_a3f2c891_int8_tensorrt_aarch64/
    └── yolo26n_a3f2c891_fp16_onnxruntime_x86_64/
```

Slug leggibile davanti, hash corto come suffisso disambiguante. Lo slug
contiene solo i campi che variano realmente fra le run.

### 3.3 Resume

`results/` è piatto e indicizzato per `cell_key`, separato dalle cartelle di
Hydra (che sono indicizzate per timestamp e quindi non stabili fra sweep).

```
results/
├── a91c3e7f0b22.json
└── a91c3e7f0b22.raw.txt      # stdout grezzo del tool di misura
```

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cid = cell_key(cfg)
    dest = Path(cfg.results_dir) / f"{cid}.json"

    if dest.exists() and not cfg.force:
        prev = json.loads(dest.read_text())
        st = prev["status"]
        if st == "ok":
            log.info(f"{cid} già completata, salto"); return
        if st == "skipped":
            log.info(f"{cid} non applicabile ({prev['reason']}), salto"); return
        if st == "failed" and not cfg.retry_failed:
            log.info(f"{cid} fallita in precedenza, salto "
                     f"(usa +retry_failed=true)"); return
        log.info(f"{cid} fallita, riprovo")

    run_cell(cfg, cid, dest)
```

```python
def run_cell(cfg, cid, dest):
    rec = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cid,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "env": collect_versions(),
        "order_index": int(os.environ.get("HYDRA_JOB_NUM", -1)),
        "timestamp": now_iso(),
    }
    try:
        ok, reason = is_valid(cfg)
        if not ok:
            rec |= {"status": "skipped", "reason": reason}
        else:
            with connection(cfg) as conn, tuned(conn, cfg):
                rec |= {"status": "ok", **execute_benchmark(conn, cfg, dest)}
    except Exception as e:
        rec |= {"status": "failed", "error": str(e),
                "trace": traceback.format_exc()}
    finally:
        atomic_write_json(dest, rec)      # scrivi su .tmp, poi rename
```

Il `finally` è obbligatorio: senza, una cella che esplode non lascia traccia e
verrebbe ritentata alla cieca a ogni rilancio. `atomic_write_json` scrive su
file temporaneo e poi rinomina, perché un'interruzione durante la scrittura
lascerebbe un JSON troncato che al rilancio verrebbe interpretato come cella
completata.

### 3.4 Dispatch degli stadi in `run.py`

`run.py` è l'unico entry point. Il dispatch avviene per `cfg.stage.name`:

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    stage = cfg.stage.name

    if stage == "provision":
        from src.remote.provision import provision
        provision(cfg)
        return

    if stage == "train":
        from src.stages.train import run_training
        run_training(cfg)
        return

    if stage == "export":
        from src.stages.export import run_export
        run_export(cfg)
        return

    if stage == "benchmark":
        cid = cell_key(cfg)
        dest = Path(cfg.results_dir) / f"{cid}.json"
        # ... logica di resume e run_cell come in 3.3
```

Ogni stadio ha il suo entry point e gestisce la propria cache in modo
indipendente. Il benchmark è l'unico che scrive in `results/`; training ed
export scrivono in `artifacts/`.

### 3.5 Flusso di `stages/train.py`

```python
def run_training(cfg):
    tkey = training_key(cfg)
    slug = f"{cfg.model.name}_{cfg.dataset.name}_e{cfg.train.epochs}"
    out_dir = Path(cfg.artifacts_dir) / "weights" / f"{slug}_{tkey}"

    # --- cache check ---
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not cfg.force_retrain:
        log.info(f"training già in cache: {out_dir.name}")
        return out_dir

    # --- training ---
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if cfg.model.source == "pretrained":
        model = YOLO(cfg.model.weights)
    else:
        model = YOLO(cfg.model.cfg)

    train_args = OmegaConf.to_container(cfg.train, resolve=True)
    results = model.train(
        data=cfg.dataset.yaml,
        project=str(out_dir),
        name="run",
        exist_ok=True,
        **train_args,
    )

    # --- salva artefatti ---
    best_pt = out_dir / "run" / "weights" / "best.pt"
    shutil.copy2(best_pt, out_dir / "best.pt")

    meta = {
        "training_key": tkey,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
        "metrics": {"map50": float(results.box.map50),
                    "map50_95": float(results.box.map)},
        "epochs_completed": results.epoch,
        "timing": {
            "wall_s": time.time() - t0,
            "device": cfg.hardware.board,
            "started_at": iso_now(),
        },
        "env": collect_versions(),
    }
    atomic_write_json(meta_path, meta)
    update_index(cfg.artifacts_dir / "weights" / "index.json", tkey, meta)
    return out_dir
```

### 3.6 Flusso di `stages/export.py`

```python
def run_export(cfg):
    # 1. trova i pesi allenati
    tkey = training_key(cfg)
    weights_dir = find_weights(cfg.artifacts_dir, tkey)
    if weights_dir is None:
        raise MissingWeights(f"nessun artefatto con training_key={tkey}")
    src_pt = weights_dir / "best.pt"

    # 2. calcola la chiave dell'export
    ekey = export_key(cfg)
    dst_dir = Path(cfg.artifacts_dir) / "exports" / ekey

    if (dst_dir / "meta.json").exists() and not cfg.force_reexport:
        log.info(f"export già in cache: {ekey}")
        return dst_dir

    # 3. dispatch al backend
    backend = get_backend(cfg.backend.name)     # registry → Backend
    dst_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with connection(cfg) as conn:
        # se on_target: rsync del .pt sulla board, esporta là
        if backend.builds_on_target:
            ensure_env(conn, cfg)
            ensure_dataset(conn, cfg)           # serve per calibrazione INT8
            remote_pt = sync_artifact(conn, cfg, src_pt)
            artifact = backend.export(conn, cfg, remote_pt, dst_dir)
        else:
            artifact = backend.export(LocalConnection(), cfg, src_pt, dst_dir)

        # 4. validazione numerica
        val = backend.validate(conn, cfg, artifact, ref=src_pt)
        if val["status"] == "failed":
            log.error(f"validazione fallita per {ekey}: {val}")

        # 5. ispezione e2e
        info = backend.inspect(artifact)

    meta = {
        "export_key": ekey,
        "training_key": tkey,
        "validation": val,
        "inspection": info,
        "timing": {"wall_s": time.time() - t0, "device": cfg.hardware.board},
        "config": OmegaConf.to_container(cfg, resolve=True),
        "env": collect_versions(),
    }
    atomic_write_json(dst_dir / "meta.json", meta)
    return dst_dir
```

`find_weights` cerca in `artifacts/weights/` la directory il cui `meta.json`
contiene il `training_key` corrispondente. Non assume il nome della cartella.

### 3.7 Flusso di `execute_benchmark` in `stages/benchmark.py`

```python
def execute_benchmark(conn, cfg, raw_dest):
    backend = get_backend(cfg.backend.name)

    # 1. trova l'artefatto esportato
    ekey = export_key(cfg)
    export_dir = find_export(cfg.artifacts_dir, ekey)
    if export_dir is None:
        raise MissingExport(f"nessun artefatto con export_key={ekey}")
    artifact = locate_artifact(export_dir, cfg.backend.export_format)

    # se l'artefatto è on_target, è già sulla board; altrimenti rsync
    if backend.builds_on_target:
        remote_artifact = Path(cfg.hardware.remote.workdir) / "exports" / ekey
    else:
        remote_artifact = sync_artifact(conn, cfg, artifact)

    # 2. risolvi compute target
    ct = cfg.hardware.compute[cfg.compute_target]
    n_cores = ct.get("n_cores")
    if n_cores == "all" or n_cores is None:
        n_cores = board_controller(cfg).online_cores(conn)
    affinity = f"0-{n_cores-1}" if ct.device == "cpu" and n_cores > 1 else None

    # 3. componi la command line
    cmd = backend.build_cmd(cfg, remote_artifact)

    # wrapping: affinity + thread + priorità
    env_vars = {}
    if ct.device == "cpu":
        env_vars["OMP_NUM_THREADS"] = str(n_cores)
    if affinity:
        cmd = f"taskset -c {affinity} {cmd}"
    cmd = f"chrt -f 80 {cmd}"

    # 4. esegui
    t0 = time.time()
    r = conn.run(cmd, hide=True, warn=True, env=env_vars)
    wall_s = time.time() - t0

    # 5. salva raw stdout
    raw_dest.with_suffix(".raw.txt").write_text(r.stdout)
    log.debug(f"stdout:\n{r.stdout}")

    if r.failed:
        raise BenchmarkFailed(f"exit code {r.return_code}:\n{r.stderr}")

    # 6. parsa
    lat = backend.parse(r.stdout)

    # 7. raccogli contesto
    bc = board_controller(cfg)
    result = {
        "latency": asdict(lat),
        "runtime_state": {
            "freq_requested_khz": cfg.hardware.freq[cfg.freq_target].get("khz"),
            "freq_actual_khz": bc.read_freq(conn),
            "governor": bc.read_governor(conn),
            "cores_online": bc.online_cores(conn),
            "omp_num_threads": n_cores,
            "temp_start_c": bc.read_temp(conn),     # già letto in settle()
            "temp_end_c": bc.read_temp(conn),
            "throttled": bc.read_throttle(conn),
            "swap_off": bc.swap_is_off(conn),
            "isolation": "none",
            "post_reboot": getattr(cfg, "_post_reboot", False),
        },
        "timing": {"wall_s": wall_s, "device": cfg.hardware.board},
    }

    # 8. accuratezza (una volta per export, non per cella)
    acc_path = Path(cfg.artifacts_dir) / "exports" / ekey / "accuracy.json"
    if not acc_path.exists():
        acc = compute_map(conn, cfg, remote_artifact)
        atomic_write_json(acc_path, acc)
    else:
        acc = json.loads(acc_path.read_text())
    result["accuracy"] = acc

    # 9. energia (se disponibile)
    if cfg.hardware.get("power_sensors"):
        result["energy"] = bc.read_energy(conn)

    # 10. validazione (già calcolata all'export, la riporta)
    export_meta = json.loads((Path(cfg.artifacts_dir) / "exports" / ekey / "meta.json").read_text())
    result["validation"] = export_meta.get("validation", {})
    result["validation"]["requested_e2e"] = True
    result["validation"]["actual_e2e"] = export_meta.get("inspection", {}).get("actual_e2e")

    return result
```

Questo è il flusso completo. La separazione delle responsabilità:
- `benchmark.py` orchestra (trova artefatti, risolve compute, raccoglie contesto)
- il backend compone il comando e parsa l'output
- `remote/` gestisce la connessione
- `measure/` legge lo stato della board

---

## 4. Validazione delle celle

Nessuna tabella di compatibilità hardcoded in Python. La maggior parte delle
combinazioni illegali non è nemmeno esprimibile, perché `freq` e `compute` sono
chiavi dentro il file della board: una chiave assente solleva `KeyError` prima
di qualunque validazione. Resta da controllare solo ciò che dipende dal
backend e dal runtime.

```python
def is_valid(cfg, conn=None) -> tuple[bool, str | None]:
    hw, be = cfg.hardware, cfg.backend
    fq = hw.freq[cfg.freq_target]              # KeyError se inesistente
    ct = hw.compute[cfg.compute_target]        # KeyError se inesistente

    # capability del backend vs capability della board
    if be.requires.get("trt") and not hw.trt_available:
        return False, "tensorrt non disponibile su questa board"
    if be.requires.get("axelera") and not hw.axelera_available:
        return False, "acceleratore Axelera non presente"
    if hw.arch not in be.requires.arch:
        return False, f"backend non supportato su {hw.arch}"

    # coerenza backend / device
    if ct.device == "metis" and be.name != "axelera":
        return False, f"{be.name} non può girare sull'AIPU"
    if ct.device == "cuda" and not be.get("supports_gpu", True):
        return False, f"{be.name} non supporta GPU"

    # unico vincolo interno alla board: core online per profilo
    req = ct.get("requires_freq")
    if req and cfg.freq_target not in req:
        return False, f"{cfg.compute_target} non disponibile in {cfg.freq_target}"

    # dinamico, solo se già connessi
    if conn and ct.device == "cpu" and ct.n_cores > online_cores(conn):
        return False, f"richiesti {ct.n_cores} core, online {online_cores(conn)}"
    return True, None
```

La validazione avviene in due momenti: **statica** prima di connettersi, per
non pagare SSH su celle che verranno scartate; **dinamica** dopo la
connessione, solo per il conteggio dei core online.

**Attenzione al conteggio dei core su Jetson.** `nvpmodel` controlla
esplicitamente quali core sono online, uno per uno: il profilo `MODE_15W` della
developer guide NVIDIA porta online i core 0-3 e offline i core 4-7, insieme ai
tetti di frequenza per CPU, GPU, EMC, DLA e PVA. I valori esatti dipendono
dalla variante di modulo e dalla versione di JetPack, quindi `cores_online` nel
config va compilato leggendo `sudo nvpmodel -q --verbose` sulla board, mai
copiato dalla documentazione. Il numero va comunque riletto a runtime da
`/sys/devices/system/cpu/online` o `nproc`, e registrato nel risultato.

Una cella non valida non solleva: scrive un `results.json` con
`status: "skipped"` e il motivo. Senza questo, in fase di analisi non si
distingue una combinazione non supportata da una che è crashata.

---

## 5. Connessione remota e provisioning

### 5.1 Connessione

Fabric, che legge `~/.ssh/config`. Nessun host, utente, chiave o password nel
repository o nei config.

```python
from fabric import Connection

@contextmanager
def connection(cfg):
    if "remote" not in cfg.hardware:        # workstation: esecuzione locale
        yield LocalConnection()
        return
    conn = Connection(cfg.hardware.remote.host)
    try:
        yield conn
    finally:
        conn.close()
```

`LocalConnection` espone la stessa interfaccia (`run`, `put`, `get`) eseguendo
in locale, così `execute_benchmark` non contiene rami locale/remoto.

I config dichiarano solo l'**alias**, mai IP o credenziali. `ControlMaster` in
`~/.ssh/config` riusa una sola connessione TCP per l'intero sweep.

Nota: la config risolta viene copiata da Hydra in `.hydra/config.yaml` per ogni
cella. Un segreto nei config si troverebbe duplicato in centinaia di cartelle.

### 5.2 Provisioning

Idempotente, con sentinella basata sull'hash dei requirements:

```python
def ensure_env(conn, cfg):
    want = sha256_file(cfg.hardware.provision.requirements)[:12]
    r = conn.run(f"cat {cfg.hardware.remote.workdir}/.env_hash", hide=True, warn=True)
    if r.ok and r.stdout.strip() == want:
        return
    conn.put(cfg.hardware.provision.script, "/tmp/provision.sh")
    conn.run("bash /tmp/provision.sh", pty=True)
    conn.run(f"echo {want} > {cfg.hardware.remote.workdir}/.env_hash")
```

Gli script di provisioning sono **per board**, perché la logica non è
esprimibile in YAML senza inventare un DSL. Il config dichiara *quale* script;
lo script contiene il *come*.

**Jetson:** i wheel di PyTorch e ONNX Runtime da PyPI non funzionano su
aarch64 Tegra. Servono i build NVIDIA per la specifica versione di JetPack. Lo
script deve verificare `/etc/nv_tegra_release` contro il campo `jetpack` del
config e fallire subito in caso di mismatch, invece di lasciare emergere un
import error a metà sweep.

**Raspberry Pi + Axelera:**

1. `metis-dkms` (≥ 1.6.2 per SDK 1.8) installato **sull'host**, perché è un
   kernel module. Repository apt di Axelera con chiave GPG.
2. `modprobe metis`, poi verifica con `axdevice`.
3. Container Ubuntu 22.04 con dentro Voyager SDK e Ultralytics, perché
   Raspberry Pi OS non è una piattaforma supportata.
4. `sudo apt install libgl1` (richiesto da OpenCV, non incluso via pip).

Il driver kernel e l'SDK si aggiornano separatamente e **nulla verifica la
coerenza al momento dell'installazione**: un driver sotto la versione minima
lascia il device inaccessibile con un errore di connessione. Quindi
`axdevice` va eseguito come health check prima di ogni sweep che coinvolga
l'acceleratore.

### 5.3 Ciclo di vita del container

Il container viene **fermato** (`docker stop`) alla fine dello sweep, non
rimosso, per non ricostruirlo al lancio successivo. Per le celle CPU-only sulla
stessa board il container deve essere fermo, non semplicemente inattivo: il
daemon e i processi interni competono comunque per CPU e memoria.

### 5.4 Sincronizzazione del dataset

```python
def ensure_dataset(conn, cfg):
    remote = cfg.hardware.remote.workdir + "/data/aod4"
    if conn.run(f"test -f {remote}/.complete", warn=True).ok:
        return
    local(f"rsync -az --partial --info=progress2 "
          f"{cfg.dataset.local_path}/ {cfg.hardware.remote.host}:{remote}/")
    conn.run(f"touch {remote}/.complete")
```

`rsync` è incrementale e riprendibile. Al secondo lancio non trasferisce nulla.

Per la sola latenza basta un subset rappresentativo: la mAP si calcola dove
conviene, non necessariamente sulla board. Se la board monta una microSD,
l'I/O può inquinare le misure — preferire NVMe per dataset e workdir dove
disponibile.

---

## 6. Tuning della board — sempre temporaneo

### 6.1 Context manager

```python
@contextmanager
def tuned(conn, cfg):
    saved = read_state(conn, cfg)          # governor, freq, nvpmodel, swap
    try:
        apply_state(conn, cfg)
        settle(conn, cfg)                  # stabilizzazione + attesa termica
        yield
    finally:
        restore_state(conn, saved)         # eseguito anche su eccezione
```

Granularità **per cella**, non per sweep. Tutto passa da sysfs o da comandi
runtime: niente modifiche a `/boot/firmware/config.txt`, niente unit systemd,
niente parametri kernel. Questo è anche il motivo per cui `isolcpus` resta
fuori: richiederebbe un boot modificato e quindi una modifica persistente.

Per sweep lunghi non presidiati, prevedere un guard lato board: uno script che
salva lo stato originale su file e un timer che lo ripristina se non riceve un
heartbeat entro N minuti. Il `finally` Python non copre `kill -9` né una
connessione che cade.

Registrare nel risultato sia lo stato pre-applicazione sia quello
post-ripristino: se una cella lascia la macchina in uno stato diverso da come
l'ha trovata, deve essere visibile subito.

### 6.2 Cosa applicare

**Governor.** Default `performance` su tutte le board: frequenza sempre al
massimo consentito, nessuna transizione durante la misura. Un governor
dinamico (`ondemand`, `schedutil`) fa partire l'inferenza a frequenza bassa
mentre risale, e su misure da pochi millisecondi la rampa è una quota
significativa e variabile.

`userspace` + `scaling_setspeed` solo quando la frequenza è un asse
dell'esperimento.

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# oppure
echo userspace | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 1500000   | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed

cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq   # SEMPRE rileggere
```

Un valore non supportato viene arrotondato in silenzio: registrare
`freq_requested` e `freq_actual` come campi separati.

**Gerarchia su Jetson.** `nvpmodel` impone il tetto, il governor sfrutta il
tetto. L'ordine è: prima `nvpmodel -m <id>`, poi eventualmente cpufreq nel
range consentito da quel profilo. Scrivere una frequenza superiore al tetto non
ha effetto. Dopo un cambio di profilo servono alcune decine di secondi di
stabilizzazione.

**Il cambio di profilo non è sempre applicabile a runtime.** Se la transizione
richiede un valore diverso di `tpc_pg_mask`, NVIDIA documenta che è necessario
il riavvio del sistema. Questo rompe l'assunzione che `tuned()` possa cambiare
profilo per singola cella.

Conseguenza sull'architettura: il cambio di `freq_target` è un'operazione di
**livello superiore allo sweep**, non un'azione della singola cella. Lo sweep va
raggruppato per profilo — cosa già consigliata per il termico:

1. applica il profilo,
2. verifica con `nvpmodel -q` che sia attivo,
3. esegui tutte le celle di quel profilo,
4. passa al successivo.

Se dopo l'applicazione il profilo attivo non corrisponde a quello richiesto, il
tool deve **fermarsi con un messaggio esplicito**, non proseguire: eseguire una
matrice intera su un profilo diverso da quello dichiarato produce risultati
silenziosamente sbagliati.

#### Gestione del riavvio

```python
def set_profile(conn, cfg, target):
    fq = cfg.hardware.freq[target]
    r = conn.run(f"sudo nvpmodel -m {fq.nvpmodel_id}", warn=True, pty=True)

    if needs_reboot(r.stdout):
        if not cfg.allow_reboot:
            raise ProfileChangeRequiresReboot(
                f"{target} richiede riavvio. Rilancia con +allow_reboot=true, "
                f"oppure riavvia a mano e riprendi lo sweep.")
        conn = reboot_and_wait(conn, cfg)
        ensure_env(conn, cfg)              # lo stato post-reboot è azzerato

    active = parse_active_profile(conn.run("sudo nvpmodel -q", hide=True).stdout)
    if active != fq.nvpmodel_id:
        raise ProfileMismatch(f"richiesto {fq.nvpmodel_id}, attivo {active}")
    return conn
```

```python
def reboot_and_wait(conn, cfg, timeout_s=300, poll_s=5, grace_s=15):
    host = cfg.hardware.remote.host
    log.warning(f"riavvio {host}")
    conn.run("sudo shutdown -r now", warn=True, disown=True)
    conn.close()
    clear_control_socket(host)             # il ControlPath resta stantio
    time.sleep(grace_s)                    # la board risponde ancora per qualche secondo

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            c = Connection(host, connect_timeout=5,
                           connect_kwargs={"banner_timeout": 5})
            c.run("true", hide=True, timeout=5)
            log.info(f"board tornata dopo {time.time()-t0:.0f}s")
            return c
        except Exception:
            time.sleep(poll_s)
    raise BoardUnreachable(f"{host} non tornata entro {timeout_s}s")
```

Quattro dettagli senza i quali il riavvio blocca lo sweep invece di gestirlo.

**Grace period iniziale.** Per qualche secondo dopo `shutdown -r` la board
risponde ancora a SSH. Senza attesa ci si riconnette al sistema che sta
terminando, la connessione cade subito dopo e il reboot sembra fallito.

**Connessione ricreata, non riusata.** Con `ControlMaster` attivo il socket di
controllo resta stantio dopo il riavvio e i comandi falliscono in modo
fuorviante. Rimuovere esplicitamente il `ControlPath` prima di riconnettersi.

**Stato azzerato.** Dopo il reboot il container Axelera è fermo, il modulo
`metis` può non essere caricato, lo swap è riattivo e il governor è tornato al
default. Va richiamato il provisioning check, non solo un ping.

**Board fredda.** È l'unico effetto favorevole: la prima cella dopo il riavvio
parte da temperatura ambiente mentre le altre partono dal cooldown
condizionale. Passa comunque da `wait_thermal`, ma registrare
`post_reboot: true` nel risultato permette di verificare in analisi che non ci
sia un salto sistematico.

`allow_reboot` ha default **`false`**. Con lo sweep raggruppato per profilo i
riavvii sono due o tre in tutta la matrice, quindi eseguirli a mano è
sostenibile, e uno sweep notturno non deve poter riavviare una board mentre
nessuno guarda. Va attivato esplicitamente.

`nvpmodel` chiede conferma interattiva quando il riavvio è necessario: serve
`pty=True` con risposta automatica, oppure intercettare il prompt e riavviare
esplicitamente.

**MAXN non è "il profilo veloce".** NVIDIA lo descrive come modalità non
vincolata e sperimentale per la regolazione dei clock, che non garantisce le
prestazioni migliori in tutti i casi d'uso perché il throttling hardware
interviene quando la potenza totale del modulo supera il budget TDP, e
sconsiglia carichi pesanti prolungati in quella modalità. Sulle celle MAXN il
campo `throttled` è quindi più importante che altrove, e il fatto va riportato
in tesi invece di presentare MAXN come limite superiore pulito.

Sulla Orin esistono inoltre i domini GPU, DLA ed EMC, assenti sul Pi.
`jetson_clocks` li fissa tutti al massimo del profilo corrente. Per i test su
GPU, fissare il clock GPU conta quanto fissare quello CPU.

**Dominio di clock condiviso sul Pi 5.** I quattro core condividono il clock:
scrivere su `cpu0` li muove tutti. La frequenza è quindi proprietà
dell'`hardware`, non del `compute`. Verificare che `userspace` sia disponibile
in `scaling_available_governors`; l'alternativa (`arm_freq` in
`/boot/firmware/config.txt`) richiede reboot ed è persistente, quindi da
evitare.

**Altre misure**, in ordine di efficacia:

- `chrt -f 80 taskset -c <affinity> <cmd>` — scheduling FIFO, riduce il jitter
  senza reboot.
- Servizi disattivati: desktop (`systemctl set-default multi-user.target`),
  bluetooth, avahi, cups, snapd e soprattutto `unattended-upgrades`, che si
  attiva da solo e può rovinare un blocco di celle.
- `swapoff -a` durante le misure: meglio un OOM visibile che una cella in swap.
- `sync && echo 3 > /proc/sys/vm/drop_caches` prima di ogni cella, così il
  primo caricamento del modello è confrontabile fra celle.
- Verifica del carico (`uptime` sotto soglia) prima di partire.

**Pinning e thread.** `taskset` limita su quali core il processo *può* girare,
ma ONNX Runtime e PyTorch decidono quanti thread creare leggendo il numero di
CPU totali. Pinnare su 2 core senza limitare i thread produce 8 thread su 2
core, con latenze peggiori del caso monocore. Impostare sempre sia
`OMP_NUM_THREADS` sia `intra_op_num_threads`, coerenti con `compute.n_cores`.

### 6.3 Gestione termica

```python
def wait_thermal(conn, threshold_c=45, timeout_s=300):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        t = read_temp(conn)
        if t <= threshold_c:
            return t
        time.sleep(10)
    log.warning(f"timeout termico, procedo a {t}°C")
    return t
```

Cooldown **condizionale**, non fisso: non spreca tempo quando la board è già
fredda, attende davvero quando serve.

Registrare in ogni risultato: `temp_start`, `temp_end`, `throttled`,
`order_index`. Se emerge una correlazione fra ordine di esecuzione e latenza,
il termico stava inquinando le misure e lo si vede in analisi.

| Board | Temperatura | Throttling | Potenza |
|---|---|---|---|
| Jetson | `tegrastats` | contatori in `/sys/devices/virtual/thermal/` | `tegrastats` (sensori INA) |
| Pi 5 | `vcgencmd measure_temp` | `vcgencmd get_throttled` | non disponibile |

Sul Pi il firmware applica un proprio throttling **indipendente dal governor**,
per temperatura e per tensione di alimentazione: si può avere `userspace` a
1.5 GHz e il firmware che scala comunque. `vcgencmd get_throttled` diverso da
zero significa cella contaminata.

Ordinare lo sweep **per power profile**, non per modello: ogni cambio di
profilo richiede stabilizzazione, e ordinando così se ne fanno pochi invece di
uno per cella.

---

## 7. Misura

### 7.1 Principio

Il benchmark **orchestra, non cronometra**. Ogni backend ha un tool nativo del
vendor, scritto in C++, che riporta il tempo del solo motore di inferenza. Il
tool compone la command line, la esegue e ne parsa l'output.

| Backend | Tool |
|---|---|
| ONNX Runtime | `onnxruntime_perf_test` |
| TensorRT | `trtexec` |
| OpenVINO | `benchmark_app` |
| ExecuTorch | runner C++ della build |
| Axelera | `yolo predict` via Voyager SDK |

Motivo: il package Python di ONNX Runtime è un binding sottile su una libreria
C++. L'inferenza avviene interamente in C++ in entrambi i casi; ciò che cambia
è l'overhead intorno alla chiamata (GIL, conversione numpy↔OrtValue, copie,
allocazione degli oggetti di ritorno). Quell'overhead è **costante**, non
proporzionale: su un modello nano quantizzato su CPU ARM a basso clock diventa
una quota rilevante e **sottostima lo speedup della quantizzazione**, che è
esattamente il fenomeno da misurare.

Un backend `onnxruntime_py` opzionale permette di quantificare quell'overhead
come risultato a sé.

### 7.2 Parametri fissi in tutte le celle

- `batch=1`
- `imgsz` dichiarata esplicitamente
- numero di iterazioni e warm-up identici
- input fisso: stesse immagini, stesso ordine
- perimetro della misura: **end-to-end sul device, trasferimenti inclusi,
  pre/post-processing escluso**

Ogni tool definisce "latenza" a modo suo (`trtexec` di default esclude i
trasferimenti, `perf_test` no). Il perimetro va allineato esplicitamente e
documentato, altrimenti si confrontano quantità diverse.

Riportare sempre mediana e p95/p99, mai la sola media: su edge, con throttling
e governor, la coda conta più del valore centrale.

### 7.3 Accuratezza

mAP@50 e mAP@50-95 si calcolano **una volta per artefatto esportato**, non per
cella: non dipendono da power profile, numero di core o frequenza. Tenerle
separate evita di ripetere la validazione su ogni riga della matrice.

### 7.4 Verifica di correttezza dell'export

Obbligatoria **prima** di ammettere un artefatto al benchmark. Un modello INT8
mal calibrato è velocissimo e predice rumore: senza controllo comparirebbe come
il miglior risultato in tabella.

Due livelli:

1. **Centrale (workstation):** output ONNX contro output PyTorch su un batch
   fisso. Intercetta errori di export.
2. **Sulla board:** output dell'artefatto hardware-specifico (engine TensorRT,
   modello Axelera) contro FP32. Intercetta errori di quantizzazione, che
   avviene in fase di build e dipende dall'hardware.

```json
"validation": {
  "max_abs_diff_vs_fp32": 0.0031,
  "map50_delta": -0.004,
  "status": "ok"
}
```

`status: "degraded"` quando si superano le soglie di
`quantization.validation`. Nel report, una tabella degli artefatti con il loro
stato, così un artefatto rotto è visibile a colpo d'occhio invece di
nascondersi dentro un grafico di latenza.

### 7.5 Rilevamento della testa (NMS / end-to-end)

YOLO26 ha un'architettura a doppia testa: una testa one-to-one per inferenza
end-to-end senza NMS, e una one-to-many tradizionale che richiede NMS. Entrambe
vengono allenate.

Il percorso end-to-end viene **disabilitato automaticamente** da alcune
combinazioni, fra cui:

- TensorRT precedente a 8.5.0
- **TensorRT 10.3.0 con quantizzazione a 8 bit su JetPack 6**
- LiteRT con quantizzazione a 8 bit o `w8a16`

In questi casi viene loggato un warning e viene esportata la testa
one-to-many. La cella INT8 + TensorRT + JetPack 6 è esattamente una delle
combinazioni previste dalla matrice.

Conseguenza: **ispezionare l'artefatto esportato** (output del grafo) per
determinare quale testa contiene, invece di assumere quella richiesta.
Registrare `requested_e2e` e `actual_e2e` come campi distinti del risultato e
segnalare il disallineamento nel report.

Per decisione presa, `e2e` **non è un asse della matrice** in questa
iterazione: viene solo rilevato e loggato.

---

## 8. Schema di `results.json`

Versionato fin dall'inizio: quando si aggiunge un campo a metà lavoro, le run
precedenti restano leggibili.

```json
{
  "schema_version": 1,
  "cell_id": "a91c3e7f0b22",
  "status": "ok",
  "timestamp": "2026-09-17T14:32:11+02:00",
  "order_index": 47,

  "axes": {
    "model": "yolo26n",
    "training_key": "a3f2c891",
    "quantization": "int8",
    "backend": "tensorrt",
    "board": "jetson_orin_nx",
    "freq_target": "w15",
    "compute_target": "cpu_2"
  },

  "latency": {
    "mean_ms": 12.4, "median_ms": 12.1,
    "p90_ms": 13.8, "p95_ms": 14.2, "p99_ms": 18.9,
    "throughput_qps": 80.6,
    "iters": 200, "warmup_ms": 2000,
    "scope": "end_to_end_with_transfers"
  },

  "accuracy": { "map50": 0.712, "map50_95": 0.441 },

  "energy": { "mean_power_w": 8.3, "energy_j": 2.47, "source": "tegrastats" },

  "runtime_state": {
    "freq_requested_khz": 1497600,
    "freq_actual_khz": 1497600,
    "governor": "performance",
    "cores_online": 4,
    "omp_num_threads": 2,
    "temp_start_c": 42.1,
    "temp_end_c": 51.7,
    "throttled": false,
    "swap_off": true,
    "isolation": "none",
    "post_reboot": false
  },

  "validation": {
    "max_abs_diff_vs_fp32": 0.0031,
    "map50_delta": -0.004,
    "requested_e2e": true,
    "actual_e2e": false,
    "e2e_fallback_reason": "trt 10.3 + int8 + jetpack6",
    "status": "ok"
  },

  "env": {
    "ultralytics": "8.x.y", "torch": "2.x.y",
    "onnxruntime": "1.x.y", "tensorrt": "10.3.0",
    "cuda": "12.x", "jetpack": "6.2",
    "axelera_sdk": null, "metis_dkms": null,
    "git_sha": "abc1234", "python": "3.11.9"
  },

  "config": { "...": "config Hydra risolta completa" }
}
```

Per `status: "skipped"` servono solo `cell_id`, `status`, `reason`, `axes`,
`config`. Per `status: "failed"`, in più `error` e `trace`.

Lo stdout grezzo del tool di misura va salvato accanto, in
`<cell_id>.raw.txt`: quando un parser sbaglia a estrarre una latenza, si vuole
poter rileggere l'output originale senza rilanciare la cella.

---

## 8b. Machine hours — blocco `timing`

Presente in **ogni** artefatto e in ogni cella: `meta.json` dei pesi allenati,
`meta.json` degli export, `results.json` del benchmark.

```json
"timing": {
  "wall_s": 18432.7,
  "compute_s": 18310.2,
  "started_at": "2026-09-17T02:14:03+02:00",
  "ended_at": "2026-09-17T07:21:15+02:00",
  "device": "rtx6000_ada",
  "phase_s": {"setup_s": 42.1, "build_s": 0, "calibration_s": 0,
              "compute_s": 18310.2, "teardown_s": 80.4}
}
```

Regole:

- `wall_s` include attese (cooldown termico, reboot, stabilizzazione
  nvpmodel), `compute_s` no. Servono entrambi: la sola wall gonfia il totale,
  il solo compute sottostima l'occupazione della macchina.
- `device` identifica **dove** è stato speso il tempo, perché un'ora di
  RTX 6000 Ada e un'ora di Pi 5 non sono equivalenti in tesi.
- Il timing va scritto anche su `status: "failed"`: una cella che crasha dopo
  venti minuti di build TensorRT ha comunque occupato la macchina.
- Training: aggiungere `epochs_completed` accanto al tempo. Con `patience`
  attiva Ultralytics può fermarsi prima, quindi il tempo per epoca è più
  informativo del totale.
- Export: `build_s` e `calibration_s` separati. Su Axelera la quantizzazione
  può andare da secondi a diverse ore secondo la dimensione del modello; in un
  unico numero non è interpretabile.

Aggregazione:

```python
df.groupby(["stage", "device"])["wall_s"].sum() / 3600
```

---

## 9. Logging

Hydra configura già il `logging` standard di Python e scrive un file per cella.
Usare `logging`, mai `print`. Console a INFO, file a DEBUG.

Catturare esplicitamente l'output dei comandi remoti, che non passa dal
logging Python:

```python
r = conn.run(cmd, hide=True, warn=True)
log.debug(f"stdout:\n{r.stdout}")
if r.failed:
    log.error(f"comando fallito ({r.return_code}):\n{r.stderr}")
```

Silenziare i logger di terze parti (`paramiko`, `ultralytics`) a WARNING.

---

## 10. Aggregazione, analisi e report

### 10.1 Aggregazione

`tools/aggregate.py` percorre `results/*.json`, appiattisce `axes`, `latency`,
`accuracy`, `energy`, `runtime_state` e `validation` in colonne, e scrive un
Parquet. Include anche le celle `skipped` e `failed`, con lo stato come
colonna: servono a distinguere una combinazione non supportata da una che è
crashata.

### 10.2 Notebook

`notebooks/01_analysis.ipynb` carica il Parquet e produce i grafici;
`02_figures.ipynb` li esporta. Ogni figura viene salvata **sia in PNG sia in
PDF**: il PDF è vettoriale e in LaTeX resta nitido a qualsiasi zoom.

### 10.3 Struttura del report

```
reports/
├── latest -> 2026-09-17_14-32-11/
└── 2026-09-17_14-32-11/
    ├── report.md          # riferimenti RELATIVI alle immagini
    ├── report.html        # autoconsistente, immagini in base64
    ├── figures/           # .png + .pdf
    ├── tables/            # .csv
    └── data.parquet
```

Riferimenti relativi (`![](figures/x.png)`) e immagini nella stessa cartella:
l'insieme è un'unità autoconsistente, spostabile e comprimibile. L'HTML
autoconsistente si genera con:

```bash
pandoc report.md -o report.html --embed-resources --standalone
```

Il timestamp in formato `%Y-%m-%d_%H-%M-%S` ordina lessicograficamente come
cronologicamente, senza spazi né due punti.

### 10.4 Contenuto generato automaticamente

- configurazione dell'esperimento e ambiente
- numero di celle eseguite, saltate (con motivo) e fallite
- tabella di stato degli artefatti esportati (validazione numerica, e2e
  effettivo)
- tabelle riassuntive per asse
- fatti estratti dai dati: fronte di Pareto latenza/accuratezza, guadagno della
  quantizzazione per backend, celle con throttling rilevato
- sezioni `## Discussione` vuote come segnaposto per l'interpretazione

---

## 11. Comandi

```bash
# esecuzione singola
python run.py stage=benchmark model=yolo26s quantization=int8 \
  backend=tensorrt hardware=jetson_orin freq_target=w15 compute_target=gpu

# override di un singolo campo
python run.py model=yolo26s model.imgsz=1280 train.batch=8

# aggiunta o rimozione di una chiave
python run.py +stage.warmup_runs=20 ~quantization.calibration

# matrice
python run.py -m model=yolo26n,yolo26s quantization=fp32,fp16,int8 \
  backend=onnxruntime,tensorrt

# tutti i profili e tutti i compute di una board
python run.py -m hardware=jetson_orin freq_target=maxn,w30,w15 \
  compute_target=gpu,cpu_1,cpu_2,cpu_4

# compute_target omesso = tutti quelli dichiarati dalla board
python run.py -m hardware=rpi5 freq_target=max,mid,low

# ispezione senza esecuzione
python run.py -m ... --cfg job --resolve
python run.py -m ... +dry_run=true

# resume e ripetizione
python run.py -m ...                       # solo il mancante
python run.py -m ... +retry_failed=true
python run.py -m ... +force=true

# utility
python -m tools.artifacts ls
python -m tools.aggregate
```

Parallelismo: `hydra/launcher=joblib hydra.launcher.n_jobs=N` è utilizzabile
per training ed export, **mai per la misura di latenza**, perché job
concorrenti sulla stessa GPU o sugli stessi core producono numeri
inutilizzabili.

---

## 12. Ordine di implementazione consigliato

1. `run.py` + `conf/` minimale + `stages/train.py`, un solo modello, workstation.
2. `cache.py` con `training_key`, `meta.json`, riuso degli artefatti.
3. `stages/export.py` + `backends/onnxruntime.py`, export ONNX locale.
4. `schema.py`, `run_cell`, resume su `results/`. Da qui in poi lo sweep è
   interrompibile.
5. `validation/compat.py` — celle valide, `status: "skipped"`.
6. `remote/` — Fabric, provisioning, rsync. Prima board: Jetson.
7. `measure/thermal.py` + `tuned()` — tuning temporaneo.
8. `backends/tensorrt.py` — build on target e parsing di `trtexec`.
9. `validation/numerical.py` e `validation/graph.py`.
10. `measure/power.py` — energia su Jetson.
11. Raspberry Pi 5 + Axelera: container, `metis-dkms`, `axdevice`.
12. `tools/aggregate.py`, notebook, generazione del report.
13. Opzionale: W&B, backend aggiuntivi (OpenVINO, ExecuTorch,
    `onnxruntime_py`).

Ogni stadio è utilizzabile da solo. Non serve che la matrice sia completa
perché il tool produca risultati validi.

---

## 13. Interfacce interne

Sono i contratti che tengono insieme i moduli. Rispettarli è ciò che permette
di aggiungere un backend o una board senza toccare lo sweep.

### 13.1 `backends/base.py`

Ogni backend implementa questa interfaccia e nient'altro. Aggiungere un backend
significa: un file qui, più un YAML in `conf/backend/`.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LatencyResult:
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_qps: float
    iters: int
    warmup_ms: int
    scope: str                  # "end_to_end_with_transfers" | "compute_only"
    raw_stdout: str             # salvato in <cell_id>.raw.txt

class Backend(ABC):
    name: str
    export_format: str
    builds_on_target: bool

    @abstractmethod
    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        """Produce l'artefatto. Se builds_on_target, conn è la board;
        altrimenti è LocalConnection. Idempotente: se dst esiste ed è
        valido, ritorna senza rifare."""

    @abstractmethod
    def build_cmd(self, cfg, artifact: Path) -> str:
        """Command line del tool nativo di misura, già sostituita."""

    @abstractmethod
    def parse(self, stdout: str) -> LatencyResult:
        """Estrae le metriche. Solleva ParseError se l'output non è
        riconoscibile: mai restituire zeri o valori inventati."""

    @abstractmethod
    def inspect(self, artifact: Path) -> dict:
        """Ispeziona l'artefatto: {'actual_e2e': bool, 'precision': str,
        'input_shape': tuple, 'output_names': list}."""

    def validate(self, conn, cfg, artifact: Path, ref) -> dict:
        """Confronto numerico vs FP32 su batch fisso.
        Ritorna il blocco 'validation' dello schema."""
```

Regola sui parser: non devono mai inventare. Se `trtexec` cambia formato di
output fra versioni e il parsing fallisce, la cella deve risultare `failed` con
il raw stdout salvato, non `ok` con numeri a caso.

### 13.2 `remote/connection.py`

`LocalConnection` deve esporre la stessa superficie di `fabric.Connection`
usata dal resto del codice, così `execute_benchmark` non ha rami.

```python
class LocalConnection:
    """Stessa interfaccia di fabric.Connection, eseguita in locale."""
    def run(self, cmd, hide=False, warn=False, env=None, pty=False) -> Result: ...
    def put(self, local, remote): ...    # copia su filesystem
    def get(self, remote, local): ...
    def close(self): ...
    is_local = True
```

`Result` espone almeno `stdout`, `stderr`, `return_code`, `ok`, `failed`.

### 13.3 `measure/` — astrazione board

Due implementazioni dietro un'interfaccia comune, selezionate da
`cfg.hardware.board`.

```python
class BoardController(ABC):
    @abstractmethod
    def read_state(self, conn) -> dict: ...
    @abstractmethod
    def apply_state(self, conn, cfg) -> None: ...
    @abstractmethod
    def restore_state(self, conn, saved: dict) -> None: ...
    @abstractmethod
    def read_temp(self, conn) -> float: ...
    @abstractmethod
    def read_throttle(self, conn) -> bool: ...
    @abstractmethod
    def online_cores(self, conn) -> int: ...
    def read_power(self, conn) -> float | None:
        return None              # default: sensori assenti
```

`JetsonController` usa `nvpmodel`, `jetson_clocks`, `tegrastats`.
`RPi5Controller` usa cpufreq via sysfs e `vcgencmd`.
`X86Controller` gestisce solo il governor.

### 13.4 `env.py`

```python
def collect_versions() -> dict:
    """Popola il blocco 'env' dello schema. Ogni campo non determinabile
    va a None, mai omesso: la presenza della chiave documenta il tentativo."""
```

Fonti: `importlib.metadata` per i pacchetti Python, `trtexec --version` o
`tensorrt.__version__` per TensorRT, `/etc/nv_tegra_release` per JetPack,
`nvcc --version` o `nvidia-smi` per CUDA, `axdevice` e `dpkg -l metis-dkms` per
Axelera, `git rev-parse --short HEAD` per il commit.

Su celle remote le versioni vanno raccolte **sulla board**, non sulla
workstation.

---

## 14. Requirements

Un file per architettura e piattaforma. Le versioni marcate `# VERIFICARE`
dipendono da JetPack e SDK installati e vanno fissate al primo provisioning
riuscito, non indovinate.

### 14.1 `scripts/requirements/x86_64.txt`

```
# core
ultralytics>=8.3
torch>=2.8,<2.13
onnx>=1.16
onnxsim
onnxruntime-gpu          # CPU-only: onnxruntime

# configurazione e orchestrazione
hydra-core>=1.3
omegaconf>=2.3
fabric>=3.2

# dati e analisi
pandas>=2.2
pyarrow>=16
numpy<2.3
matplotlib>=3.8
seaborn

# report
markdown
jinja2

# opzionali
hydra-joblib-launcher
wandb
```

`pandoc` è una dipendenza di sistema, non pip: `sudo apt install pandoc`.

### 14.2 `scripts/requirements/aarch64-jp62.txt`

```
# NON installare torch, torchvision o onnxruntime-gpu da PyPI su Jetson:
# servono i wheel NVIDIA per la specifica versione di JetPack.
# Lo script di provisioning li scarica dall'indice NVIDIA.

ultralytics>=8.3
hydra-core>=1.3
omegaconf>=2.3
numpy<2.3
pyyaml
```

`torch`, `torchvision`, `onnxruntime-gpu` e `tensorrt` vanno installati da
`jetson_jp62.sh`. TensorRT è preinstallato con JetPack: usare quello, non
installarlo da pip.

### 14.3 `scripts/requirements/aarch64-rpi5.txt`

```
# host (fuori dal container): solo utility
# il grosso vive nel container Ubuntu 22.04

ultralytics>=8.3
hydra-core>=1.3
omegaconf>=2.3
numpy<2.3
onnxruntime               # build CPU aarch64, disponibile da PyPI
```

### 14.4 `scripts/requirements/axelera-container.txt`

```
# dentro il container Ubuntu 22.04
ultralytics>=8.3
torch>=2.8,<2.13
axelera-devkit==1.8.0     # VERIFICARE: deve corrispondere a metis-dkms
axelera-rt==1.8.0
```

Indice extra richiesto per i pacchetti Axelera:
`--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple`

Vincoli di piattaforma dichiarati da Axelera: Linux (Ubuntu 22.04 o 24.04),
Python 3.10–3.13, `torch>=2.8,<2.13`, Voyager SDK 1.8.0, `metis-dkms` ≥ 1.6.2,
`libgl1` come dipendenza di sistema.

---

## 15. Script

### 15.1 `scripts/provision/jetson_jp62.sh`

Passi, in ordine, tutti idempotenti:

1. Verifica `/etc/nv_tegra_release` contro il campo `jetpack` del config.
   **Fallire subito** in caso di mismatch.
2. Crea il venv in `workdir/.venv` con `--system-site-packages` (serve per
   vedere il TensorRT di sistema).
3. Installa i wheel NVIDIA di `torch` e `torchvision` per quella JetPack.
4. Installa `onnxruntime-gpu` dal build NVIDIA per Jetson.
5. Installa i requirements generici.
6. Verifica: `python -c "import torch, tensorrt; print(torch.cuda.is_available())"`.
7. Verifica `trtexec` nel PATH (di norma `/usr/src/tensorrt/bin/trtexec`).
8. Scrive `.env_hash`.

### 15.2 `scripts/provision/rpi5_axelera.sh`

1. Chiave GPG e repository apt di Axelera per la distro corrente.
2. `apt install -y metis-dkms libgl1`, poi `modprobe metis`.
3. `axdevice` per confermare che il device sia visto. **Fallire se no.**
4. Build o pull del container Ubuntu 22.04 (`docker build` da
   `scripts/docker/axelera.Dockerfile`).
5. Dentro il container: `pip install` di `axelera-devkit` e `axelera-rt` alle
   versioni dichiarate nel config, più Ultralytics.
6. Verifica end-to-end: un `yolo export format=axelera` su un modello minuscolo.
7. Venv host separato per le celle CPU-only, che non usano il container.
8. Scrive `.env_hash`.

Il passo 3 è il più importante: kernel driver e SDK si aggiornano
separatamente e nulla verifica la coerenza a install time.

### 15.3 `scripts/prepare_board.sh`

Invocato da `apply_state`. Parametrizzato, idempotente, e **ogni azione ha il
suo inverso** in `restore_board.sh`. Argomenti: governor, frequenza target,
id nvpmodel, flag swap.

Emette su stdout un JSON con lo stato **applicato davvero** (frequenza
riletta, core online, governor effettivo), che finisce in `runtime_state`.

### 15.4 `scripts/docker/axelera.Dockerfile`

Base `ubuntu:22.04`. Installa Python 3.11, `libgl1`, i pacchetti Axelera
dall'indice dedicato, Ultralytics. Il container va avviato con accesso al
device PCIe e ai volumi di lavoro:

```
--device /dev/metis0 -v <workdir>:/bench --network host
```

Verificare il nome effettivo del device con `ls /dev | grep -i metis`.

---

## 16. Struttura dei moduli e responsabilità

| File | Responsabilità | Non deve |
|---|---|---|
| `run.py` | entry point Hydra, dispatch di stadio, resume | contenere logica di misura |
| `cache.py` | chiavi, naming, `meta.json`, `index.json` | conoscere i backend |
| `schema.py` | costruzione e validazione di `results.json` | scrivere su disco |
| `env.py` | raccolta versioni | assumere di essere in locale |
| `stages/train.py` | wrapper Ultralytics | conoscere l'hardware |
| `stages/export.py` | dispatch al backend, cache | implementare export |
| `stages/benchmark.py` | orchestrazione: connessione, tuning, esecuzione, raccolta | cronometrare |
| `backends/*.py` | export, command line, parsing, ispezione | gestire SSH o termico |
| `remote/*.py` | connessione, provisioning, sync | conoscere i backend |
| `measure/*.py` | stato board, termico, energia, mAP, risoluzione di `freq`/`compute` | conoscere i backend |
| `validation/*.py` | compatibilità, check numerico, ispezione grafo | modificare stato |
| `tools/*.py` | CLI fuori dallo sweep | importare Hydra |

La separazione che conta di più: **`benchmark.py` non sa come si misura** (lo
sa il backend) e **il backend non sa dove gira** (lo sa `remote/`).

---

## 17. Utility CLI

### `tools/artifacts.py`

```
python -m tools.artifacts ls [--kind weights|exports] [--json]
python -m tools.artifacts show <slug>
python -m tools.artifacts prune --dry-run
```

`ls` legge `index.json` e stampa una riga per artefatto con i campi che
distinguono le run (seed, epoche, mAP, data). `prune` rimuove artefatti non
referenziati da alcun `results.json`, **sempre** con `--dry-run` come default.

### `tools/aggregate.py`

```
python -m tools.aggregate [--results-dir results] [--out reports/<ts>/data.parquet]
```

Include celle `ok`, `skipped` e `failed`, con `status` come colonna. Avvisa se
trova `schema_version` diversi nello stesso set.

### `tools/report.py`

```
python -m tools.report --run <timestamp> [--no-html]
```

Genera `report.md` dal template Jinja2, poi invoca `pandoc` per l'HTML
autoconsistente. Aggiorna il symlink `reports/latest`.

---

## 18. Test

Non serve copertura estesa, servono i test che intercettano gli errori
silenziosi — quelli che producono numeri plausibili ma sbagliati.

**Unitari, senza hardware:**

- `training_key` è stabile a parità di config e cambia al cambiare di ognuno
  dei tre gruppi che lo compongono.
- `training_key` **non** cambia al variare di backend, hardware, `freq_target`
  o `compute_target`. È il test più importante della suite.
- `cell_key` cambia per ognuno degli assi di `cell_key_fields`.
- Parser: per ogni backend, un file di output reale salvato in
  `tests/fixtures/` e il confronto con i valori attesi. Un output troncato o di
  formato diverso deve sollevare `ParseError`.
- `is_valid`: casi noti (TensorRT su Pi → falso, Axelera su Jetson → falso,
  `cpu_8` + `w15` → falso via `requires_freq`).
- `hardware.compute["cpu_8"]` su `rpi5` solleva `KeyError`: la combinazione non
  deve essere rappresentabile.
- `atomic_write_json`: un file troncato non viene mai lasciato sul disco.
- Logica di resume: `ok` salta, `skipped` salta, `failed` salta senza flag e
  riprova con.

**Di integrazione, con hardware:**

- Smoke test: una cella completa sulla workstation, dal `.pt` al
  `results.json`.
- `tuned()` ripristina lo stato anche quando il corpo solleva.
- `set_profile` con `allow_reboot=false` su una transizione che richiede
  riavvio solleva `ProfileChangeRequiresReboot` e non riavvia nulla.
- `reboot_and_wait` scade con `BoardUnreachable` se la board non torna.
- Provisioning eseguito due volte di fila: la seconda non fa nulla.

**Fixture utili:** un modello minuscolo (`yolo26n` a `imgsz=160`, 2 epoche) e
un sottoinsieme di AOD4 da qualche decina di immagini, così i test di
integrazione durano secondi.

---

## 19. Errori da non commettere

1. Mettere backend, hardware, `freq_target` o `compute_target` dentro
   `training_key`: riallenerebbe il modello per ogni cella.
2. Usare `time.perf_counter()` attorno a una chiamata Python come misura di
   latenza.
3. Lasciare il governor dinamico durante la misura.
4. Assumere il numero di core online dal config invece di leggerlo a runtime.
4b. Cambiare `nvpmodel` per singola cella: alcune transizioni richiedono
    reboot. Raggruppare lo sweep per profilo.
4c. Riusare la connessione SSH dopo un riavvio senza aver ripulito il
    `ControlPath`, o riconnettersi senza grace period.
5. Confrontare `trtexec` e `perf_test` senza allineare il perimetro dei
   trasferimenti host↔device.
6. Variare la soglia di confidence fra celle di latenza.
7. Alzare la soglia di confidence per il calcolo della mAP.
8. Variare il calibration set fra artefatti quantizzati.
9. Compilare un engine TensorRT o un modello Axelera sulla workstation.
10. Assumere che l'artefatto esportato contenga la testa richiesta senza
    ispezionarlo.
11. Ammettere al benchmark un artefatto quantizzato non validato numericamente.
12. Scrivere `results.json` solo in caso di successo, o ometterne il blocco
    `timing`: le celle fallite fanno parte delle machine hours.
13. Scrivere `results.json` senza rename atomico.
14. Applicare modifiche di sistema persistenti (`config.txt`, parametri
    kernel, unit systemd).
15. Lasciare un segreto in un file di `conf/`: finirebbe duplicato in ogni
    cartella di output di Hydra.

---

## 20. Scostamenti dell'implementazione rispetto a questa specifica

Dove il codice si discosta da quanto scritto sopra, il motivo è qui. Ogni voce
è una decisione presa in fase di implementazione, non una dimenticanza.

**Flag da riga di comando (`§4.4`, `§11`).** I flag (`dry_run`, `force`,
`retry_failed`, `allow_reboot`, …) hanno un default in `conf/config.yaml`, come
prescrive `§2.1`. Per Hydra `+chiave=valore` significa "aggiungi una chiave che
non c'è" e fallisce proprio perché la chiave c'è. `run.py` riscrive `+flag=` in
`++flag=` prima di passare la riga a Hydra: entrambe le forme funzionano e i
comandi documentati restano validi come sono scritti.

**`prepare_board.sh` (`§15.3`).** Il tuning lo applicano i controller Python in
`src/measure/`, non lo script: sono gli stessi oggetti che leggono lo stato,
lo ripristinano e gestiscono il riavvio, e tenere applicazione e ripristino
nello stesso posto è ciò che rende il `finally` di `tuned()` affidabile. Le
garanzie di `§15.3` restano: ogni azione ha il suo inverso, e `apply_state`
ritorna lo stato riletto (frequenza effettiva, core online, governor) che
finisce in `runtime_state`. `prepare_board.sh` e `restore_board.sh` esistono
per l'uso manuale e per il guard degli sweep non presidiati di `§6.1`.

**INT8 su TensorRT.** L'engine viene costruito da un ONNX già QDQ (precisione
esplicita) invece che passando `--int8` a `trtexec` su un ONNX float. Con la
seconda strada `trtexec` calibra da solo, con dati casuali, e il calibration
set non sarebbe più quello degli altri artefatti INT8 della matrice — che è
esattamente l'errore n. 8 di `§19`.

**`Backend.prepare()`.** Aggiunto all'interfaccia di `§13.1`: i backend che
buildano on-target dichiarano quale artefatto *portabile* va sincronizzato
(TensorRT vuole l'ONNX, Axelera il `.pt`). Senza, lo stadio di export dovrebbe
sapere che TensorRT parte da ONNX, cioè conoscere i backend.

**Espansione di `compute_target: null`.** Avviene dentro lo stadio benchmark e
produce una cella per target, come da `§2.3`. Hydra non può espanderla in
multirun, perché l'elenco dei target dipende dal file dell'hardware scelto.

**Percentili obbligatori.** `validate_record` esige media e mediana; i
percentili alti sono richiesti ma non bloccanti, perché `benchmark_app` di
OpenVINO non li riporta. Quando mancano viene loggato un warning e le colonne
restano vuote: preferibile a ricostruirli da media e deviazione standard.

**Moduli in più.** `src/jsonio.py` (scrittura atomica, usata da tutti gli
stadi), `src/timing.py` (blocco `timing` di `§8b`), `scripts/remote/` (gli
helper Python che devono girare *dove vive l'artefatto*: mAP e confronto
numerico contro FP32).
