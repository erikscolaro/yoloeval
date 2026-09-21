# yolo-bench

Strumento automatizzato per l'addestramento, l'export e il benchmark sistematico
di modelli YOLO26 su hardware edge eterogeneo.

> Questo progetto è **separato e indipendente** dalla pipeline di compressione
> PLiNIO. Qui si usano esclusivamente le API standard di Ultralytics e la
> quantizzazione post-training nativa di ciascun backend. Nessun confronto con
> la pipeline custom è in scope.

---

## 1. Obiettivo

Produrre, in modo riproducibile e non presidiato, una matrice di misure di
**latenza**, **accuratezza** ed **energia** per la famiglia YOLO26, variando in
modo controllato sei dimensioni:

| Asse | Valori tipici |
|---|---|
| `model` | yolo26n, yolo26s, yolo26m, varianti custom |
| `quantization` | fp32, fp16, int8 (PTQ) |
| `backend` | ONNX Runtime, TensorRT, OpenVINO, ExecuTorch, Axelera |
| `hardware` | workstation x86, Jetson Orin, Raspberry Pi 5 |
| `freq_target` | profili nvpmodel su Jetson, clock fisso su Pi |
| `compute_target` | GPU, CPU 1/2/N core, acceleratore Axelera |

`freq_target` e `compute_target` non sono gruppi separati: ogni board dichiara
nel proprio file quali profili e quali target di calcolo supporta. Una
combinazione inesistente (`cpu_8` su Raspberry Pi) non è rappresentabile, non
è una cella da scartare.

Dataset unico: **AOD4** (4 classi — airplane, bird, drone, helicopter).

Il risultato finale è un dataframe unico, un set di grafici e un report
generato automaticamente, pronti per il capitolo risultati della tesi.

---

## 2. Principi di progetto

1. **Configurazione dichiarativa.** Ogni asse è un gruppo di file YAML gestito
   da [Hydra](https://hydra.cc). Aggiungere un modello, un backend o una board
   significa aggiungere un file, mai modificare il codice.
2. **Tre stadi con cardinalità diverse.** Il training gira una volta per
   modello, l'export una volta per artefatto, il benchmark su tutta la matrice.
   Ogni stadio legge dalla cache del precedente.
3. **La misura la fanno i tool nativi.** `trtexec`, `onnxruntime_perf_test`,
   `benchmark_app` sono eseguibili C++ scritti dai vendor. Il tool compone la
   command line, la lancia e ne parsa l'output: non implementa timer propri,
   così l'overhead del binding Python non entra nella misura.
4. **Modifiche di sistema sempre temporanee.** Governor, frequenze, profili di
   potenza e swap vengono applicati per la singola cella e ripristinati subito
   dopo, anche in caso di errore.
5. **Idempotenza e resume.** Ogni cella è identificata da un hash del suo
   contenuto. Interrompere e rilanciare riprende dal punto in cui si era
   fermati.

---

## 3. Struttura del repository

```
yolo-bench/
├── conf/                    # configurazione Hydra — un gruppo per asse
│   ├── config.yaml          #   defaults list e impostazioni globali
│   ├── stage/               #   train | export | benchmark | provision
│   ├── model/               #   yolo26n, yolo26s, yolo26m, custom_pruned
│   ├── train/               #   iperparametri (coincidono con quelli Ultralytics)
│   ├── dataset/             #   aod4: path, data yaml, checksum
│   ├── quantization/        #   fp32, fp16, int8 + parametri di calibrazione
│   ├── backend/             #   per backend: come esportare, come misurare
│   ├── hardware/            #   un file per board, con dentro freq e compute
│   ├── eval/                #   soglie conf/iou per mAP e per latenza
│   └── logging/             #   local | wandb (opzionale)
│
├── src/
│   ├── stages/              # train.py, export.py, quantize.py, benchmark.py
│   ├── backends/            # un adapter per backend (export + bench + parse)
│   ├── remote/              # connessione SSH, provisioning, sync dataset
│   ├── measure/             # board, termico, energia, accuratezza, freq/compute
│   ├── validation/          # celle valide, check numerico, ispezione grafo
│   ├── cache.py             # hash di training, naming artefatti, indice
│   ├── schema.py            # schema versionato di results.json
│   ├── timing.py            # blocco timing / machine hours
│   ├── jsonio.py            # scrittura atomica
│   ├── errors.py            # eccezioni tipizzate
│   └── env.py               # raccolta versioni di runtime e commit git
│
├── scripts/
│   ├── provision/           # uno script per board (Jetson, Pi+Axelera, x86)
│   ├── remote/              # helper eseguiti dove vive l'artefatto (mAP, diff)
│   ├── docker/              # immagine Ubuntu 22.04 per il Voyager SDK
│   ├── prepare_board.sh     # governor, frequenze, swap, guard
│   ├── restore_board.sh     # l'inverso di prepare_board.sh
│   └── requirements/        # requirements per arch e versione JetPack
│
├── artifacts/               # cache: pesi allenati e modelli esportati
├── results/                 # un JSON per cella, indicizzato per hash
├── multirun/                # output Hydra: config risolta e log per cella
├── reports/                 # report generati, con figures/ e tables/
├── templates/               # template Jinja2 del report
├── notebooks/               # aggregazione e generazione grafici
├── tests/                   # unitari, con fixture di output reali
├── tools/                   # utility CLI (listing artefatti, aggregazione, report)
└── run.py                   # unico entry point
```

---

## 4. Come riprodurre i risultati

### 4.1 Prerequisiti

**Workstation (x86_64, Ubuntu):**

```bash
git clone <repo> && cd yolo-bench
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements/x86_64.txt
```

**Accesso alle board.** Ogni board è raggiungibile via un alias definito in
`~/.ssh/config`. Nessuna credenziale è contenuta nel repository.

```
Host jetson-orin
    HostName <ip>
    User <user>
    IdentityFile ~/.ssh/id_ed25519_bench
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m

Host rpi5
    HostName <ip>
    User <user>
    IdentityFile ~/.ssh/id_ed25519_bench
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Sulle board serve `sudo` senza password limitato ai soli comandi di tuning
(`nvpmodel`, `jetson_clocks`, scrittura su `cpufreq`), altrimenti lo sweep si
blocca a ogni cambio di profilo.

**Dataset.** AOD4 risiede sulla workstation. Viene sincronizzato sulle board
una sola volta, in modo incrementale, al primo sweep che lo richiede.

### 4.2 Provisioning delle board

Idempotente: se l'ambiente è già presente e coerente, non fa nulla.

```bash
python run.py stage=provision hardware=jetson_orin
python run.py stage=provision hardware=rpi5
```

Su Raspberry Pi il provisioning costruisce un container Ubuntu 22.04 per il
Voyager SDK, perché Raspberry Pi OS non è una piattaforma supportata da
Axelera. Il kernel driver `metis-dkms` viene installato sull'host.

### 4.3 Esecuzione

Gli stadi si eseguono in ordine. Ognuno salta ciò che è già in cache.

```bash
# 1. training — una volta per modello, sulla workstation
python run.py -m stage=train model=yolo26n,yolo26s,yolo26m

# 2. export — ONNX sulla workstation; engine TensorRT e modelli Axelera
#    vengono compilati direttamente sulla board di destinazione
python run.py -m stage=export \
  model=glob\(*\) quantization=fp32,fp16,int8 backend=onnxruntime,tensorrt

# 3. benchmark — la matrice vera e propria
python run.py -m stage=benchmark \
  model=glob\(*\) quantization=fp32,int8 backend=tensorrt \
  hardware=jetson_orin freq_target=maxn,w15 compute_target=gpu,cpu_2,cpu_4

# omettere compute_target significa "tutti quelli dichiarati dalla board"
python run.py -m stage=benchmark hardware=rpi5 freq_target=max,mid,low
```

Prima di lanciare uno sweep ampio, conviene ispezionarlo:

```bash
python run.py -m ... --cfg job --resolve     # config risolta, nessuna esecuzione
python run.py -m ... +dry_run=true           # matrice espansa e celle valide
```

### 4.4 Ripresa e ripetizione

```bash
python run.py -m ...                      # riprende: esegue solo il mancante
python run.py -m ... +retry_failed=true   # ritenta anche le celle fallite
python run.py -m ... +force=true          # rifà tutto da capo
```

I flag hanno un default in `conf/config.yaml`, quindi per Hydra la forma
corretta sarebbe `retry_failed=true` (senza `+`, che serve ad aggiungere una
chiave che non esiste). `run.py` accetta entrambe le forme e riscrive `+flag=`
in `++flag=` prima di passare la riga a Hydra, così i comandi qui sopra
funzionano come sono scritti.

### 4.5 Analisi e report

```bash
python -m tools.aggregate                 # results/*.json -> data.parquet
jupyter lab notebooks/01_analysis.ipynb
```

Il notebook produce una cartella datata sotto `reports/`, contenente il report
in Markdown e HTML autoconsistente, le figure in PNG e PDF, le tabelle in CSV e
il dataframe aggregato. Un symlink `reports/latest` punta sempre all'ultima.

---

## 5. Cosa viene misurato

**Latenza** — media, mediana, p90, p99, throughput. Misurata dai tool nativi
del backend, con warm-up e numero di iterazioni identici in tutte le celle,
`batch=1` e input fisso.

**Accuratezza** — mAP@50 e mAP@50-95, calcolate una sola volta per artefatto
esportato: non dipendono da profilo di potenza o numero di core.

**Energia** — potenza istantanea e integrale sulla durata della misura, dove i
sensori sono disponibili (Jetson via `tegrastats`).

**Contesto** — temperatura a inizio e fine, stato di throttling, frequenza
richiesta e frequenza effettiva, core online, versioni di tutti i runtime,
commit git del tool. Senza questi campi una misura non è difendibile.

Su Jetson il profilo di potenza determina quanti core sono online (il profilo
15W ne porta online 4 su 8), quindi il conteggio va letto a runtime e non
assunto dal config. Alcune transizioni fra profili richiedono il riavvio del
sistema: lo sweep è perciò raggruppato per profilo, e il tool si ferma se il
profilo attivo non corrisponde a quello richiesto invece di proseguire con
risultati silenziosamente sbagliati. Il riavvio automatico esiste ma è
disattivato per default (`+allow_reboot=true` per abilitarlo), così uno sweep
notturno non può riavviare una board mentre nessuno guarda.

**Tempo macchina** — durata di training, export, quantizzazione e benchmark,
con la macchina su cui è stata spesa, per il calcolo delle machine hours.

**Validità** — ogni artefatto esportato viene confrontato numericamente con il
modello FP32 prima di essere ammesso al benchmark. Un modello quantizzato male
è velocissimo e predice rumore: senza questo controllo comparirebbe in tabella
come il risultato migliore.

---

## 6. Note importanti

**YOLO26 e NMS.** YOLO26 ha un'architettura a doppia testa e supporta
inferenza end-to-end senza NMS. Il percorso end-to-end viene però disabilitato
automaticamente da alcune combinazioni di runtime e quantizzazione — tra cui
INT8 su TensorRT con JetPack 6 — con conseguente fallback sulla testa
one-to-many, che include NMS. Il tool **rileva e registra** quale testa sia
effettivamente presente in ogni artefatto, senza trattarla come asse della
matrice. Ignorare questo fatto significa attribuire alla quantizzazione una
differenza di latenza che dipende invece dal post-processing.

**Gli engine non sono portabili.** Un engine TensorRT è compilato per una
specifica combinazione di GPU, architettura e versione della libreria. Lo
stesso vale per i modelli Axelera, che richiedono la presenza fisica
dell'acceleratore Metis e la cui versione di formato cambia con l'SDK. La
compilazione avviene sempre sulla board di destinazione; la workstation produce
solo artefatti portabili (`.pt`, `.onnx`).

**La quantizzazione Axelera non è un asse.** Il Voyager SDK quantizza e compila
automaticamente in INT8 per l'architettura mixed-precision dell'AIPU. Le celle
Axelera non sono confrontabili riga per riga con l'INT8 di TensorRT, perché lo
schema di quantizzazione è diverso.

**MAXN non è il profilo veloce.** NVIDIA lo descrive come modalità non
vincolata e sperimentale: il throttling hardware interviene quando la potenza
del modulo supera il budget TDP, e i carichi pesanti prolungati in quella
modalità sono sconsigliati. Le celle MAXN vanno lette insieme al campo
`throttled`, non come limite superiore pulito.

**Il termico inquina le misure.** Uno sweep lungo scalda le board, e le ultime
celle risultano sistematicamente più lente delle prime. Il tool attende il
raffreddamento prima di ogni misura e registra l'indice di esecuzione, così una
correlazione tra ordine e latenza è visibile in analisi invece di passare
inosservata.

---

## 7. Test

```bash
pytest                      # unitari, nessun hardware richiesto
pytest tests/test_keys.py   # il test che conta: cosa entra nel training_key
```

Non c'è copertura estesa: ci sono i test che intercettano gli errori
silenziosi, quelli che producono numeri plausibili ma sbagliati. Che
`training_key` non cambi al variare di backend, hardware o profilo di potenza è
il più importante della suite: se cambiasse, lo stesso modello verrebbe
riallenato per ogni cella e la cosa si noterebbe solo a GPU già sprecata.

I parser hanno per fixture output reali dei tool (`tests/fixtures/`), e ogni
parser ha anche il test opposto: su un output troncato deve sollevare
`ParseError`, non restituire zeri.

---

## 8. Limitazioni note

- Il training non è distribuito: gira su una sola workstation.
- L'isolamento dei core (`isolcpus`, cpuset) non è implementato. Interferisce
  con il load balancing dello scheduler nei test multicore e richiede una
  valutazione dedicata.
- La quantization-aware training è fuori scope: solo PTQ.
- Il logging su Weights & Biases è opzionale e agisce da mirror. La fonte di
  verità resta il JSON locale.
