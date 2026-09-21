# yolo-bench

Misura latenza, accuratezza ed energia di YOLO26 su hardware edge, in modo
ripetibile e senza doverci stare davanti.

Il problema che risolve è banale da descrivere e noioso da fare a mano: sei
variabili (modello, quantizzazione, backend, board, profilo di potenza,
target di calcolo) che si incrociano in qualche centinaio di misure, ognuna
delle quali va fatta partire, tenuta pulita e annotata. Questo tool le fa
partire, le annota tutte allo stesso modo e alla fine ne tira fuori un
dataframe, dei grafici e un report.

Dataset: **AOD4**, quattro classi (airplane, bird, drone, helicopter).

> Niente a che vedere con la pipeline di compressione PLiNIO. Qui si usano
> solo le API standard di Ultralytics e la quantizzazione post-training nativa
> di ogni backend. Il confronto fra le due pipeline è un altro lavoro.

## Le sei variabili

| | valori |
|---|---|
| `model` | yolo26n, yolo26s, yolo26m, varianti custom |
| `quantization` | fp32, fp16, int8 (solo PTQ) |
| `backend` | ONNX Runtime, TensorRT, OpenVINO, ExecuTorch, Axelera |
| `hardware` | workstation x86, Jetson Orin, Raspberry Pi 5 |
| `freq_target` | profili nvpmodel su Jetson, clock fisso sul Pi |
| `compute_target` | GPU, CPU a 1/2/N core, acceleratore Axelera |

Le ultime due non sono assi indipendenti: ogni board dichiara nel proprio file
quali profili e quali target di calcolo ha. `cpu_8` su un Raspberry Pi non è
una casella da scartare, è una cosa che non si può proprio scrivere, e infatti
solleva un errore prima ancora di partire.

## Come è fatto

Quattro idee, tutte per lo stesso motivo: non ritrovarsi con numeri sbagliati
che sembrano giusti.

**La configurazione è tutta in YAML.** Un gruppo [Hydra](https://hydra.cc) per
asse. Aggiungere un modello, un backend o una board vuol dire aggiungere un
file, mai toccare il codice.

**A cronometrare ci pensano i tool dei vendor.** `trtexec`,
`onnxruntime_perf_test`, `benchmark_app` sono eseguibili C++ scritti da chi ha
fatto il runtime. Il tool compone la riga di comando, la lancia e legge
l'output. Non c'è un solo `time.perf_counter()` attorno a una chiamata Python,
perché quell'overhead è costante e su un modello nano quantizzato su CPU ARM
finirebbe per nascondere proprio lo speedup che stai misurando.

**Quello che si tocca si rimette a posto.** Governor, frequenze, profili di
potenza e swap vengono cambiati per la singola cella e ripristinati subito
dopo, anche se la misura esplode a metà.

**Si può interrompere.** Ogni cella ha un id che dipende dal suo contenuto; se
fermi lo sweep e lo rilanci, riprende da dove era.

## Dove sta cosa

```
conf/        un gruppo per asse: stage, model, train, dataset, quantization,
             backend, hardware, eval, logging
src/
  stages/    train, export, benchmark, quantize
  backends/  un adapter per backend: come esportare, come misurare, come
             leggere l'output
  remote/    SSH, provisioning, rsync
  measure/   stato della board, termico, energia, mAP
  validation/celle valide, confronto numerico, ispezione del grafo
  cache.py   le chiavi di cache e il naming degli artefatti
  schema.py  lo schema (versionato) di results.json
scripts/     provisioning per board, tuning, requirements, Dockerfile
tools/       aggregazione, listing artefatti, report
notebooks/   analisi e figure
run.py       unico entry point
```

`artifacts/` tiene i pesi e i modelli esportati, `results/` un JSON per cella,
`reports/` i report generati.

## Prepararsi

Sulla workstation:

```bash
git clone <repo> && cd yolo-bench
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements/x86_64.txt
sudo apt install pandoc          # serve solo per l'HTML del report
```

Le board si raggiungono tramite un alias in `~/.ssh/config`. Nel repo non c'è
nessuna credenziale, e non deve entrarcene nessuna: Hydra copia la config
risolta in ogni cartella di output, quindi un segreto nei config si
ritroverebbe duplicato in centinaia di posti.

```
Host jetson-orin
    HostName <ip>
    User <user>
    IdentityFile ~/.ssh/id_ed25519_bench
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Serve anche `sudo` senza password sulle board, limitato ai comandi di tuning
(`nvpmodel`, `jetson_clocks`, scrittura su `cpufreq`). Senza, lo sweep si
pianta a ogni cambio di profilo aspettando una password che nessuno digiterà.

Il dataset sta sulla workstation e viene copiato sulle board da solo, una
volta, al primo sweep che lo richiede.

### Provisioning

```bash
python run.py stage=provision hardware=jetson_orin
python run.py stage=provision hardware=rpi5
```

È idempotente: se l'ambiente c'è già e va bene, non fa niente.

Sul Raspberry Pi costruisce anche un container Ubuntu 22.04 per il Voyager
SDK, perché Raspberry Pi OS non è una piattaforma che Axelera supporta. Il
driver `metis-dkms` invece resta sull'host, che è un modulo kernel.

## Lanciare

Gli stadi vanno in ordine e ognuno salta quello che trova già in cache.

```bash
# 1. training, una volta per modello, sulla workstation
python run.py -m stage=train model=yolo26n,yolo26s,yolo26m

# 2. export: l'ONNX si fa qui, gli engine TensorRT e i modelli Axelera
#    si compilano sulla board di destinazione
python run.py -m stage=export \
  model=glob\(*\) quantization=fp32,fp16,int8 backend=onnxruntime,tensorrt

# 3. benchmark, la matrice vera
python run.py -m stage=benchmark \
  model=glob\(*\) quantization=fp32,int8 backend=tensorrt \
  hardware=jetson_orin freq_target=maxn,w15 compute_target=gpu,cpu_2,cpu_4
```

Se ometti `compute_target` li prende tutti, quelli che la board dichiara:

```bash
python run.py -m stage=benchmark hardware=rpi5 freq_target=max,mid,low
```

`freq_target` invece va sempre indicato per le board. Il default che trovi in
`config.yaml` vale per la workstation, e sulle altre non esiste: se lo
dimentichi te lo dice subito, con l'elenco di quelli buoni.

Prima di lanciare qualcosa di grosso, conviene guardarlo:

```bash
python run.py -m ... --cfg job --resolve     # la config risolta, senza eseguire
python run.py -m ... +dry_run=true           # quali celle verrebbero fatte
```

Il dry-run stampa una riga per cella con l'esito previsto, incluse quelle che
verrebbero saltate e perché.

### Riprendere, ripetere

```bash
python run.py -m ...                      # rifà solo quello che manca
python run.py -m ... +retry_failed=true   # riprova anche le celle fallite
python run.py -m ... +force=true          # rifà tutto
```

(Hydra vorrebbe `retry_failed=true` senza il `+`, perché quelle chiavi hanno
già un default. `run.py` accetta entrambe le forme, così i comandi qui sopra
funzionano come sono scritti.)

Un avvertimento sul parallelismo: `hydra/launcher=joblib` va benissimo per
training ed export, ma **mai** per la misura di latenza. Due job che si
contendono la stessa GPU o gli stessi core producono numeri inservibili.

### Guardare i risultati

```bash
python -m tools.aggregate --out data.parquet
jupyter lab notebooks/01_analysis.ipynb
python -m tools.report
```

Il report finisce in una cartella datata sotto `reports/`, con dentro il
markdown, l'HTML autoconsistente, le figure in PNG e PDF, le tabelle in CSV e
il dataframe. `reports/latest` punta sempre all'ultima.

## Cosa viene registrato

**Latenza**: media, mediana, p90, p95, p99, throughput. Sempre `batch=1`,
stesso numero di iterazioni, stesso input, misurata dal tool nativo del
backend.

**Accuratezza**: mAP@50 e mAP@50-95, calcolate una volta per artefatto
esportato. Non dipendono dal profilo di potenza né da quanti core usi, quindi
rifarle per ogni riga della matrice sarebbe solo tempo buttato.

**Energia**: potenza media e integrale sulla durata della misura, dove ci sono
i sensori (su Jetson via `tegrastats`; sul Pi non c'è niente di accessibile).

**Contesto**: temperatura prima e dopo, throttling, frequenza richiesta e
frequenza effettiva, core online, versioni di tutti i runtime, commit del
tool. Sembra pedante finché non ti serve difendere un numero.

**Tempo macchina**: quanto è durato ogni stadio e su quale macchina, anche per
le celle fallite. Una cella che crasha dopo venti minuti di build TensorRT ha
occupato la macchina uguale.

**Validità**: ogni artefatto esportato viene confrontato numericamente con
l'FP32 prima di essere ammesso al benchmark. Un modello INT8 calibrato male è
velocissimo e predice rumore, e senza questo controllo comparirebbe in tabella
come il risultato migliore.

## Cose che è meglio sapere

**YOLO26 ha due teste.** Una one-to-one per l'inferenza end-to-end senza NMS,
e una one-to-many tradizionale che l'NMS ce l'ha. Certe combinazioni di
runtime e quantizzazione (fra cui INT8 su TensorRT con JetPack 6) disabilitano
il percorso end-to-end da sole, con un warning e un fallback silenzioso. Il
tool guarda dentro l'artefatto e registra quale testa c'è davvero. Se non lo
facesse, attribuiresti alla quantizzazione una differenza di latenza che viene
invece dal post-processing.

**Gli engine non si spostano.** Un engine TensorRT è legato a GPU,
architettura e versione della libreria; un modello Axelera richiede la scheda
Metis fisicamente presente in fase di export e cambia formato con l'SDK. Per
questo si compilano sulla board, e la workstation produce solo `.pt` e
`.onnx`.

**La quantizzazione Axelera non è un asse.** Il Voyager SDK quantizza e
compila in INT8 per conto suo, quindi le celle Axelera non si confrontano riga
per riga con l'INT8 di TensorRT: lo schema di quantizzazione è diverso. Una
cella `axelera + fp32` viene saltata, perché misurerebbe un modello INT8 con
l'etichetta sbagliata.

**MAXN non è "il profilo veloce".** NVIDIA lo descrive come modalità non
vincolata e sperimentale: il throttling hardware interviene quando la potenza
del modulo supera il budget, e i carichi pesanti prolungati lì dentro sono
sconsigliati. Le celle MAXN vanno lette insieme al campo `throttled`.

**Certi cambi di profilo vogliono il riavvio.** Per questo lo sweep va
raggruppato per profilo, e il tool si ferma se il profilo attivo non è quello
richiesto invece di andare avanti con risultati sbagliati in silenzio. Il
riavvio automatico esiste ma è spento di default (`+allow_reboot=true`), così
uno sweep notturno non può riavviarti una board mentre non guardi.

**Il caldo sporca le misure.** Uno sweep lungo scalda le board e le ultime
celle vengono sistematicamente più lente delle prime. Il tool aspetta che si
raffreddi prima di ogni misura e registra l'indice di esecuzione, così se
salta fuori una correlazione fra ordine e latenza la vedi in analisi.

## Test

```bash
pytest
```

Non c'è copertura estesa, ci sono i test che prendono gli errori silenziosi.
Il più importante è quello che verifica che backend, hardware e profilo di
potenza **non** finiscano nella chiave di training: se ci finissero, lo stesso
modello verrebbe riallenato per ogni cella e te ne accorgeresti solo a GPU già
sprecata.

I parser hanno per fixture output veri dei tool, salvati in `tests/fixtures/`,
e per ognuno c'è anche il test contrario: davanti a un output troncato devono
sollevare, non restituire zeri.

## Limiti attuali

- Il training gira su una sola workstation, non è distribuito.
- L'isolamento dei core (`isolcpus`, cpuset) non c'è: richiede un boot
  modificato, quindi una modifica persistente, che è fuori dalle regole che si
  è dato il tool.
- Solo PTQ. La quantization-aware training è un altro discorso.
- Il mirror su Weights & Biases è opzionale e resta un mirror: la fonte di
  verità è il JSON locale.
- Niente di tutto questo è mai girato contro hardware vero. I parser sono
  testati su output reali salvati come fixture, ma la prima cella end-to-end
  su workstation, e poi su Jetson, resta da fare.
