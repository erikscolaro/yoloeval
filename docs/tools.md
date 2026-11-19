# Tool da riga di comando

I moduli in `tools/` si usano **fuori dallo sweep**, su `results/` e
`artifacts/` gia' scritti: non importano Hydra, non misurano niente e non
toccano le board. Funzionano anche su risultati copiati da un'altra macchina.

Si lanciano come moduli, **dalla root del progetto**:

```bash
python -m tools.<nome> [opzioni]
```

Da un'altra cartella Python non trova il pacchetto
`tools` (`No module named 'tools'`), e i path di default (`results`,
`artifacts`, `lab/reports`) sono relativi alla cartella corrente.

| tool | legge | scrive | a cosa serve |
|---|---|---|---|
| [`aggregate`](#toolsaggregate) | `results/*.json` | `lab/reports/data.parquet` | una tabella con tutte le celle |
| `results` | `results/*.json` | niente | misure per esito, fallite e saltate con il motivo, dettaglio di una (`riepilogo`, `falliti`, `saltati`, `cella ID`) |
| [`report`](#toolsreport) | `results/*.json`, `artifacts/weights`, `artifacts/finetune` | `lab/reports/<data>/` | griglia, reti potate, varianti, fallite |
| [`roofline`](#toolsroofline) | `results/*.json`, `results/roofline/*.json` | `lab/reports/roofline/` | efficienza dei modelli e confronto con la baseline |
| [`artifacts`](#toolsartifacts) | `artifacts/`, `results/` | niente (tranne `prune --apply`) | vedere e ripulire la cache di pesi ed export |

Gli script delle figure si lanciano invece come file (`python tools/<nome>.py`), ognuno con l'uso scritto nella
docstring. Senza cartella di uscita scrivono in `docs/figures/<nome>/` (le figure già fatte, con la data nel nome della
cartella, sono lì):

| script | legge | scrive | a cosa serve |
|---|---|---|---|
| `curva_fbeta.py` | pesi `.pt`, dataset in `data/air-detection/all/` | `docs/figures/fbeta/` | curva F-beta per classe in funzione della soglia di confidenza (`--beta`, `--split`) |
| `figure_sintesi.py` | `results/` (con `--risultati`) o dati di esempio | `docs/figures/sintesi/` (A, B, C) | figure di sintesi del blocco 1 per dispositivo (`NEW_MASTERPLAN.md`, sezione 13) |
| `search_channels.py` | `pruned.pt` di una o più cartelle in `artifacts/search/` | un PNG (solo il nome: in `docs/figures/canali/`) | canali tenuti da una ricerca PIT, per layer |
| `canali_famiglie.py` | `pruned.pt` di una ricerca, facoltativo un `best.pt` di confronto | un PNG (solo il nome: in `docs/figures/canali/`) | canali tenuti per famiglia di blocco |
| `figure_griglia.py` | `artifacts/weights/*/run/results.csv` | `docs/figures/griglia/` | mAP della griglia larghezza × profondità, a barre e a linee |
| `figura_pareto.py` | `results/`, `artifacts/weights/` | `docs/figures/pareto/` | fronte di Pareto mAP-latenza della griglia |
| `figure_guadagno_latenza.py` | `artifacts/weights/`, esperimenti B4-R e pavimento | `docs/figures/guadagno_latenza_640/` | mAP per ms e spazio di potatura per architettura |

`order.py` e `plots.py` sono moduli di supporto (ordinamento naturale dei nomi, stile dei grafici).

Le figure di tutti i tool e degli esperimenti usano
`tools/thesis.mplstyle` e `tools/plots.py` (`save`, `color`, `WIDTH`): stesso
stile, stesso colore per la stessa entità, PNG e PDF per LaTeX.

---

## `tools.aggregate`

Raccoglie tutti i risultati in un'unica tabella, una riga per cella, da leggere con pandas. Dal launcher:
utilità, aggrega.

```bash
python -m tools.aggregate [--results-dir results] [--out lab/reports/data.parquet] [--csv]
```

| opzione | default | |
|---|---|---|
| `--results-dir` | `results` | cartella con i JSON delle celle |
| `--out` | `lab/reports/data.parquet` | file Parquet di uscita |
| `--csv` | spento | scrive anche un CSV accanto al Parquet |

**Input.** Solo `results/*.json`. Entrano tutte le celle, anche `failed` e
`skipped`: servono a distinguere una combinazione non supportata da una che e'
crashata. Un JSON non valido viene saltato con un avviso; se nello stesso set
ci sono `schema_version` diversi lo segnala, perche' le run piu' vecchie
potrebbero non avere tutte le colonne.

**Colonne.** Ogni blocco del JSON diventa un gruppo di colonne con un
prefisso; i valori annidati (dict, liste) restano come stringa JSON.

| blocco | prefisso | esempi |
|---|---|---|
| — | — | `cell_id`, `status`, `reason`, `error`, `timestamp`, `order_index`, `schema_version` |
| `axes` | nessuno | `model`, `quantization`, `backend`, `board`, `arch`, `freq_target`, `compute_target`, `training_key` |
| `latency` | `lat_` | `lat_mean_ms`, `lat_median_ms`, `lat_p99_ms`, `lat_throughput_qps` |
| `accuracy` | `acc_` | `acc_map50`, `acc_map50_95`, `acc_precision`, `acc_recall` |
| `energy` | `energy_` | `energy_mean_power_w` |
| `runtime_state` | `rt_` | `rt_throttled`, `rt_temp_end_c` |
| `validation` | `val_` | `val_status`, `val_match_rate`, `val_max_score_diff` |
| `timing` | `time_` | `time_wall_s`, `time_device` |
| `env` | `env_` | versioni di librerie e runtime |
| `config` | `cfg_` | solo `cfg_stage`, `cfg_imgsz`, `cfg_iters`, `cfg_bench_conf` |

Le righe sono ordinate per `board`, `backend`, `quantization`, `model`. A fine
esecuzione stampa quante celle ci sono per stato:

```
45 celle -> lab/reports/data.parquet
  failed: 22, skipped: 15, ok: 7
```

Esce con codice 1 se `results/` e' vuota, 2 se manca pandas.

---

## `tools.report`

Report delle misure sul dispositivo del profilo di deploy, in una cartella datata autoconsistente e spostabile
(immagini e tabelle con percorsi relativi). Si lancia anche dal launcher: `lab/launcher`, utilità, report.

```bash
python -m tools.report [--target rpi5_640] [--deploy 640] [--budget MS] [--run NOME] [--no-html]
```

| opzione | default | |
|---|---|---|
| `--results` | `results` | cartella con i JSON delle misure |
| `--extra` | `experiments/*/*/results` | results/ degli esperimenti: ne entrano solo le reti potate |
| `--reports` | `lab/reports` | dove creare la cartella del report |
| `--target` | `rpi5_640` | profilo di `conf/target/` da cui si legge il budget del modello (risolto da Hydra) |
| `--deploy` | `640` | risoluzione delle misure da confrontare; le altre contano come storiche |
| `--budget` | dal profilo | budget del modello in ms, se si vuole forzarlo |
| `--run` | data e ora, `%Y-%m-%d_%H-%M-%S` | nome della cartella; con un nome esistente la riscrive |
| `--no-html` | spento | salta la conversione in HTML |

Il template è `lab/report.md.j2`. Non serve lanciare prima `aggregate`: rilegge `results/` da solo.

**Contenuto.**

1. **Sintesi**: misure per esito, quante a risoluzioni storiche, migliore rete della griglia e migliore potata
   entro il budget.
2. **Griglia** larghezza × profondità: per ogni architettura la latenza fp16 OpenVINO (mediana delle misure), la
   mAP50-95 del training (miglior epoca sul val, da `artifacts/weights/*/run/results.csv`), la mAP sul dispositivo
   dove c'è, e se sta nel budget. Due figure: latenza per larghezza e profondità, e latenza / mAP50-95 con il fronte
   della griglia, le reti potate e il budget.
3. **Reti potate** (`strategy=pit_latency`, anche dagli esperimenti): latenza, mAP del training (da
   `artifacts/finetune/*/run/results.csv`) e sul dispositivo, la rete della griglia con la mAP più alta fra quelle non
   più lente (`griglia_pari`) e la differenza di mAP (`delta_mAP`, positiva se la potatura rende).
4. **Altre precisioni e backend**: ogni misura diversa da fp16 OpenVINO, con la latenza dello stesso modello e pesi
   in fp16 e il rapporto.
5. **Misure fallite** non rifatte dopo con successo, con la prima riga dell'errore.
6. **Condizioni di misura**: throttling, temperatura massima, ore macchina.

Le mAP confrontate sono tutte del training, sullo stesso val: la mAP sul dispositivo non c'è per ogni rete della
griglia.

**Output**, in `lab/reports/<data_ora>/`:

```
lab/reports/2026-11-19_12-00-00/
├── data.parquet          tutte le misure di results/ (come tools.aggregate)
├── report.md             dal template lab/report.md.j2
├── report.html           autoconsistente, immagini incluse (serve pandoc)
├── figures/              PNG (per il markdown) e PDF (per LaTeX)
│   ├── latenza_griglia.*      latenza per larghezza, una linea per profondità, con il budget
│   └── pareto.*               latenza / mAP50-95: griglia, fronte, reti potate, budget
└── tables/               griglia.csv, potate.csv, varianti.csv, fallite.csv
```

`lab/reports/latest` è un link simbolico all'ultima cartella generata. L'HTML richiede `pandoc`
(`sudo apt install pandoc`); se manca, il report viene generato lo stesso e l'HTML viene saltato con un avviso.

---

## `tools.roofline`

Mette ogni modello misurato sul roofline della sua board: quanto e' lontano
dal tetto raggiungibile, se e' limitato dalla memoria o dal calcolo, e quanto
guadagna un modello potato rispetto alla baseline.

```bash
python -m tools.roofline [--results results] [--out lab/reports/roofline]
```

| opzione | default | |
|---|---|---|
| `--results` | `results` | cartella con i JSON delle celle e la sottocartella `roofline/` |
| `--out` | `lab/reports/roofline` | dove scrivere CSV e grafici |

**Cosa serve prima**, in quest'ordine:

1. `stage=roofline` (workstation) o `stage=roofline_edge` (board) per ogni
   board, compute target e precisione: misura i due tetti, picco di calcolo
   (GMAC/s) e banda di memoria (GB/s), e li salva in
   `results/roofline/<board>_<compute>_<precisione>_<backend>.json`.
2. `stage=export` e `stage=benchmark` dei modelli: l'export calcola MAC e byte
   del modello (blocco `complexity`) e il benchmark li copia nel JSON della
   cella. Gli export fatti prima di questa funzione non li hanno: vanno rifatti
   con `+force_reexport=true`.

```bash
python run.py -m stage=roofline_edge hardware=rpi5 freq_target=max quantization=fp32,int8
python run.py -m stage=export ...
python run.py -m stage=benchmark ...
python -m tools.roofline
```

Senza tetti misurati il tool gira lo stesso, ma senza tetto, efficienza e
memory/compute bound, e lo dice con un avviso. Senza nessuna cella `ok` con la
complessita' del modello esce con errore.

**Cosa calcola**, per ogni cella `ok`:

| colonna | significato |
|---|---|
| `intensity_mac_per_byte` | intensita' aritmetica del modello, MAC / byte |
| `achieved_gmacs` | prestazione ottenuta, MAC / latenza mediana |
| `attainable_gmacs` | tetto raggiungibile: min(picco, banda × intensita') |
| `efficiency` | `achieved_gmacs` / `attainable_gmacs` |
| `bound` | `memory` se l'intensita' e' sotto il ridge point (picco / banda), altrimenti `compute` |
| `speedup_vs_baseline` | latenza della baseline / latenza del modello |
| `mac_ratio_vs_baseline` | MAC del modello / MAC della baseline |
| `map_delta_vs_baseline` | mAP@50-95 del modello meno quella della baseline |

Il confronto con la baseline e' fra celle con stessa board, compute target,
precisione, backend e modello; `label` distingue la baseline dalle strategie
(`pit_duccio N=16`, ...).

I tetti si cercano per board, compute target e precisione. Se manca un roofline
misurato con lo stesso backend, si usa quello di un altro backend sulla stessa
combinazione, e la colonna `roof_backend` dice quale. Se ce ne sono piu' d'uno,
vince il piu' recente.

**Output**, in `lab/reports/roofline/`:

```
lab/reports/roofline/
├── roofline.csv                          una riga per cella, tutte le colonne
└── <board>_<compute>_<precisione>_<backend>.png   un grafico per gruppo
```

Ogni grafico e' in scala log-log: il tetto misurato come linea continua, il
ridge point come linea verticale punteggiata, e un punto per modello con
l'efficienza in legenda. Se in `hardware.peaks.<compute_target>` ci sono i
tetti da datasheet (vedi `docs/config/hardware.yaml`), compaiono tratteggiati.
Le stesse tabelle, una per gruppo, vengono stampate a terminale.

---

## `tools.artifacts`

Ispeziona la cache di `artifacts/`: i pesi addestrati (`weights/`) e i
modelli esportati (`exports/`).

```bash
python -m tools.artifacts [--artifacts artifacts] ls [--kind weights|exports] [--json]
python -m tools.artifacts [--artifacts artifacts] show <slug>
python -m tools.artifacts [--artifacts artifacts] prune [--results results] [--apply]
```

`--artifacts` va **prima** del sottocomando.

### `ls`

Una riga per artefatto: tipo, nome della cartella (lo *slug*) e, per i pesi,
la mAP@50 del training; per gli export, l'esito della validazione e se la
testa e' end-to-end.

```
weights  yolo26n_aod4_e2_6e56352a                  mAP50 0.6854  2026-09-28T11:16:58+02:00
exports  yolo26m_0b8da266_fp32_tensorrt_x86_64     val None e2e None
```

`val None e2e None` vuol dire che la cartella non ha `meta.json`: l'export e'
iniziato ma non e' arrivato in fondo (qui, l'engine TensorRT senza
`trtexec`).

`--kind` limita a un tipo, `--json` stampa i campi completi (modello,
quantizzazione, backend, epoche, seed, tempi) per usarli in uno script.

### `show <slug>`

Stampa il `meta.json` di un artefatto: config completa, versioni, metriche o
validazione, tempi. Lo slug e' il nome della cartella, come lo mostra `ls`.

### `prune`

Cancella gli artefatti che **nessun risultato** in `results/` cita, cioe'
pesi ed export che non hanno prodotto nemmeno una cella.

- **In dry-run per default**: elenca cosa cancellerebbe e basta. Per
  cancellare davvero serve `--apply`.
- Se `results/` e' vuota si ferma, invece di considerare tutto inutilizzato.
- Una cartella senza `meta.json` (per esempio un export TensorRT mai finito)
  non viene toccata: la segnala con `?` e la lascia dov'e'.

Cancellare un export significa ricompilarlo, sulla board se e' un engine
TensorRT: costa molto piu' del disco che libera. Prima di `--apply`, leggi
l'elenco.

---

## Pulizia completa

Non c'e' un tool che cancelli tutto. Per ripartire da zero, dalla root e con
nessuno sweep in esecuzione (`pgrep -f run.py`):

```bash
rm -rf artifacts/* results/* lab/reports/ runs/hydra/ \
       data.parquet benchmark_report.csv yolo26*.pt
```

Lascia `.venv/`, `data/` e i `.gitkeep`. **Non usare `git clean -fdX`**:
cancella tutto quello che e' in `.gitignore`, compresi `.venv/` e il dataset
in `data/`.
