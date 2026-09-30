# Tool da riga di comando

I moduli in `tools/` si usano **fuori dallo sweep**, su `results/` e
`artifacts/` gia' scritti: non importano Hydra, non misurano niente e non
toccano le board. Funzionano anche su risultati copiati da un'altra macchina.

Si lanciano come moduli, **dalla root del progetto**:

```bash
python -m tools.<nome> [opzioni]
```

Da un'altra cartella (per esempio `notebooks/`) Python non trova il pacchetto
`tools` (`No module named 'tools'`), e i path di default (`results`,
`artifacts`, `reports`) sono relativi alla cartella corrente.

| tool | legge | scrive | a cosa serve |
|---|---|---|---|
| [`aggregate`](#toolsaggregate) | `results/*.json` | `data.parquet` | una tabella con tutte le celle |
| [`report`](#toolsreport) | `results/*.json` | `reports/<data>/` | figure, tabelle e report |
| [`artifacts`](#toolsartifacts) | `artifacts/`, `results/` | niente (tranne `prune --apply`) | vedere e ripulire la cache di pesi ed export |

---

## `tools.aggregate`

Raccoglie tutti i risultati in un'unica tabella, una riga per cella. E' il
file che leggono `notebooks/01_analysis.ipynb` e `02_figures`, e che produce
la cella "Aggregazione" di `00_sweep`.

```bash
python -m tools.aggregate [--results-dir results] [--out data.parquet] [--csv]
```

| opzione | default | |
|---|---|---|
| `--results-dir` | `results` | cartella con i JSON delle celle |
| `--out` | `data.parquet` (nella cartella corrente) | file Parquet di uscita |
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
45 celle -> data.parquet
  failed: 22, skipped: 15, ok: 7
```

Esce con codice 1 se `results/` e' vuota, 2 se manca pandas.

---

## `tools.report`

Genera un report completo in una cartella datata, autoconsistente e
spostabile (i riferimenti alle immagini sono relativi).

```bash
python -m tools.report [--results results] [--reports reports] [--run <timestamp>] [--no-html]
```

| opzione | default | |
|---|---|---|
| `--results` | `results` | cartella con i JSON delle celle |
| `--reports` | `reports` | dove creare la cartella del report |
| `--run` | adesso, `%Y-%m-%d_%H-%M-%S` | nome della cartella; con un nome esistente la riscrive |
| `--no-html` | spento | salta la conversione in HTML |

Non serve lanciare prima `aggregate`: rilegge `results/` da solo, con la
stessa funzione.

**Output**, in `reports/<timestamp>/`:

```
reports/2026-09-28_15-00-00/
├── data.parquet          lo stesso dataframe di tools.aggregate
├── report.md             dal template templates/report.md.j2
├── report.html           autoconsistente, immagini incluse (serve pandoc)
├── figures/              ogni figura in PNG (per il markdown) e PDF (per LaTeX)
│   ├── latency_by_backend.*   latenza mediana per backend e quantizzazione
│   ├── pareto.*               latenza / mAP@50 con il fronte di Pareto
│   ├── pareto_map50_95.*      latenza / mAP@50-95 con il fronte di Pareto
│   ├── map50_95_by_quantization.*  mAP@50-95 per modello e backend, una barra per precisione
│   └── order_drift.*          latenza rispetto all'ordine di esecuzione
└── tables/
    ├── celle_ok.csv           le celle misurate, con latenza, mAP@50, mAP@50-95, energia
    ├── stato_celle.csv        quante celle per stato
    ├── celle_saltate.csv      le celle skipped, raggruppate per motivo
    ├── artefatti.csv          esito della validazione numerica degli export
    ├── machine_hours.csv      ore macchina per dispositivo
    └── celle_throttled.csv    celle in cui la board ha fatto throttling
```

`reports/latest` e' un link simbolico all'ultima cartella generata.

Le figure e `celle_ok` usano solo le celle `ok`; una figura o una tabella
senza dati non viene prodotta. `order_drift` serve a vedere se il termico ha
inquinato la misura: se la latenza cresce con l'ordine di esecuzione, le celle
in fondo allo sweep sono state misurate su una macchina piu' calda.

Nel report ci sono anche alcuni numeri riassuntivi: celle per stato, la cella
piu' veloce, lo speedup di FP16 e INT8 rispetto a FP32 per ogni backend, e
quante celle hanno fatto throttling, hanno una validazione `degraded` o sono
ricadute sulla testa one-to-many.

L'HTML richiede `pandoc` (`sudo apt install pandoc`); se manca, il report
viene generato lo stesso e l'HTML viene saltato con un avviso.

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
rm -rf artifacts/* results/* reports/* multirun/ outputs/ \
       data.parquet benchmark_report.csv yolo26*.pt
```

Lascia `.venv/`, `data/` e i `.gitkeep`. **Non usare `git clean -fdX`**:
cancella tutto quello che e' in `.gitignore`, compresi `.venv/` e il dataset
in `data/`.
