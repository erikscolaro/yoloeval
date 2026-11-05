# LUT di latenza a passo 2 degli strati potabili del nano sul Pi 5

Questo esperimento misura le LUT di latenza delle conv potabili del nano YOLO26 (N, larghezza 0.25,
D126) sul Raspberry Pi 5. Ingresso sempre 640×640, 4 classi. Le LUT servono allo stimatore di latenza
per la potatura per canale (PIT, `~/Desktop/yolopit`): vedi `STIMATORE_LATENZA_PI.md` (sezione 6) e
`NEW_MASTERPLAN.md` (sezione 11, fase 4a).

## Perché lo facciamo

- **PIT pota solo canali.** Uno strato non sparisce mai, quindi la parte fissa (C2PSA, PSABlock,
  Detect, nodi non conv) è una costante C. Cambia solo il costo delle conv potate.
- **La latenza in funzione dei canali non è lineare né a gradini semplici.** Lo mostrano il probe
  `probe_edge` (salto del +30% a C = 33, cali a 48 e 64) e i plateau osservati sulla CPU della
  workstation. Una formula parametrica, β·MAC più un termine di allineamento, non descrive questi
  andamenti.
- **La separabilità è stata rifiutata.** L'ipotesi t(c_in, c_out) ≈ f(c_out)·g(c_in) non regge
  ([`pruning/separability`](../separability), 05/11, wandb run `6lhj71tw`):
  - punti entro 2× il rumore: 71% nella 1×1 a 40×40 e 34% nella 3×3 a 20×20, contro una soglia del 90%;
  - nella 3×3 c'è un cambio di kernel a c_in = 64 (con c_out = 128: da 0,39 a 0,22 ms);
  - nella 1×1 la colonna c_out = 128 si comporta in modo anomalo.
- **Quindi una LUT 2D misurata direttamente, a passo 2.** È il passo più piccolo sostenibile come
  tempo. PIT lavorerà con `pit.n` = 2, così ogni configurazione che può scegliere ha un valore
  misurato, senza interpolazione.
- **La LUT dà il costo dello strato da solo.** La composizione nella rete si stimerà dopo, su circa
  150 nani potati a caso misurati interi (B4-R), e non fa parte di questo esperimento. Comprende la
  costante C, un fattore a_f per famiglia ed eventualmente γ per concat, add e upsample.

## Condizioni

Le condizioni sono quelle del deploy, da `conf/hardware/rpi5.yaml`, `conf/backend/openvino.yaml` e
`conf/deploy/640.yaml`:

- **Board e sistema.** Raspberry Pi 5 Model B Rev 1.1 con Debian GNU/Linux 13 (trixie), non
  raspbian12. Kernel 6.18.50+rpt-rpi-2712, firmware ab8a9dde. OpenVINO
  2026.4.1-22982-e213a147257. I valori sono in `results/env.json`, riscritto a ogni avvio.
- **Calcolo.** 4 core (`cpu_4`), governor `performance`, 2,4 GHz (profilo `max`). Board preparata
  con lo stesso tuning del benchmark (`src/measure/board.py`: `board_controller`, `tuned`; `chrt`
  con sudo se manca `CAP_SYS_NICE`).
- **Comando**, costruito da `backend.build_cmd` e `_wrap_command` come in `src/stages/probe.py`:
  `benchmark_app -d CPU -b 1 -hint none -nstreams 1 -nthreads 4 -niter 200 -infer_precision f16
  -report_type no_counters`. La metrica è `median_ms`.
- **Iterazioni.** Si usano 200 iterazioni perché con 30 la mediana viene circa 2× il regime.
  Sul warmup: `warmup_iters: 50` nella conf di OpenVINO viene solo registrato, perché benchmark_app
  non ha un numero di iterazioni di warmup e ne fa una sola. Il transitorio lo assorbe la mediana su
  200: nella separabilità la mediana coincideva quasi con il minimo.
- **Guardia termica.** Il Pi non ha la ventola attiva. Oltre 75 °C, o con un bit di throttling
  attivo (`vcgencmd get_throttled` & 0xF), lo script aspetta di tornare a 65 °C e ripete la misura.
  Il firmware riduce la frequenza a 80 °C.
- **Stato registrato a ogni misura:** temperatura, frequenza, bit di throttling e governor, letti
  subito dopo benchmark_app.

## Rete di misura

- Una sola conv con pesi casuali (seme fisso per rete), bias e SiLU, come le Conv di YOLO26
  esportate con la BN fusa. La SiLU è scritta come Sigmoid·Mul, come nell'export ONNX.
- Le conv 3×3 hanno padding 1, le 1×1 nessun padding. Per le stride 2 l'ingresso è il doppio della
  mappa d'uscita.
- Una rete identità per ogni forma d'ingresso (c_in, mappa d'ingresso), per stimare il costo fisso:
  622 forme.
- Ogni rete viene costruita al momento e copiata sul Pi subito prima della sua misura, perché le
  reti sono più di 70 mila.

**La SiLU non è fusa nella conv** (`check_fusion.py`, `results/fusion.json`). Nel grafo compilato da
OpenVINO sul Pi (CPU ARM, f16) la conv con il bias è un nodo `Convolution`, mentre la SiLU finisce in
un nodo `Subgraph` (snippets, Swish) separato. Vale per tutte le famiglie provate: 1×1 e 3×3, stride
1 e 2, compresa la stem.

Il nano vero, compilato sul Pi nello stesso modo, fa lo stesso: 102 nodi `Convolution` e 101
`Subgraph` con la Swish. La LUT misura quindi lo strato come gira dentro la rete, cioè la conv più un
nodo di attivazione a parte. Nella rete però quel `Subgraph` può contenere anche altre operazioni
elementari dopo la SiLU. Nella rete di misura ci sono inoltre le conversioni f32 → f16 all'ingresso
e f16 → f32 all'uscita: anche questo costo, in parte, lo misura la rete identità.

## Famiglie e griglia

Le famiglie sono identificate da kernel, stride e mappa d'uscita. Per ognuna si indicano le coppie
massime (c_in, c_out) degli strati del nano:

| famiglia | strati (c_in, c_out) massimi | punti |
|---|---|---|
| `k1s1_20` (1×1 s1 20×20) | (128,64) (128,128) (256,128) (256,256) (384,256) (512,256) | 33 153 |
| `k1s1_40` | (64,32) (64,64) (128,128) (192,128) (384,128) | 12 545 |
| `k1s1_80` | (32,16) (32,32) (64,64) (96,64) (96,128) (256,64) | 5 825 |
| `k1s1_160` | (32,32) (48,64) | 825 |
| `k3s1_20` | (64,64) (64,128) (128,64) | 3 201 |
| `k3s1_40` | (32,32) | 289 |
| `k3s1_80` | (16,16) (16,32) (32,16) | 225 |
| `k3s1_160` | (8,16) (16,8) | 65 |
| `k3s2_20` | (128,128) (128,256) | 8 385 |
| `k3s2_40` | (64,64) (128,128) | 4 225 |
| `k3s2_80` | (64,64) | 1 089 |
| `k3s2_160` | (16,32) | 153 |
| `k3s2_320` (stem) | (3,16), con c_in fisso a 3 | 9 |
| **totale** | | **69 989** (+ 622 identità) |

La griglia di una famiglia è l'unione, sugli strati, dei punti con c_in ∈ {1, 2, 4, 6, …, c_in max}
e c_out ∈ {1, 2, 4, 6, …, c_out max}: cioè 1, tutti i pari e il massimo.

## Protocollo

- 3 ripetizioni per punto, in tutto 211 833 misure.
- L'ordine di tutti i punti di tutte le famiglie è mescolato con seme 0 e rimescolato a ogni
  ripetizione. Così una deriva non si concentra su una famiglia o su una zona della griglia.
- Ogni misura è scritta subito in `results/raw.csv`, in append. Lo script riprende da dove si era
  fermato, saltando le misure già fatte, quindi anche dopo un riavvio.
- Circa ogni 6 ore scrive un checkpoint in `results/status.json`: misure fatte, s/misura, pause
  termiche, temperatura massima, ETA. Lo scrive anche alla fine.
- Il processo gira staccato da Claude e dal terminale, come unit systemd utente
  `lut-pi-passo2.service` (`launch.sh`). Dopo un errore riparte da solo, fino a 20 volte.

**Pilota.** Su 100 punti a caso: 0,67 s a misura senza pause termiche (67 s, massimo 70 °C), per
una stima di 39 h. Con le pause attese senza ventola la durata cresce: con gli 1,65 s a misura della
separabilità sarebbero circa 97 h.

```bash
ITERS=200 python run.py pilot 100     # stima della durata (results/pilot.csv)
python check_fusion.py                # SiLU fusa o no nel grafo compilato (results/fusion.json)
systemd-run --user --unit=lut-pi-passo2 --working-directory=$PWD ./launch.sh   # misura completa
systemctl --user status lut-pi-passo2; tail -f runs/run.log; cat results/status.json
python analyze.py                     # LUT, riepilogo, heatmap
python log_wandb.py                   # run su wandb yolo-bench
```

## Durata reale e risultati

Avvio: 05/11/2026 alle 18:59. Da completare a fine misura.

## Limiti

- **Niente ventola.** Le pause termiche allungano la misura. Se il throttling scatta, la misura viene
  ripetuta; quante volte è successo è in `status.json` e `summary.json`.
- **Strati misurati da soli, non dentro la rete.** Cache, layout e fusioni con gli strati vicini
  sono diversi. Lo correggerà la composizione con C e a_f (B4-R).
- **SiLU non fusa nella conv**, sia qui sia nel nano vero: la LUT comprende il nodo di attivazione
  separato.
- **Conversioni f32 ↔ f16** all'ingresso e all'uscita della rete di misura.
- **Temperatura** letta dopo ogni misura, non durante.

## Percorsi

| percorso | contenuto |
|---|---|
| `run.py`, `launch.sh` | misura (pilota e completa), lancio staccato |
| `check_fusion.py` | verifica della fusione della SiLU |
| `analyze.py`, `log_wandb.py` | analisi e upload |
| `results/raw.csv` | una riga per misura: famiglia, kernel, stride, mappa, c_in, c_out, ripetizione, ms, temperatura, frequenza, throttling, timestamp |
| `results/lut_<famiglia>.csv`, `results/lut_identity.csv` | mediana, min, max e dispersione relativa per punto |
| `results/summary.json`, `results/status.json`, `results/env.json`, `results/fusion.json` | riepilogo, avanzamento, ambiente, fusione |
| `figures/heatmap_<famiglia>.png` | latenza mediana per famiglia (c_in × c_out, scala logaritmica) |
| `runs/run.log`, `runs/pilot.log` | log (ignorati da git) |
