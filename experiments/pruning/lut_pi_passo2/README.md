# LUT di latenza a passo 2 degli strati potabili del nano sul Pi 5

Questo esperimento misura le LUT di latenza delle conv potabili del nano YOLO26 (N, larghezza 0.25,
D126) sul Raspberry Pi 5. Ingresso sempre 640×640, 4 classi. Le LUT servono allo stimatore di latenza
per la potatura per canale (PIT, `~/Desktop/yolopit`): vedi
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
- Ingresso, uscita, pesi e bias sono in **fp16**, la precisione usata in inferenza: fp16 → fp16.
  Nel grafo compilato non c'è nessun `Convert` (`results/fusion.json`, `converts` vuoto per tutte le
  reti provate). La conversione fp32 → fp16 dell'ingresso fa parte del preprocessing, non dello
  strato, e se serve si misurerà a parte.
- Ingresso e uscita delle conv sono in **NHWC**, il layout che il plugin CPU ARM usa all'interno.
  L'ONNX resta NCHW: il PrePostProcessor di OpenVINO dichiara i tensori NHWC e la rete è salvata come
  IR (`.xml` + `.bin`, funzione `to_ir` di `run.py`). Con l'ONNX NCHW, invece, OpenVINO eseguiva a
  ogni inferenza un `Reorder` NCHW → NHWC sull'ingresso e uno NHWC → NCHW sull'uscita. Nel nano
  questi riordini non ci sono, perché fra uno strato e l'altro i tensori restano NHWC. Su 160×160
  costavano più della conv: 290 + 745 µs contro 381.
- **Cosa gira a ogni inferenza.** I performance counter (`PERF_COUNT`, ventesima inferenza,
  `results/fusion.json`) mostrano solo `Convolution` e `Subgraph` (la SiLU), in f16, per tutte le
  famiglie provate. Il `Reshape` sui pesi è preparato alla compilazione e non compare.
- Le conv 3×3 hanno padding 1, le 1×1 nessun padding. Per le stride 2 l'ingresso è il doppio della
  mappa d'uscita.
- Ogni rete viene costruita al momento e copiata sul Pi subito prima della sua misura, perché le
  reti sono più di 70 mila.

**La SiLU non è fusa nella conv** (`check_fusion.py`, `results/fusion.json`). Nel grafo compilato da
OpenVINO sul Pi (CPU ARM, f16) restano due nodi, tutti e due in f16: la conv con il bias in un nodo
`Convolution` e la SiLU in un nodo `Subgraph` (snippets, Swish) separato. Vale per tutte le famiglie provate: 1×1 e 3×3, stride
1 e 2, compresa la stem.

Il nano vero, compilato sul Pi nello stesso modo, fa lo stesso: 102 nodi `Convolution` e 101
`Subgraph` con la Swish. La LUT misura quindi lo strato come gira dentro la rete, cioè la conv più un
nodo di attivazione a parte. Nella rete però quel `Subgraph` può contenere anche altre operazioni
elementari dopo la SiLU.

Il nano vero non ha conversioni fra gli strati. In tutto il grafo compilato (484 nodi) c'è un solo
f32 → f16, al nodo 2, sull'immagine in ingresso. Backbone e neck girano interamente in f16: 290
nodi di calcolo. Nodi in f32 o int32 compaiono solo dal nodo 445, nel post-processing della Detect
(TopK, gather, divisioni), che non viene potato e finisce nella costante C.

**Costo fisso per inferenza: lo assorbe C, niente reti identità.** benchmark_app misura ogni inferenza
dalla chiamata `infer()` al ritorno. Dentro c'è lo strato, ma anche un lavoro fisso di OpenVINO a ogni
chiamata: avvio della richiesta, risveglio e sincronizzazione dei 4 thread, ritorno del risultato. Nel
nano vero si paga una volta per immagine, mentre sommando i 102 strati della LUT lo si conta 102 volte.

PIT pota solo canali, quindi i 102 strati ci sono sempre e quel costo entra nella somma sempre 102
volte, qualunque sia la potatura. È una costante, e la assorbe C nel fit sui nani potati (B4-R): non
serve sottrarlo strato per strato. Le identità a 2 canali di `warmup.py` danno circa 0,013 ms a chiamata,
uguale a 20, 40 e 80 di mappa (0,0128, 0,0131, 0,0134 ms).

Il piano iniziale prevedeva 622 reti identità, una per forma d'ingresso, da sottrarre forma per forma.
Sono state tolte il 06/11 per due motivi. Primo, sulle forme grandi l'identità copia tutto l'ingresso
(0,76 ms a 160×160×64), una copia che la conv non fa: la sottrazione toglierebbe un costo che lo strato
non paga, e che dipende da c_in, quindi C non lo compenserebbe. Secondo, con 200 iterazioni le identità
piccole escono gonfiate dal warmup del 47–76% (sezione Warmup). Decisione e prove per esteso in
[`DECISIONE_IDENTITA.md`](DECISIONE_IDENTITA.md). Ipotesi non verificata: il costo per
chiamata di una rete conv (NHWC) non dipende dai canali; per assorbirlo in C basta questo.

**Versioni scartate**, i cui dati sono stati cancellati il 05/11 su richiesta. Tutte e tre sono state
lanciate e fermate il 05/11:
- *I/O fp32*, dalle 18:59 alle 21:30, con pesi fp32 e calcolo f16 (`-infer_precision f16`). Il grafo
  aveva un `Convert` f32 → f16 all'ingresso e uno f16 → f32 all'uscita, un costo che lo strato
  dentro la rete non paga: le conv costavano dal 4 al 43% in più, di più sulle mappe grandi.
- *I/O fp16 ma NCHW.* Fermata prima della prima misura, quando i performance counter hanno mostrato
  i `Reorder`: senza questi le conv costano dal 10 al 58% in meno.
- *I/O fp16 NHWC senza le singole inferenze*, alle 21:48, per circa 600 misure. Fermata per salvare
  anche il tempo di ogni inferenza (`bench_dump.py`).

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
| **totale** | | **69 989** |

La griglia di una famiglia è l'unione, sugli strati, dei punti con c_in ∈ {1, 2, 4, 6, …, c_in max}
e c_out ∈ {1, 2, 4, 6, …, c_out max}: cioè 1, tutti i pari e il massimo.

## Protocollo

- 3 ripetizioni per punto, in tutto 209 967 misure.
- **Punti instabili.** Alla fine, i punti in cui le 3 ripetizioni differiscono più del 5%
  ((max − min) / mediana) vengono misurati altre 6 volte, e la LUT ne prende la mediana su 9. Serve per i
  punti con due regimi come `k1s1_160` 48×64 (sezione Due regimi). Nel pilota rimisurato nessun punto
  superava il 5,2%, quindi sono attesi pochi punti. `run.py unstable()`, test in `test_run.py`; il numero
  finisce in `summary.json` (`unstable_remeasured`).
- L'ordine di tutti i punti di tutte le famiglie è mescolato con seme 0 e rimescolato a ogni
  ripetizione. Così una deriva non si concentra su una famiglia o su una zona della griglia.
- Ogni misura è scritta subito in `results/raw.csv`, in append. Lo script riprende da dove si era
  fermato, saltando le misure già fatte, quindi anche dopo un riavvio.
- Circa ogni 6 ore scrive un checkpoint in `results/status.json`: misure fatte, s/misura, pause
  termiche, temperatura massima, ETA. Lo scrive anche alla fine.
- Il processo gira staccato da Claude e dal terminale, come unit systemd utente
  `lut-pi-passo2.service` (`launch.sh`). Dopo un errore riparte da solo, fino a 20 volte.

**Pilota** (I/O fp16 NHWC, 05/11 verso le 21:50): 0,62 s a misura, senza pause termiche (massimo 73 °C),
per una stima di 36,5 h. Il primo lancio (I/O fp32) era sceso dopo due ore a 2,5 s a misura, con il
75% del tempo in attesa termica: senza ventola il ritmo lo decide il raffreddamento, quindi la
durata reale sarà più lunga.

```bash
ITERS=200 python run.py pilot 100     # stima della durata (results/pilot.csv)
python check_fusion.py                # SiLU fusa o no nel grafo compilato (results/fusion.json)
systemd-run --user --unit=lut-pi-passo2 --working-directory=$PWD ./launch.sh   # misura completa
systemctl --user status lut-pi-passo2; tail -f runs/run.log; cat results/status.json
systemctl --user stop lut-pi-passo2   # ferma e ripristina la board (run.py tratta SIGTERM come Ctrl-C)
python analyze.py                     # LUT e riepilogo
python figures.py lut                 # heatmap per famiglia e di tutte le famiglie
python log_wandb.py                   # run su wandb yolo-bench
```

## Warmup di benchmark_app (`warmup.py`, 06/11)

**Domanda.** Le prime inferenze di ogni misura sono più lente: per quanto tempo, e la mediana su 200
iterazioni lo assorbe davvero? Un primo tentativo a 2000 iterazioni (`transient.py`) non si chiude senza
ventola: sulle reti da 3 ms il Pi arriva a 82 °C e la misura si ripete all'infinito. Fermato a 12/84.

**Come.** 19 reti fra 0,013 e 3 ms: le identità più piccole e più grandi di ogni mappa, più conv scelte
dal pilota. Ognuna gira 2 s (`-t 2` al posto di `-niter`). 5 ripetizioni in ordine mescolato: nelle
ripetizioni 3 e 4 il Pi resta fermo 10 s prima della misura. 95 misure in 21 minuti (06/11, 00:09–00:30),
15 ripetute dopo una pausa termica. Regime = mediana dell'ultimo secondo. Transitorio = primo istante da
cui 3 finestre di fila (max(20 iterazioni, 10 ms)) hanno la mediana entro il 2% del regime. Il tempo è
stimato: alle latenze si aggiunge un intervallo uniforme fra le inferenze, perché la durata reale non viene
salvata. wandb run `jgq9syim` (misure, analisi e verifica; `fz2d692t` è la prima versione, senza verifica).

**Risultati** (`results/warmup.json`, `figures/warmup.png`, `figures/warmup_gain.png`):
- **Il warmup dura un tempo fisso, non un numero di iterazioni:** circa 20 ms (mediana 17, p90 33,
  massimo 58), da 0 a 1540 iterazioni secondo la rete. Nella figura le curve si sovrappongono sull'asse
  del tempo e scendono a gradino intorno a 20 ms, non in modo graduale.
- **La pausa di 10 s non cambia nulla:** transitorio mediano 15 ms contro 17.
- **Con il protocollo attuale (mediana delle prime 200 iterazioni) le reti sotto ~0,1 ms risultano
  gonfiate:** identità a 2 canali dal +47 al +76%, `k1s1_40` 2×2 +28%, `id20` 512×512 +27%. Sopra 0,1 ms
  l'errore è sotto l'1,5%. Per questo, e per il motivo nella sezione Costo fisso, le identità sono
  uscite dal piano.
- **Scartando i primi 50 ms e prendendo la mediana dei 200 ms dopo,** l'errore rispetto al regime è dello
  0,16% in mediana, dello 0,66% al p90 e del 2,8% al massimo.
- **Cosa non torna:** fra una ripetizione e l'altra il regime delle identità più piccole (circa 0,013 ms)
  varia del 10–12%, cioè di circa 1,5 µs, anche senza warmup. Una ripetizione di `k1s1_160` 48×64 viene
  2,75 ms invece di 3,09 (−11%) alla stessa temperatura.

**Correzione: warmup a tempo in `bench_dump.py`.** benchmark_app fa una sola inferenza di riscaldamento
(`Benchmark.first_infer`) prima del ciclo misurato. `bench_dump.py` la ripete per 60 ms (`WARMUP_MS`): le
200 iterazioni partono già a regime, per qualunque rete, senza stimare prima la latenza. Costa circa 60 ms a
misura: la durata stimata passa da circa 36,5 a circa 40 h.

**Verifica** (`warmup.py check`, 06/11 verso le 00:50): le stesse 19 reti con il protocollo della LUT
(200 iterazioni) più il warmup da 60 ms, 3 ripetizioni, confrontate con il regime delle misure da 2 s
(`results/warmup_check.json`):
- **il bias del warmup sparisce:** le identità a 2 canali passano da +47/+76% a errori fra −8 e +10% di
  segno misto, `k1s1_40` 2×2 da +28% a fra −3 e +0,5%;
- **conv:** errore mediano 0,7%, p90 2,6%, massimo 10,4%. Senza `k1s1_160` 48×64: mediana 0,6%, massimo 2,9%;
- **cosa non torna:** `k1s1_160` 48×64 sembra avere due regimi, circa 3,09 ms oppure 2,75–2,78 ms (−10%).
  Succede in 1 ripetizione su 5 delle misure da 2 s e in 2 su 3 della verifica, quindi non è warmup. Da
  capire prima della LUT: con 3 ripetizioni la mediana sceglie uno dei due regimi a caso. Gli errori residui
  delle identità piccole (fino a −23% per `id320` 2×2) somigliano alla variabilità fra ripetizioni già vista
  nelle misure da 2 s (10–12%); le identità non sono più nel piano.

**Quanto guadagna la LUT** (`warmup.py pilot`, 06/11 verso le 01:05): le 97 conv del pilota, distribuite
come la LUT, rimisurate 2 volte con il warmup da 60 ms e confrontate con i valori del pilota
(`results/warmup_pilot.json`):
- 3 punti su 97 erano molto sbagliati e ora sono corretti, tutti conv sotto 0,15 ms: `k1s1_40` 36×6 +98%,
  `k1s1_20` 178×4 +80%, `k1s1_20` 26×8 +38%;
- sugli altri 94 non cambia nulla: differenza mediana 0,0%, massima +3,9%, dentro il rumore fra
  ripetizioni (mediana 0,75%, p90 2%);
- sulla somma delle latenze dei 97 punti l'effetto è +0,4%.

```bash
python warmup.py measure; python warmup.py analyze; python warmup.py check; python warmup.py pilot
python figures.py warmup; python warmup.py log
```

## Due regimi di `k1s1_160` 48×64 (`bimodal.py`, 06/11)

Con il protocollo della LUT (200 iterazioni, warmup da 60 ms) la rete misurata 20 volte dà 2,77–2,82 ms in 5
misure e 3,07–3,12 ms nelle altre 15. Dentro una misura il regime non cambia (seconda metà / prima metà =
1,000 ± 0,004): si decide all'avvio. Le 4 reti vicine misurate 10 volte (`k1s1_160` 32×32 e 48×62,
`k3s1_160` 8×16, `k1s1_80` 58×126) non hanno due regimi. 48×62 costa 4,0 ms, più di 48×64.

Sonde sul Pi, con 3 processi e 8 compilazioni per rete in ciascuno:
- **v1** (`PERFORMANCE_HINT LATENCY`, ingresso lasciato a OpenVINO): kernel sempre `gemm_acl_f16`;
  48×64 sempre fra 3,37 e 3,41 ms, regime veloce mai visto;
- **v2** (impostazioni di benchmark_app, ingresso nuovo a ogni compilazione con l'indirizzo registrato):
  48×64 fra 3,35 e 3,76 ms, regime veloce mai visto, nessun legame chiaro con l'allineamento dei buffer.

Le sonde sono più lente di benchmark_app (3,4–3,6 contro 3,09 ms) e non lo riproducono: la causa resta
aperta. Non è la scelta del kernel. Nel pilota rimisurato (97 punti, 2 ripetizioni) nessun punto differisce
più del 5,2% fra le due ripetizioni: i punti bimodali sono pochi.

```bash
python bimodal.py measure; python bimodal.py compile; python bimodal.py analyze; python figures.py bimodal
```

## Durata reale e risultati

Lanciata il 06/11 alle 11:10 con la ventola sempre al massimo (`scripts/provision/rpi5.sh`) e soglia di partenza
a 50 °C come il benchmark. Pilota con ventola (06/11, 100 punti, `results/pilot.csv`): 0,67 s/misura, 35–46 °C,
nessun throttling; stima 39 h, fine attesa domenica 08/11 verso le 02:00. Il lancio del 05/11 alle 21:48 (senza
ventola, un'inferenza di warmup) era stato fermato alle 22:06 per la prova del warmup; il suo log è `runs/run_0511.log`.

**Coda dopo la LUT principale** (decisa il 06/11):
1. estensione della LUT alle forme delle architetture B4-R che mancano (circa 2 500 punti, 1–2 h): k3s1_40 fino a
   32×64/64×32 (D101), k1s1_160 64×64 e k1s1_80 128×128 (w0.25_D170), k3s1_160 con 3 canali (w0.09375);
2. profiling per nodo sul Pi (PERF_COUNT, nano intero e un potato): quanto pesano Concat, Add, Interpolate e
   MaxPool. Sulla workstation (x86, bf16) cambiano con la potatura di circa l'1%, meno del rumore delle conv;
3. misura delle reti B4-R (`../b4r_composizione`).

**Efficienza, da fare prima della prossima LUT (estensione o S), non a misura in corso:** il Pi calcola solo per
circa il 23% del tempo (0,14 s di 200 inferenze più warmup, su 0,60 s per misura). Il resto è preparazione:
build e IR sulla workstation (0,07 s), invio, e soprattutto un Python nuovo sul Pi che importa OpenVINO e compila
(stimato 0,3–0,4 s). Due modifiche, circa 2× più veloce:
- un processo sempre aperto sul Pi (`bench_dump.py` in modalità server) che riceve le reti una dopo l'altra;
- la rete successiva costruita sulla workstation mentre il Pi misura quella corrente.

Cambia il protocollo: con il processo aperto il plugin CPU parte già caldo. Prima di usarlo va rifatta la prova
del warmup (`warmup.py check`), e una LUT va misurata tutta con lo stesso protocollo.

## Limiti

- **Ventola.** Le prove del 05–06/11 (pilota, warmup, bimodal) sono senza ventola; la LUT girerà con la
  ventola. Le pause termiche allungano la misura. Se il throttling scatta, la misura viene
  ripetuta; quante volte è successo è in `status.json` e `summary.json`.
- **Strati misurati da soli, non dentro la rete.** Cache, layout e fusioni con gli strati vicini
  sono diversi. Lo correggerà la composizione con C e a_f (B4-R).
- **SiLU non fusa nella conv**, sia qui sia nel nano vero: la LUT comprende il nodo di attivazione
  separato.
- **Temperatura** letta dopo ogni misura, non durante.
- **Pesi in cache.** Ogni strato gira 200 volte di fila, quindi i suoi pesi stanno in cache (L2 512 KB per core,
  L3 2 MB). Nel nano che gira in continuo i pesi (~5 MB in fp16) non ci stanno, e gli strati con molti pesi li
  rileggono dalla RAM. A mappa fissa, sia questo costo sia il calcolo crescono con c_in·c_out·k², quindi lo
  dovrebbe assorbire a_f; i residui di B4-R lo diranno.

## Percorsi

| percorso | contenuto |
|---|---|
| `run.py`, `launch.sh` | misura (pilota e completa), lancio staccato |
| `check_fusion.py` | verifica della fusione della SiLU |
| `bimodal.py` | indagine sui due regimi di `k1s1_160` 48×64 |
| `DECISIONE_IDENTITA.md` | perché la LUT non misura né sottrae le reti identità (06/11) |
| `warmup.py`, `transient.py` | durata del warmup di benchmark_app (`transient.py` non si chiude senza ventola) |
| `analyze.py`, `log_wandb.py` | analisi (solo numeri) e upload |
| `figures.py` | tutte le figure, stile comune `tools/plots.py` (PNG + PDF): `lut`, `warmup`, `bimodal`, `all` |
| `results/raw.csv` | una riga per misura: famiglia, kernel, stride, mappa, c_in, c_out, ripetizione, ms, temperatura, frequenza, throttling, timestamp |
| `results/lut_<famiglia>.csv` | mediana, min, max e dispersione relativa per punto |
| `results/summary.json`, `results/status.json`, `results/env.json`, `results/fusion.json` | riepilogo, avanzamento, ambiente, fusione |
| `figures/heatmap_<famiglia>.png`, `figures/heatmap_all.png` | latenza mediana per famiglia (c_in × c_out, scala logaritmica; la stem come curva) |
| `runs/run.log`, `runs/pilot.log` | log (ignorati da git) |
