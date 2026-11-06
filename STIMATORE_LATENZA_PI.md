# STIMATORE DI LATENZA SUL RASPBERRY PI — dettagli per la fase 4a

Estende la sezione 11 (fase 4a) di `NEW_MASTERPLAN.md` e ne vale la stessa convenzione: il testo fra `<esempio>` e
`</esempio>` è un'ipotesi o un numero non verificato; il resto è una decisione presa o un fatto misurato, con
l'indicazione di dove. Raccoglie le discussioni del 05/11/2026 sul predittore di latenza per la potatura sul PI.

## 1. Perimetro

- **Un solo dispositivo come dimostrazione**: Raspberry Pi 5.
- **Configurazione fissata a priori**, una sola: OpenVINO fp16 (`infer_precision` f16), 4 core, governor
  `performance`, frequenza massima (2,4 GHz). Il predittore vale solo per questa configurazione: con un altro backend,
  un'altra precisione, altri core o un'altra frequenza tutti i parametri vanno rimisurati. Dalla repo
  (`conf/hardware/rpi5.yaml`, `conf/backend/openvino.yaml`, `conf/deploy/640.yaml`):

  ```
  benchmark_app -d CPU -b 1 -hint none -nstreams 1 -nthreads 4 -niter 200 -infer_precision f16 -report_type no_counters
  ```

  con 50 iterazioni di warmup, metrica `median_ms`, tuning della board di `src/measure/board.py` (chrt con sudo se
  manca `CAP_SYS_NICE`).
- **Ingresso sempre 640×640**, **4 classi**: non si cambiano. Le risoluzioni di deploy (480p, 720p, 1080p) restano
  fuori dal predittore: 640×640 basta a dimostrare che funziona.
- **Modello da potare**: N (0.25, D126).

## 2. Cosa pota PIT (yolopit)

Verificato sul codice di yolopit (`~/Desktop/yolopit`, HEAD `fd2f3a3`) sul computer aziendale:

- La potatura è **solo per canale**: toglie filtri di una conv (o neuroni di un layer lineare). **Uno strato non viene
  mai eliminato**: sopravvive anche con un solo filtro. Il numero di strati della rete potata è sempre quello del
  modello di partenza.
- **Esclusi**: C2PSA e PSABlock (attenzione) e Detect (tutta la testa, comprese le conv di uscita). Sono foglie fx
  (`LEAVES = (C2PSA, PSABlock, Detect)`, `yolopit/search.py:63`); il loro costo è contato come parte fissa
  (`costs.py`).
- Gli strati che alimentano una foglia hanno i **canali di uscita congelati** (`pit_layers_feeding`,
  `freeze_output_channels`, `masks.py:256-320`); i loro canali di ingresso si adattano.
- **Residui, add e concat**: maschere condivise fra i rami (`masks.py:171`).
- **C3k2**: potato all'interno (sostituito da `C3k2Split`, equivalente senza `chunk`).
- **Depthwise**: nessuna regola esplicita in yolopit; in YOLO26 stanno tutte dentro Detect, quindi non si potano
  (\<esempio>da verificare\</esempio>).
- Le strategie attuali (`conf/strategy/pit_standard.yaml`, `pit_duccio.yaml`) hanno `pit.n: 16`: oggi i canali si
  potano **a blocchi di 16**. L'obiettivo di questo predittore è anche togliere questo vincolo: è il predittore a
  dire quanto costa un numero di canali non allineato, non la potatura a imporre l'allineamento.

## 3. Correzioni alla fase 4a del masterplan

La formula della fase 4a, `c0 + strati × α + Σ calcoli(canali arrotondati a g) × β`, va adattata alla potatura per
canale:

1. **`α × numero di strati` è costante** su tutte le reti potate dallo stesso modello: si fonde con `c0` in un'unica
   costante `C`. `α` si può ancora misurare a parte (catene di B1-1) per spiegare quanta parte di `C` è costo degli
   strati, ma non guida la potatura e non si può stimare dalle reti potate.
2. Le frasi "togliere canali o togliere blocchi" (2c, 4b) non si applicano: l'unica leva della potatura è la
   larghezza.
3. **Niente arrotondamento forzato a `g`**: l'allineamento diventa un parametro stimato, non un vincolo imposto alla
   potatura. Prima versione discussa:

   ```
   latenza = C + Σ_strati [ β_tipo · MAC(c_in, c_out) + δ · MAC_pad ]
   MAC(c_in, c_out) = H_out · W_out · k² · c_in · c_out
   MAC_pad = MAC(⌈c_in/g⌉·g, ⌈c_out/g⌉·g) − MAC(c_in, c_out)
   ```

   `δ` fra 0 (l'allineamento non conta) e `β` (un blocco iniziato costa come pieno). La potatura di uno strato cambia
   il suo `c_out` e il `c_in` dello strato successivo.

Questa formula **non basta**: le curve canali → latenza osservate non sono né lineari né a gradini semplici (sezione
5). Da qui la scelta della LUT (sezione 6). Widget interattivo della formula parametrica:
https://claude.ai/artifact/394Eb6UYTtSqU7UPAqa7G2 (valori di AGX e DX inventati).

## 4. Strati potabili del nano a 640×640

Ricavati dalla struttura di `yolo26n.pt` ufficiale (stessa topologia di N D126; 126 conv, nessun layer lineare: anche
l'attenzione usa conv 1×1). Esclusi C2PSA, PSABlock e Detect restano **66 conv in 13 famiglie**. Una famiglia è
(tipo, stride, mappa d'uscita); `c_in` e `c_out` sono i massimi del nano, cioè i limiti della LUT.

| famiglia | mappa | strati | coppie (c_in, c_out) del nano | MAC del nano | punti a passo 1 |
|---|---|---|---|---|---|
| 1×1 s1 | 20×20 | 9 | (128,64) (128,128) (256,128) (256,256) (384,256) (512,256) | 10% | 131 072 |
| 1×1 s1 | 40×40 | 15 | (64,32) (64,64) (128,128) (192,128) (384,128) | 13% | 49 152 |
| 1×1 s1 | 80×80 | 7 | (32,16) (32,32) (64,64) (96,64) (96,128) (256,64) | 12% | 22 528 |
| 1×1 s1 | 160×160 | 2 | (32,32) (48,64) | 5% | 3 072 |
| 3×3 s1 | 20×20 | 6 | (64,64) (64,128) (128,64) | 5% | 12 288 |
| 3×3 s1 | 40×40 | 12 | (32,32) | 8% | 1 024 |
| 3×3 s1 | 80×80 | 6 | (16,16) (16,32) (32,16) | 5% | 768 |
| 3×3 s1 | 160×160 | 2 | (8,16) (16,8) | 3% | 192 |
| 3×3 s2 | 20×20 | 2 | (128,128) (128,256) | 8% | 32 768 |
| 3×3 s2 | 40×40 | 2 | (64,64) (128,128) | 13% | 16 384 |
| 3×3 s2 | 80×80 | 1 | (64,64) | 11% | 4 096 |
| 3×3 s2 | 160×160 | 1 | (16,32) | 5% | 512 |
| 3×3 s2 | 320×320 | 1 | (3,16): `c_in` = 3, l'immagine | 2% | 16 |

- La mappa è quella d'uscita dello strato (per le s2 l'ingresso è il doppio).
- Il peso in MAC è distribuito: nessuna famiglia supera il 13%. Escludere famiglie per risparmiare misure vorrebbe dire
  non predire una parte reale della rete.
- I punti a passo 1 contano l'unione dei rettangoli `[1..c_in] × [1..c_out]` dei singoli strati: **273 872 in
  tutto**, quasi metà nella 1×1 a 20×20.

## 5. Probe già misurati

Sul computer aziendale, in `~/Desktop/yoloeval/.old/results/probe/` (26 probe; export CSV in
`~/probe_export_20261105/`). P1-1 e B1-1 non sono mai stati lanciati (i preset esistono:
`conf/stage/probe_p1.yaml`, `conf/stage/layers_b1.yaml`).

- Tutti con il preset `probe_edge`: conv 3×3 su mappa 20×20, 4 strati, `c_in = c_out = C` da 1 a 96 a passo 1,
  3 ripetizioni (288 misure per probe), board rpi5, backend onnxruntime (fp32, fp16, int8) e OpenVINO fp16, 1/2/4
  core, frequenza massima e bassa.
- **Tempo per misura**: circa 0,4 s con onnxruntime, circa 0,8–1,1 s con OpenVINO (fino a 1,4 s a frequenza bassa).
  `PhaseTimer` salva solo setup (0,2–1 s) e compute (compilazione, benchmark_app e misura insieme).
- **Curva OpenVINO fp16, 4 core** (`2db8ff70`, frequenza bassa): lineare a tratti da 0,169 ms (C = 1) a 0,469 ms
  (C = 32); **salto di +30% a C = 33** (0,610 ms); **cali a C = 48 e 64** (a 64: 0,91 ms contro 1,08 a 63); sopra 1 ms
  benchmark_app dà due decimali. Dispersione fra le ripetizioni tipicamente 0,5–3%.
- **Attenzione**: i probe `probe_edge` usano 30 iterazioni. Nell'esperimento del 05/11 (sezione 6) con 30 iterazioni
  la mediana veniva circa 2× il regime per il transitorio iniziale: questi probe sono probabilmente distorti e vanno
  considerati poco affidabili. Da qui in poi si usano 200 iterazioni.
- Osservazione anche sulla CPU della workstation: andamento lineare con **plateau** (64, 68, 72 e 76 canali hanno quasi
  la stessa latenza). Né uno sconto fisso sui multipli di `g` né un gradino semplice descrivono entrambi i casi.

## 6. LUT per famiglia

Ogni famiglia ha una tabella misurata `LUT_f(c_in, c_out)`: latenza di una conv di quella famiglia, da sola, sul PI
nella configurazione fissata. Il passo deve essere **1**: con un passo più largo si perdono salti e cali come quelli
della sezione 5.

**Costo**: a passo 1 su tutta la griglia sono 273 872 punti; a circa 1 s per misura e 3 ripetizioni,
\<esempio>circa 10 giorni di macchina\</esempio>: non fattibile.

### Ipotesi di separabilità

```
LUT_f(c_in, c_out) ≈ F_f(c_out) · G_f(c_in)      (log t additivo nelle due variabili)
```

Motivo: la conv si esegue come un prodotto di matrici `(H·W) × c_out × (c_in·k²)`, con blocchi lungo `c_out`
(registri SIMD) e lungo `c_in` (cache): `t ≈ ⌈c_out/t_N⌉ · ⌈c_in·k²/t_K⌉ · costo del blocco`, prodotto di due
funzioni a una variabile. Non era garantito: cade se OpenVINO sceglie kernel diversi a seconda della forma. Se avesse
retto, la LUT di una famiglia si sarebbe ricavata da due scansioni 1D a passo 1 (3 024 punti, circa 2,5 ore).

### Esperimento di verifica (fatto il 05/11/2026): separabilità rifiutata

Sul PI, nella configurazione fissata, con una sola conv (bias, niente attivazione, pesi casuali) e una rete vuota per
il costo fisso di benchmark_app:

- **Famiglie**: A = 1×1 s1 40×40; B = 3×3 s1 20×20.
- **Scansioni 1D a passo 1**: `c_out` da 1 a 128 con `c_in ∈ {32, 64, 128}`, e `c_in` da 1 a 128 con
  `c_out ∈ {32, 64, 128}`.
- **Griglia 2D di verifica**: `c_in, c_out ∈ {8, 16, 24, 31, 32, 33, 40, 47, 48, 49, 63, 64, 65, 80, 96, 112, 127,
  128}` (18×18).
- 3 ripetizioni per punto, ordine mescolato a ogni ripetizione (seed fisso), mediana; condizioni registrate.
  \<esempio>Circa 1 700 punti per famiglia\</esempio>.
- **Analisi**: soglia di rumore (mediana della dispersione); fit di `log t = a(c_in) + b(c_out)` sulla griglia 2D
  (R², errore relativo mediano e massimo, mappa dei residui); errore della predizione dalle sole scansioni 1D
  (riferimento `ci_ref = co_ref = 64`).
- **Criterio**: separabile se l'errore della predizione 1D è entro 2 volte la soglia di rumore per almeno il 90% dei
  punti.

**Esito** (5 928 misure in 162 minuti, 200 iterazioni per misura, Pi senza ventola: 57 misure ripetute sopra 75 °C,
throttling mai scattato, frequenza sempre 2,4 GHz):

| | A (1×1, 40×40) | B (3×3, 20×20) |
|---|---|---|
| rumore (mediana della dispersione) | 3,4% | 2,2% |
| fit additivo su log t: R² | 0,973 | 0,879 |
| fit additivo: errore mediano / massimo | 3,3% / 35% | 8,0% / 54% |
| predizione 1D (riferimento 64): errore mediano / massimo | 3,8% / 64% | 14% / 99% |
| punti entro 2× il rumore | 71% | 34% |
| verdetto | **non separabile** | **non separabile** |
| costo della rete identità | 0,060 ms | 0,036 ms |

Sottrarre il costo dell'identità peggiora il fit (A: dal 71% al 47% entro soglia): il costo fisso non spiega lo
scarto. Dove fallisce:

- **B, cambio a `c_in` = 64**: sotto 64 il tempo cresce a dente di sega con pendenza che dipende da `c_out`, a 64
  crolla (con `c_out` = 128: 0,39 ms a `c_in` = 63, 0,22 ms a 64). Probabile cambio di kernel che dipende da tutti e
  due gli assi.
- **B, `c_in` piccolo**: il punto di cambio si sposta con `c_out` (circa 9 per `c_out` = 64, circa 20 per 32).
- **Gradini su `c_out` a 32, 64 e 96** in tutte e due le famiglie, con altezza che dipende da `c_in` (B: +50% a 96
  con `c_in` = 32, circa +15% con 64 e 128).
- **A, `c_out` = 128**: alcuni `c_in` vanno circa 2× più veloci dei vicini (17: 0,22 ms; 18: 0,41 ms), riproducibile
  nelle 3 ripetizioni, senza legame con i multipli di 8. Causa non chiarita: servirebbe vedere quale kernel sceglie
  OpenVINO.
- **A, `c_in` ≤ 24 con `c_out` ≤ 32**: il fit sovrastima del 30–60%.

Dati (computer aziendale): `~/sep_probe_20261105/` (`data/raw.csv`, `results/`, `scripts/`); wandb
https://wandb.ai/yolo-bench/yolo-bench/runs/6lhj71tw.

### Decisione: LUT 2D misurata direttamente, passo 2

Il passo della LUT coincide con `pit.n`: se PIT pota a multipli di `n` e la LUT è misurata a passo `n`, ogni
configurazione che PIT può scegliere ha un valore misurato, senza interpolazione. Si sceglie il passo più piccolo che
il tempo permette. Griglia di una famiglia: unione, sugli strati, dei punti con `c_in, c_out ∈ {1, n, 2n, …, massimo}`.
Costo con 1,57 s a misura (pause termiche comprese, senza ventola):

| passo | punti | 3 ripetizioni | 2 ripetizioni |
|---|---|---|---|
| 1 | 273 872 | circa 15 giorni | circa 10 giorni |
| **2** | **69 989** | **circa 92 ore** | circa 61 ore |
| 3 | 32 551 | circa 43 ore | circa 28 ore |
| 4 | 17 885 | circa 23 ore | circa 16 ore |
| 6 | 8 712 | circa 11 ore | circa 8 ore |
| 8 | 4 670 | circa 6 ore | circa 4 ore |

**Scelta: passo 2, 3 ripetizioni, `pit.n` = 2.**

Protocollo della misura (affidata a un agente sul computer aziendale, in `~/lut_pi_passo2_<data>/` con README):

- rete di misura: una sola conv con bias e SiLU, come le Conv di YOLO26 esportate con la BN fusa (verificare nel
  grafo compilato che la SiLU sia fusa); stride 2 con ingresso doppio della mappa d'uscita e padding 1; niente reti
  identità: il costo fisso per chiamata è uguale per tutti gli strati e lo assorbe C (tolte il 06/11, vedi il
  README dell'esperimento);
- 200 iterazioni, 3 ripetizioni, ordine di tutti i punti di tutte le famiglie mescolato con seed fisso e rimescolato a
  ogni ripetizione;
- pausa e ripetizione sopra 75 °C; condizioni registrate a ogni misura; Pi senza ventola (limite da dichiarare);
- prova su 100 punti per stimare la durata (fermarsi se oltre 110 ore); esecuzione in background con salvataggio in
  append e ripresa dopo un riavvio; checkpoint di avanzamento ogni 6 ore circa;
- output: `data/raw.csv`, `results/lut_<famiglia>.csv`, `results/summary.json`, heatmap per famiglia, upload su wandb.

## 7. Dalla LUT alla latenza della rete

La LUT dà il costo di uno strato misurato da solo. Dentro la rete il costo cambia (cache già calda con l'uscita dello
strato precedente, fusioni) e ci sono operazioni che non sono conv. Modello:

```
latenza_rete = C + Σ_f a_f · Σ_{strati in f} LUT_f(c_in, c_out)  [+ γ · Σ byte spostati da concat, add, upsample]
latenza di uno strato = a_f · LUT_f(c_in, c_out)
```

- `C`: tutto quello che la LUT non vede: C2PSA, PSABlock e Detect (non potati), le matmul dell'attenzione, i nodi
  non conv. Vale per la rete intera, non per lo strato.
- `a_f`: correzione della famiglia `f` dentro la rete (per esempio 0,8 = dentro la rete quegli strati costano il 20%
  in meno che da soli).
- `γ` (facoltativo): concat, somme dei residui e upsample costano in proporzione ai canali, quindi con la potatura
  cambiano e non stanno davvero in `C`.

**Stima dei parametri** (circa 20: `C`, 13 `a_f`, eventualmente `γ`):

1. Circa 150 nani potati a caso (rapporto di potatura indipendente per strato, vincoli di yolopit rispettati), pesi
   casuali, misurati interi sul PI (mediana di 3 passate). Sono le reti B4-R del masterplan. \<esempio>Meno di un'ora
   di macchina.\</esempio>
2. Per la rete `j` e la famiglia `f`: `x_jf = Σ LUT_f(c_in, c_out)` sui suoi strati.
3. Minimi quadrati con ridge verso 1: `min ‖y − C − X·a‖² + λ‖a − 1‖²`, con `a ≥ 0`; `λ` per cross-validation a 5
   fold. Forma chiusa: niente gradient descent (servirebbe solo per l'MLP sul residuo, L4-2).
4. Famiglie che variano sempre insieme (collineari) si uniscono in un solo `a`.
5. Verifica su reti tenute fuori dalla stima, più F, P e N interi.

**Ipotesi sulla composizione**: sul PI OpenVINO esegue i nodi uno alla volta e i 4 core lavorano dentro ogni nodo,
quindi i rami paralleli (C3k2, FPN) non si sovrappongono e sommare gli strati è corretto. I tensori dei salti che
restano in memoria occupano cache: lo assorbono solo in parte gli `a_f`.

**Alternativa più fedele, se gli `a_f` non bastano**: misurare la latenza degli strati dentro la rete con il profilo
per strato (perf counters di OpenVINO) sulle reti B4-R, così cache e parallelismo sono quelli reali.

## 8. Integrazione con PIT (da decidere)

- `pit.n` = 2, uguale al passo della LUT: ogni scelta discreta di PIT cade su un punto misurato.
- Per il gradiente la LUT va interpolata (lineare fra punti della griglia) sul numero di canali atteso dalle maschere
  morbide di PIT.
- In alternativa: search con la sola parte liscia (`C + Σ β·MAC`) e, dopo la search, ricerca locale strato per
  strato (qualche passo in più o in meno) verso il valore che costa meno secondo la LUT.

## 9. Prossimi passi

1. ~~Esperimento di separabilità~~: fatto, rifiutata (sezione 6).
2. LUT 2D a passo 2 per le 13 famiglie (circa 92 ore di macchina, in corso).
3. Generatore delle reti B4-R (nano potato a caso con i vincoli di yolopit) e loro misura.
4. Stima di `C`, `a_f` (ed eventualmente `γ`) e verifica.
5. Integrazione in yolopit come costo `latency` e confronto con la potatura guidata dai calcoli (S4-01 contro S4-02).
