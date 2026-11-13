# B4-R: composizione della LUT nella latenza del nano potato sul Pi 5

Questo esperimento stima i parametri del modello che passa dalla LUT di latenza per strato
([`lut_pi_passo2`](../lut_pi_passo2)) alla latenza della rete intera, e ne verifica l'errore. Sono le reti
B4-R del masterplan (`NEW_MASTERPLAN.md`, sezione 11, 4a, «Dalla LUT alla latenza della rete»).

```
latenza_rete = C + Σ_f a_f · Σ_{strati in f} LUT_f(c_in, c_out)
```

- **C** comprende tutto quello che la LUT non vede: C2PSA, PSABlock e Detect (non potati), i nodi che non
  sono conv, e il costo fisso per chiamata, che la somma della LUT conta 74 volte
  ([`DECISIONE_IDENTITA.md`](../lut_pi_passo2/DECISIONE_IDENTITA.md)).
- **a_f** è la correzione della famiglia f dentro la rete: cache, layout e fusioni sono diversi dallo
  strato misurato da solo.

**Stato (13/11):** chiuso. 765 reti generate e misurate (× 3), fit fatto (`results/fit.json`, λ = 1). Sulle 750 reti
potate delle architetture tenute fuori l'errore relativo è mediano 0,9% e p95 3,0%, contro 2,9% e 10,2% della somma
della LUT + C e 7,2% e 26,9% dei MAC. La procedura (generazione, misura, fit) ora sta in `src/lut` ed è la stessa di
`scripts/build_lut.sh`; `./rifai.sh` la ripete con i parametri di allora. `fit.json` e le LUT sono quelli di
`yolopit/data/rpi5` (identici, `scripts/sync_yolopit_lut.sh`).

## Reti (`src/lut/b4r.py`, allora `generate.py`)

- **Architetture:** 15, tutte quelle fino al nano: larghezze 0,0625, 0,09375, 0,125, 0,1875 e 0,25 × profondità
  D101, D126 e D170 (il nano è w0.25_D126). YOLO26 con 4 classi, costruito da `src.models.arch` come lo stadio
  `train`, con pesi casuali: la latenza non dipende dai pesi.
- **Potatura con i vincoli di yolopit:** `build_pit_model(n=2)`, con lo stesso passo della LUT. Le maschere
  a blocchi di yolopit contengono già tutti i vincoli: maschere condivise fra residui e concat, uscite
  congelate per gli strati che alimentano C2PSA, PSABlock e Detect, C3k2 diviso in `cv1a` e `cv1b`, un
  blocco sempre tenuto. Le maschere potabili sono 55.
- **Casualità:** per ogni rete un livello g ~ U(0,1; 1). Per ogni maschera la frazione tenuta è
  g · lognormale(0; 0,5), limitata fra un blocco e tutti. Le reti coprono così tutto l'intervallo, da
  circa il 10% alla rete intera, e le famiglie variano in modo indipendente fra loro, come richiede il fit
  degli a_f. Il seme della rete j è j.
- **50 reti per architettura più la rete intera** (`<arch>_full`, per tarare C): 765 reti. Le 151 della prima
  versione, solo nano (`full`, `r000`…), restano in `runs/nets/` ma non si misurano.
- **Export:** `OpenVINOBackend.export`, lo stesso dello stadio `export`: IR fp16, testa one-to-one
  (`nms=False`), 640×640, batch 1. Gli IR stanno in `runs/nets/<rete>/`, ignorata da git, insieme al `.pt`.
- **Strati:** letti con forward hook sul modello esportato, sono le conv fuori da C2PSA, PSABlock e
  Detect, cioè **74 per rete** (i 66 strati della LUT, con gli 8 `cv1` dei C3k2 divisi in due). Ogni
  (famiglia, c_in, c_out) deve cadere nella griglia della LUT: lo script si ferma se non succede. Nelle
  prime prove tutti i punti ci cadono.

## Misura (`src/lut/pi.py b4r`, allora `measure.py`)

Condizioni e protocollo della LUT: stessa preparazione della board, stesso comando (benchmark_app tramite
`bench_dump.py`, warmup di 60 ms, `-infer_precision f16`, 4 core, 2,4 GHz) e stessa guardia termica.
L'ingresso è l'immagine fp32, con un solo Convert f32 → f16, come nel deploy. **50 iterazioni** per misura:
una rete intera dura decine di ms, quindi 50 inferenze vanno già ben oltre i circa 20 ms di warmup. 3
ripetizioni, ordine mescolato con seme 0, ripresa da `results/raw.csv` dopo un'interruzione.

Durata: stimata con `measure.py r000 r075 full` (5 misure × 3 ripetizioni, `results/check.csv`; ora
`python -m src.lut.pi b4r <dir> r000 r075 full`).

## Fit e verifica (`src/lut/fit.py`, allora `fit.py`)

- y_j = mediana delle ripetizioni; x_jf = somma dei valori della LUT dei suoi strati della famiglia f.
- **Un C per architettura** (larghezza × profondità): C2PSA, PSABlock, Detect e i nodi che non sono conv non si
  potano, quindi quello che la LUT non vede cambia con l'architettura e non con la potatura. Gli a_f sono comuni a
  tutte le architetture.
- Ridge verso 1 con a_f ≥ 0: `min ‖y − C_arch − X·a‖² + λ‖a − 1‖²`, minimi quadrati vincolati (`lsq_linear`). λ si
  sceglie con una cross-validation a 5 fold **per architettura**: con fold di reti casuali λ usciva 0 e gli a_f
  andavano da 0 a 9, perché le reti della stessa architettura si somigliano e il fit le impara a memoria. Le
  famiglie con correlazione > 0,98 condividono un solo a. Test in `tests/test_lut.py`.
- **Verifica, leave-one-architecture-out (annidata):** a turno un'architettura resta fuori, il fit (λ compreso) si
  fa sulle altre, il suo C si tara sulla sua rete intera (`C = y_full − x_full·a`) e si stimano le sue 50 reti
  potate. È l'uso reale: un'architettura nuova costa una misura della rete intera, poi lo stimatore dà la latenza
  di ogni sua potatura.
- **Confronto** con la stessa divisione e la stessa taratura di C: *somma LUT + C* (a_f = 1) e *MAC*
  (C_arch + β·MAC, con i MAC di yolopit).

## Comandi

```bash
./rifai.sh --dry-run                         # passi e conteggi, senza toccare il Pi
systemd-run --user --unit=b4r --working-directory=$PWD bash -c './rifai.sh > runs/measure.log 2>&1'
                                             # genera (se manca results/nets.csv), misura, fit
./rifai.sh /altra/dir                        # in un'altra cartella, senza toccare questa
python figures.py nets; python figures.py fit
```

## Limiti

- **La LUT deve essere completa:** il fit legge `../lut_pi_passo2/results/lut_<famiglia>.csv`.
- **Nessun γ** per concat, add e upsample: se il residuo cresce con i canali, va aggiunto.
- **Solo fino al nano:** le architetture sono le larghezze 0,0625–0,25 × D101, D126, D170. Larghezze 0,375, S e
  oltre solo in futuro: prima va estesa la LUT ai loro canali (fino a 4× i punti).

## Percorsi

| percorso | contenuto |
|---|---|
| `rifai.sh` | generazione, misura e fit con `src/lut` (`scripts/build_lut.sh`, passo `b4r`); allora `generate.py`, `measure.py`, `fit.py` (storia git) |
| `figures.py` | figure |
| `results/nets.csv` | una riga per rete: seme, livello, strati, MAC e parametri (yolopit) |
| `results/layers.csv` | una riga per conv di ogni rete: famiglia, c_in, c_out |
| `results/raw.csv`, `results/env.json` | misure e ambiente |
| `results/fit.json`, `results/predictions.csv` | parametri, errori, stima per rete |
| `figures/nets.png`, `figures/fit.png` | reti generate; stima contro misura, a_f, confronto |
| `runs/` | IR e `.pt` delle reti, log, latenze per iterazione (ignorati da git) |
