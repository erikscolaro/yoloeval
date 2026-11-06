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

## Reti (`generate.py`)

- **Modello:** nano YOLO26 (larghezza 0,25, D126, 4 classi), costruito da `src.models.arch` come lo
  stadio `train`, con pesi casuali: la latenza non dipende dai pesi.
- **Potatura con i vincoli di yolopit:** `build_pit_model(n=2)`, con lo stesso passo della LUT. Le maschere
  a blocchi di yolopit contengono già tutti i vincoli: maschere condivise fra residui e concat, uscite
  congelate per gli strati che alimentano C2PSA, PSABlock e Detect, C3k2 diviso in `cv1a` e `cv1b`, un
  blocco sempre tenuto. Le maschere potabili sono 55.
- **Casualità:** per ogni rete un livello g ~ U(0,1; 1). Per ogni maschera la frazione tenuta è
  g · lognormale(0; 0,5), limitata fra un blocco e tutti. Le reti coprono così tutto l'intervallo, da
  circa il 10% al nano intero, e le famiglie variano in modo indipendente fra loro, come richiede il fit
  degli a_f. Il seme della rete j è j.
- **150 reti più il nano intero** (`full`, riferimento).
- **Export:** `OpenVINOBackend.export`, lo stesso dello stadio `export`: IR fp16, testa one-to-one
  (`nms=False`), 640×640, batch 1. Gli IR stanno in `runs/nets/<rete>/`, ignorata da git, insieme al `.pt`.
- **Strati:** letti con forward hook sul modello esportato, sono le conv fuori da C2PSA, PSABlock e
  Detect, cioè **74 per rete** (i 66 strati della LUT, con gli 8 `cv1` dei C3k2 divisi in due). Ogni
  (famiglia, c_in, c_out) deve cadere nella griglia della LUT: lo script si ferma se non succede. Nelle
  prime prove tutti i punti ci cadono.

## Misura (`measure.py`)

Condizioni e protocollo della LUT: stessa preparazione della board, stesso comando (benchmark_app tramite
`bench_dump.py`, warmup di 60 ms, `-infer_precision f16`, 4 core, 2,4 GHz) e stessa guardia termica.
L'ingresso è l'immagine fp32, con un solo Convert f32 → f16, come nel deploy. **50 iterazioni** per misura:
una rete intera dura decine di ms, quindi 50 inferenze vanno già ben oltre i circa 20 ms di warmup. 3
ripetizioni, ordine mescolato con seme 0, ripresa da `results/raw.csv` dopo un'interruzione.

Durata: da stimare con `measure.py r000 r075 full` (5 misure × 3 ripetizioni, `results/check.csv`).

## Fit e verifica (`fit.py`)

- y_j = mediana delle ripetizioni; x_jf = somma dei valori della LUT dei suoi strati della famiglia f.
- Ridge verso 1 con a_f ≥ 0: `min ‖y − C − X·a‖² + λ‖a − 1‖²`, minimi quadrati vincolati (`lsq_linear`). λ
  si sceglie con una cross-validation a 5 fold sulle reti di training. Le famiglie con correlazione > 0,98
  sulle reti di training condividono un solo a. Test in `test_fit.py`.
- **Verifica:** il 20% delle reti casuali (seme 0) e il nano intero restano fuori dal fit.
- **Confronto** sulle stesse reti con due stimatori più semplici: *somma LUT + C* (a_f = 1, si stima solo
  C) e *MAC* (C + β·MAC, con i MAC di yolopit).

## Comandi

```bash
python generate.py 150                       # ~15 min sulla workstation: runs/nets/, results/nets.csv, layers.csv
python figures.py nets                       # cosa è stato generato
ITERS=50 python measure.py r000 r075 full    # prova breve e stima della durata
systemd-run --user --unit=b4r --working-directory=$PWD bash -c 'ITERS=50 ../../../.venv/bin/python measure.py > runs/measure.log 2>&1'
python fit.py                                # dopo la LUT: results/fit.json, predictions.csv
python figures.py fit
```

## Limiti

- **La LUT deve essere completa:** `fit.py` legge `../lut_pi_passo2/results/lut_<famiglia>.csv`.
- **Nessun γ** per concat, add e upsample: se il residuo cresce con i canali, va aggiunto.
- **Estensione decisa il 06/11:** il dataset diventerà le potature casuali di tutte le architetture fino al
  nano (larghezze 0,0625–0,25 × D101, D126, D170), con il test diviso per architettura e C tarato per
  architettura sulla rete intera non potata (C2PSA, PSABlock e Detect non si potano). Larghezze 0,375, S e
  oltre solo in futuro: prima va estesa la LUT ai loro canali (fino a 4× i punti).
- **Stessa architettura di partenza:** le reti sono tutte potature del nano D126; per F, P e N
  interi servono misure a parte.

## Percorsi

| percorso | contenuto |
|---|---|
| `generate.py`, `measure.py`, `fit.py`, `figures.py` | generazione, misura, fit, figure |
| `test_fit.py` | test del fit (C e a_f ritrovati, a_f ≥ 0, ridge verso 1, famiglie collineari) |
| `results/nets.csv` | una riga per rete: seme, livello, strati, MAC e parametri (yolopit) |
| `results/layers.csv` | una riga per conv di ogni rete: famiglia, c_in, c_out |
| `results/raw.csv`, `results/env.json` | misure e ambiente |
| `results/fit.json`, `results/predictions.csv` | parametri, errori, stima per rete |
| `figures/nets.png`, `figures/fit.png` | reti generate; stima contro misura, a_f, confronto |
| `runs/` | IR e `.pt` delle reti, log, latenze per iterazione (ignorati da git) |
