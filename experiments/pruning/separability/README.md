# Separabilità della latenza di una conv sul Pi 5

La latenza di una conv si può scrivere come t(c_in, c_out) ≈ f(c_out)·g(c_in)? Se sì, il costo di
una rete potata si ricava da poche scansioni 1D, senza misurare tutte le coppie di canali.

## Setup

- Raspberry Pi 5 con OpenVINO 2026.4.1, fp16 (`-infer_precision f16`, l'ONNX resta fp32), 4 core,
  governor performance a 2,4 GHz. Configurazione e tuning della pipeline (`stage=probe_edge`,
  `src.measure.board.tuned`, `benchmark_app` costruito da `backend.build_cmd` e `_wrap_command`).
- Reti di una sola conv, con bias e senza attivazione:
  - famiglia A, conv 1×1 su 40×40;
  - famiglia B, conv 3×3 con padding 1 su 20×20.
  
  Più una rete identità per stimare il costo fisso.
- Punti misurati:
  - scansioni 1D da 1 a 128 su un asse, con l'altro asse a 32, 64 o 128;
  - griglia 18×18 su {8, 16, 24, 31, 32, 33, 40, 47, 48, 49, 63, 64, 65, 80, 96, 112, 127, 128}.
  
  In tutto 988 punti per famiglia.
- Ripetizioni: 3, con ordine mescolato a ogni ripetizione (seed 0). La latenza di un punto è la
  mediana delle 3. Il rumore è la mediana di (max − min)/mediana.
- Iterazioni di benchmark_app: 200, non le 30 di `probe_edge`. Con 30 la mediana veniva circa 2× il
  regime per il transitorio iniziale (64×64: 0,47 ms contro 0,21). Anche i vecchi probe
  `probe_edge` a 30 iterazioni ne sono probabilmente affetti.
- Guardia termica: oltre 75 °C, o con un bit di throttling attivo, lo script aspetta di tornare a
  65 °C e rimisura il punto.

## Comandi

```bash
ITERS=200 python run.py pilot 20   # tempo per punto su 20 punti, stima della durata
ITERS=200 python run.py run        # misura completa (~2,7 h), riprende da results/raw.csv
python analyze.py                  # results/{points.csv,summary.json,fails_c.csv}, figures/
python sizes.py                    # results/sizes.json (correlazione con le dimensioni)
python log_wandb.py                # run su wandb yolo-bench
```

Gli ONNX generati e il log della misura sono in `runs/`, ignorata da git.
Run su wandb: <https://wandb.ai/yolo-bench/yolo-bench/runs/6lhj71tw>

## Risultati (05/11/2026)

5928 misure in 162 minuti.

**Criteri della tabella.**
- Fit additivo: log t = a(c_in) + b(c_out) sulla griglia 18×18.
- Predizione 1D: t(c_in, 64)·t(64, c_out)/t(64, 64), escludendo i punti con un asse a 64.
- Verdetto: separabile se almeno il 90% dei punti ha un errore entro 2× il rumore.

| | A: conv 1×1, 40×40 | B: conv 3×3, 20×20 |
|---|---|---|
| Rumore | 3,4% | 2,2% |
| Costo dell'identità | 0,060 ms | 0,036 ms |
| Fit additivo: R² | 0,973 | 0,879 |
| Fit additivo: errore mediano / massimo | 3,3% / 35% | 8,0% / 54% |
| Predizione 1D: errore mediano / massimo | 3,8% / 64% | 14% / 99% |
| Punti con errore entro 2× il rumore | 71% | 34% |
| Verdetto | non separabile | non separabile |

Sottraendo il costo dell'identità le cose peggiorano: i punti entro 2× il rumore scendono al 47% in
A e al 32% in B.

**Dove fallisce.** Vedi `figures/heatmap_residui.png`, `figures/curve_1d_normalizzate.png` e
`results/fails_c.csv`.

- **B, c_in < 64.** Il tempo cresce a dente di sega con una pendenza che dipende da c_out, poi a
  c_in = 64 crolla. Con c_out = 128 passa da 0,39 ms (c_in 63) a 0,22 ms (c_in 64). Nella heatmap è
  il blocco c_in 16–63, quasi tutto fuori soglia. Il punto di cambio si sposta con c_out: circa 9
  per c_out = 64, circa 20 per c_out = 32.
- **Gradini su c_out.** Dopo 32 e dopo 96 il tempo sale (+15–50%). A 64 cambia la famiglia: in B
  scende (−24%) e a 65 risale. L'altezza dei gradini dipende da c_in: in B, a c_out = 97, +50% con
  c_in = 32 e +15–20% con 64 o 128.
- **A, c_out = 128.** Alcuni c_in vanno circa il doppio più veloci dei vicini: 17 misura 0,22 ms,
  18 misura 0,41 ms. È riproducibile nelle 3 ripetizioni e non segue i multipli di 8.
- **A, c_in ≤ 24 con c_out ≤ 32.** Il fit sovrastima del 30–60%.

### Riscaldamento (`analyze.py`, `summary.json` → `temp`)

- La frequenza è rimasta a 2,4 GHz in tutte le misure e il throttling non è mai scattato.
- 57 misure sono state rimisurate dalla guardia termica.
- Le temperature vanno da 58 a 75 °C, quasi tutte fra 65 e 75 °C.
- Scarto di ogni ripetizione dalla mediana del suo punto, in funzione della temperatura:
  - A: +0,03%/°C, r = 0,02;
  - B: +0,07%/°C, r = 0,04.
  
  Su 10 °C fanno al massimo lo 0,7%, contro un rumore del 2–3%.

Il riscaldamento non spiega i salti. Due limiti: la temperatura è letta subito dopo il benchmark, e
il caso a freddo, sotto 60 °C, ha solo 4 misure.

### Dimensioni di attivazioni e pesi (`sizes.py`, `results/sizes.json`)

Le dimensioni sono in KB a 2 byte per valore, senza padding né layout interno di OpenVINO.

Dentro una famiglia valgono queste proporzioni:
- input ∝ c_in;
- output ∝ c_out;
- pesi ∝ MAC ∝ c_in·c_out.

Una legge di potenza nelle tre dimensioni equivale quindi al modello separabile già testato.

| | A | B |
|---|---|---|
| Spearman con t: input / output / pesi | 0,62 / 0,71 / 0,91 | 0,63 / 0,64 / 0,90 |
| Spearman con t: in + out + pesi | 0,98 | 0,93 |
| Fit lineare t0 + a·in + b·out + c·w: errore mediano / massimo | 9,0% / 51% | 8,9% / 44% |
| Fit lineare: punti entro 2× il rumore | 37% | 24% |

Il fit lineare va peggio di quello additivo su log t.

I salti cadono a numeri fissi di canali, non a soglie di memoria:
- c_out = 33, 65 e 97 saltano allo stesso modo con input da 25 KB o da 400 KB, e con pesi da 19 KB
  o da 218 KB;
- in B il crollo a c_in = 64 avviene sia con pesi da 72 KB sia da 144 KB.

Nemmeno la cache spiega i salti:
- in A, 287 punti superano i 512 KB della L2 (in + out + pesi) e hanno lo stesso errore mediano
  degli altri (9%);
- in B nessun punto supera i 512 KB;
- le oscillazioni di A con c_out = 128 avvengono con input fra 38 e 91 KB, sotto la L1 da 64 KB.

L'ipotesi è che OpenVINO scelga il kernel in base ai canali, con blocchi di 32 canali di output e
un cambio di algoritmo a c_in = 64 per la 3×3. Si può verificare con i performance counter di
benchmark_app (`-pc`) su una decina di punti attorno ai salti. Non è stato fatto.
