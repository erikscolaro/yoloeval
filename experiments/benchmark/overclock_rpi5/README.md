# Overclock del Raspberry Pi 5

**Scopo**: capire quanta latenza si guadagna alzando la frequenza della CPU (fino a 3,0 GHz) e della GPU (fino a
1000 MHz), con OpenVINO fp16 sui modelli P e N standard. Esplorativo: i numeri restano fuori dal Pareto e dal
confronto fra modelli, che sono a 2,4 GHz.

**Stato**: chiuso il 16/11/2026.

## Setup

- **Modelli**: P `59d8534a` e N `9dca8d15` (D126), gli stessi di `experiments/deploy/ncnn_rpi5`.
- **Misura**: pipeline (`stage=benchmark`, `target=rpi5_640`), OpenVINO fp16, 4 core, governor performance,
  3 ripetizioni, latenza media. `hardware.freq.max.khz` segue la frequenza provata.
- **Frequenze**: ogni configurazione si carica con `tryboot`. `tryboot.txt` è `config.txt` più `arm_freq`,
  `gpu_freq` e `over_voltage_delta`, e vale per un solo avvio, quindi `config.txt` resta intatto. Se il Pi non
  ripartisse, basterebbe togliere e ridare corrente.
- **Tensione**: `over_voltage_delta` a gradini fissi, presi dalle guide comuni e non tarati su questo Pi: 0 fino a
  2600 MHz, +25 mV fino a 2800, +50 mV a 2900 e 3000 (`shortcut:` in `rifai.sh`).
- **Controlli per ogni configurazione** (`results/clock.csv`): `cpuinfo_max_freq` uguale alla frequenza chiesta
  (altrimenti la configurazione non si misura), clock del core, tensione del core, `get_throttled` prima e dopo.
- **Ordine**: A (CPU da 2400 a 3000 a passi di 100, GPU di serie), poi CPU e GPU al massimo (3000/1000). Se la GPU
  sposta la latenza meno dell'1%, le scansioni della GPU (B a CPU 2400, C a CPU 3000) si saltano. Infine D:
  NCNN fp16 a CPU 3000.

```bash
systemd-run --user --unit=overclock-rpi5 --working-directory=$PWD \
  bash -c "flock ~/Desktop/yoloeval/runs/.pi.lock experiments/benchmark/overclock_rpi5/rifai.sh"
.venv/bin/python experiments/benchmark/overclock_rpi5/run.py      # results/tabella.csv
```

## Risultati

OpenVINO fp16, latenza media sulle 3 ripetizioni; Δ rispetto a 2400 MHz con GPU di serie.

| CPU (MHz) | GPU (MHz) | tensione core | P (ms) | Δ P | N (ms) | Δ N |
|---|---|---|---|---|---|---|
| 2400 | 910 (di serie) | 0,879 V | 27,62 | rif. | 56,62 | rif. |
| 2500 | 910 | 0,894 V | 26,98 | -2,3% | 54,82 | -3,2% |
| 2600 | 910 | 0,908 V | 25,91 | -6,2% | 53,30 | -5,9% |
| 2700 | 910 | 0,947 V | 25,17 | -8,9% | 51,76 | -8,6% |
| 2800 | 910 | 0,961 V | 24,63 | -10,8% | 50,27 | -11,2% |
| 2900 | 910 | 0,999 V | 23,72 | -14,1% | 48,88 | -13,7% |
| **3000** | 910 | 1,000 V | **23,34** | **-15,5%** | **47,58** | **-16,0%** |
| 3000 | 1000 | 1,000 V | 23,11 | -16,3% | 47,67 | -15,8% |
| 2400 | 500 | 0,881 V | 27,90 | +1,0% | 57,12 | +0,9% |

NCNN fp16 (prova D), confrontato con le sue misure a 2400 MHz (`experiments/deploy/ncnn_rpi5`):

| backend | CPU (MHz) | P (ms) | N (ms) |
|---|---|---|---|
| NCNN | 2400 | 25,40 | 57,65 |
| NCNN | 3000 | 23,11 (-9,0%) | 54,41 (-5,6%) |
| OpenVINO | 3000 | 23,34 | 47,58 |

Nessun throttling in nessuna configurazione (`get_throttled` 0x0 prima e dopo ogni misura), temperatura massima
a fine misura 52,4 °C (`temp_end_c` nei json). Il Pi è ripartito a ogni frequenza fino a 3000 MHz.

## Commento

- **La CPU conta, la GPU no.** Da 2400 a 3000 MHz la latenza scende del 16% su entrambi i modelli. Con la CPU a
  3000, portare la GPU da 910 a 1000 MHz cambia meno dell'1%; con la CPU a 2400, scenderla a 500 MHz costa l'1%,
  dentro il rumore. L'inferenza gira tutta sulla CPU.
- **La latenza scende meno del clock**: +25% di clock dà -16% di latenza. Circa un terzo del tempo non scala con la
  CPU, probabilmente accesso alla memoria (la frequenza della SDRAM sul Pi 5 non si cambia).
- **Fra 2900 e 3000 il guadagno quasi si ferma** (-0,4 ms su P, -1,3 ms su N) e la tensione è già a 1,000 V.
- **NCNN scala molto peggio di OpenVINO**: -9% su P e -6% su N per lo stesso +25% di clock. A 3000 MHz NCNN e
  OpenVINO sono pari su P, e su N NCNN resta indietro di 7 ms. Il vantaggio di NCNN su P a 2400 MHz sparisce.
  Ipotesi non verificata: NCNN è più limitato dalla memoria, oppure a frequenze alte usa meno bene i core.

## Limiti

- La frequenza effettiva è quella di `cpufreq` (chiesta al kernel), non il clock del firmware sotto carico
  (`vcgencmd measure_clock arm`); il calo di latenza che segue il clock dice che è applicata davvero.
- Tensioni prese dalle guide e stabilità non provata con un carico lungo: le misure durano pochi minuti per
  configurazione. Per un uso continuo a 3000 MHz servirebbe una prova di stabilità di ore.
- Scansioni della GPU saltate per la regola dell'1%: misurati solo 3000/1000 e 2400/500.
- In `results/clock.csv` c'è una riga "non applicata" per 3000/1000: il primo tentativo è partito mentre il Pi stava
  ancora riavviando; il secondo è andato.
