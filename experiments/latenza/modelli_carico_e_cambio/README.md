# Caricamento e cambio di modello sul Raspberry Pi 5

**Scopo**: quanto costa caricare i due modelli del tracker e passare dall'uno all'altro, per il budget (SCAN ↔ TRACK).
**Stato**: misurato il 23/11/2026.

## Setup

- **Modelli**: SCAN = PN- (`yolo26w0.1875_D101`) a 640×384; TRACK = S- (`yolo26w0.5_D101`) a 256×256, batch 1. Pesi
  casuali, OpenVINO 2026.4.1 fp16, 4 thread, un solo stream, governor performance, `chrt -f 80`, lock del Pi.
- **`misura.py`** (API Python di OpenVINO, gira sul Pi), una riga JSON per misura:
  - `carico`: `read_model`, `compile_model`, prima inferenza, inferenza a regime (100 chiamate), RSS. Un processo nuovo
    per ogni misura: 5 a freddo (cache dei file svuotata), 5 a caldo, 5 con la cache di OpenVINO (1 miss, 4 hit).
  - `cambio`: i due modelli compilati nello stesso processo. Tempo di ogni chiamata: modelli da soli, alternati a
    ogni fotogramma, e nel ciclo reale (1 SCAN ogni 30 fotogrammi). Il ciclo c'è anche con 16 MB toccati prima di ogni
    chiamata, per sporcare la cache come fa il resto della pipeline. 3 passate.
- **`rifai.sh`**: copia i modelli sul Pi, fa tutte le misure in `results/carico_cambio.jsonl` (circa 5 minuti).

```bash
experiments/latenza/modelli_carico_e_cambio/rifai.sh
```

## Risultati (mediana delle ripetizioni)

**Caricamento**

| modello | read_model | compile_model | prima inferenza | a regime |
|---|---|---|---|---|
| PN- (SCAN), a freddo | 126 ms | 885 ms | 29,0 ms | 25,9 ms |
| PN- (SCAN), a caldo | 16 ms | 244 ms | 27,6 ms | 25,9 ms |
| PN- (SCAN), cache OpenVINO | 16 ms | 129 ms | 27,0 ms | 25,7 ms |
| S- (TRACK), a freddo | 235 ms | 1.003 ms | 46,1 ms | 25,2 ms |
| S- (TRACK), a caldo | 17 ms | 260 ms | 45,8 ms | 25,2 ms |
| S- (TRACK), cache OpenVINO | 16 ms | 150 ms | 45,5 ms | 25,3 ms |

- Tutto a freddo (avvio del programma): circa 2,3 s per i due modelli. Con la cache di OpenVINO e i file in memoria: circa 0,3 s.
- La prima inferenza di S- costa 20 ms in più del regime (inizializzazione): va fatta all'avvio, fuori dal ciclo.
- Memoria (RSS): 53 MB di base, 136 MB con PN-, 174 MB con S-, **212 MB con tutti e due**. Il Pi ha 8 GB.

**Cambio** (modelli già compilati nello stesso processo), mediana / p99 / massimo, ms

| chiamata | da solo | alternato ogni fotogramma | nel ciclo | nel ciclo, cache sporca |
|---|---|---|---|---|
| SCAN | 25,6 / 25,9 / 31,7 | 25,7 / 26,0 / 28,5 | 25,7 / 25,9 / 25,9 | 25,7 / 25,9 / 26,0 |
| TRACK dopo uno SCAN | — | — | 25,1 / 25,3 / 25,3 | 25,1 / 25,3 / 25,3 |
| TRACK (altri) | 24,9 / 25,2 / 25,2 | 25,1 / 25,4 / 26,8 | 24,9 / 25,2 / 25,4 | 24,9 / 25,1 / 25,2 |

(valori della prima passata; le altre due differiscono di meno di 0,2 ms, salvo qualche massimo isolato fino a 31,7 ms.)

## Commento

- **Il cambio non costa**: con i due modelli compilati e tenuti in memoria, alternarli aggiunge 0,1–0,2 ms per chiamata. Sporcare
  la cache non cambia il risultato. Il budget non ha bisogno di una voce per il cambio.
- **Non conviene scaricare e ricaricare un modello**: costerebbe 260–280 ms a caldo (circa 8 fotogrammi) più 20 ms di
  prima inferenza per S-.
- **L'API Python costa circa 1 ms in più di `benchmark_app`**: PN- 25,9 ms contro 24,3 ms, S- 25,2 ms contro 24,4 ms
  (invio dell'ingresso e copia dell'uscita, con una sola richiesta sincrona).
- **Massimi**: in 2 passate su 3 lo SCAN da solo ha un massimo di 31,3–31,7 ms (una chiamata su 300), sopra il budget
  del caso peggiore (30,4 ms). Il p99 è 26 ms.

## Limiti

- Pesi casuali: i tempi di caricamento dipendono dalla forma, non dai valori.
- La cache sporcata con 16 MB è un'approssimazione del lavoro fra due chiamate (ritaglio, Kalman).
- Non ho misurato il carico mentre il resto della pipeline usa il Pi, né il batch di più crop.
