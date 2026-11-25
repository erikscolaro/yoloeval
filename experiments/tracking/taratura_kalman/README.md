# Taratura del Kalman (senza crop)

**Scopo**: trovare i parametri del `KalmanTracker` (`src/tracking.py`) sui video di val di `data/tracking`, con
risultati separati per Svanström e Anti-UAV. Il tracker con crop usa lo stesso Kalman: questa taratura è il punto di
partenza per la sua.

**Comando unico**, dal dataset ai risultati:

```bash
bash experiments/tracking/taratura_kalman/rifai.sh [split=val] [prove=1000]
```

Servono `data/tracking/<split>` (da `../video_val_test`), i pesi di P- e S- in `weights/` e il venv con Optuna
(`scripts/requirements/x86_64.txt`). Una fase già fatta si salta: per rifarla, cancellare il suo file di uscita.

## Fasi

| Fase | Script | Uscita | Dove gira |
|---|---|---|---|
| 1. Detection | `detection.py` | `runs/detection_<split>.pkl` | GPU, circa 5 min |
| 2. R e Q dai dati | `rq.py` | `results/rq_<split>.json` | CPU, circa 40 s |
| 3. Optuna | `optuna_kalman.py` (F1), `optuna_kalman_f2.py` (F2 sui droni + vincolo) | `results/optuna_<studio>_<split>.json` e `.csv` (tutte le prove), `runs/optuna_<split>.db` | tutti i core, circa 15 min |
| 4. Valutazione sui video | `../valuta_tracker_video/run.py` | `results/valuta_<split>.txt` | GPU |

`comune.py` contiene le parti comuni: lettura delle detection, abbinamento con la verità, F1.

## Le scelte automatiche e perché

**1. Detection salvate una volta.** P- guarda il fotogramma intero, che non dipende dai parametri del Kalman. Le sue
detection si calcolano quindi una volta sola (conf ≥ 0,001, sotto la soglia più bassa che la ricerca prova) e ogni prova
le rilegge. Una prova dura così 3,5 s invece di minuti. Il fotogramma si riduce con INTER_AREA a 640 di lato lungo,
come sul Pi. Le detection di S- invece dipendono da dove il Kalman mette il ritaglio: non si possono salvare, e S- gira
dal vivo nella fase 4.

**2. R dagli scarti.** R è l'errore del modello sulla posizione. Il filtro lo modella proporzionale all'altezza della
box, quindi lo scarto fra detection e verità (IoU ≥ 0,5, stessa classe, conf ≥ 0,3) si divide per l'altezza vera. Si
usa la deviazione robusta (1,4826 × MAD), che non risente delle poche detection molto sbagliate. `r_pos` viene da
centro x e y, `r_size` da larghezza e altezza. Il file riporta R anche per classe, per decidere se servono valori
diversi.

**3. Q dal NIS.** Q è quanto il moto del bersaglio può cambiare da un fotogramma all'altro. Si provano 11 valori di
`q_pos` (da 1 a 1024, con `q_size` = `q_pos` / 2). Per ognuno il filtro segue l'oggetto vero con le sue detection, e a
ogni fotogramma si calcola il NIS: l'innovazione normalizzata con l'incertezza che il filtro si aspetta. Se il filtro è
coerente con i dati, il NIS segue una χ² con 4 gradi di libertà (cx, cy, w, h), e la sua media vale 4. Si sceglie il
`q_pos` con media più vicina a 4. Si usano solo i fotogrammi con un oggetto, per evitare scambi di identità.
R e Q così sono misure, non parametri da cercare: Optuna non li tocca.

**4. Optuna per il ciclo di vita del track.** Gli altri parametri non hanno una misura diretta: dipendono da quanto è
affidabile il modello e da quanto è prevedibile il moto. Optuna (TPE multivariato) prova una combinazione, misura il
risultato e usa le prove già fatte per scegliere la combinazione successiva. Il multivariato tiene conto delle
interazioni, per esempio fra `max_age` e `show_age`. Spazio di ricerca:

| Parametro | Intervallo | Significato |
|---|---|---|
| `hi` | 0,005–0,8 (log) | conf minima per far nascere un track |
| `lo` | 0,001–`hi` (log) | conf minima per aggiornare un track esistente |
| `min_hits` | 1–8 | detection prima di mostrare un track |
| `max_age` | 1–30 | fotogrammi senza detection prima di cancellare il track |
| `show_age` | 0–`max_age` | fotogrammi in cui si mostra un track solo predetto |
| `gate` | 2–200 (log) | soglia sulla distanza di Mahalanobis² per l'abbinamento |
| `v0` | 1–500 (log) | incertezza iniziale della velocità, in altezze di box al secondo |
| `iou_dup` | 0,2–0,9 | IoU oltre cui due detection sono lo stesso oggetto |
| `vote` | 0–0,99 | memoria del voto di classe |

La prima prova sono i parametri di partenza (default del codice con R e Q della fase 2), così il risultato non può
essere peggiore della partenza.

**5. L'obiettivo è l'F1.** Per ogni fotogramma i track mostrati si abbinano alla verità (IoU ≥ 0,5 e stessa classe).
Un oggetto vero senza track è un mancato, un track senza oggetto è un falso allarme. L'F1 pesa i due errori in modo
uguale, al punto di lavoro reale del tracker, cioè con le soglie scelte e non su tutte le soglie come la mAP. Si usa
l'F1 su tutti i fotogrammi del gruppo, non la media per sequenza: le sequenze lunghe pesano di più.

**6. Tre studi.** Svanström e Anti-UAV hanno camere, risoluzioni e moti diversi. Uno studio per fonte mostra quanto
cambiano i parametri migliori. Lo studio `tutte` cerca un solo set per le due fonti insieme: è quello da usare se i due
studi per fonte non danno un vantaggio chiaro.

**7. Confronto onesto.** Per ogni studio il file riporta anche il detector da solo alla conf con F1 massimo e il Kalman
con i parametri di partenza. Il Kalman è utile solo se batte il detector alla sua conf migliore.

**8. Tutte le risorse.** La fase 3 usa tutti i core: più processi per studio condividono le prove tramite il database
SQLite. Il Kalman non usa la GPU. La fase 4 lancia i tre studi in parallelo sulla GPU, perché un processo solo la
occupa intorno al 10%.

## Limiti

- Le prove in parallelo rendono la ricerca non ripetibile prova per prova: due lanci danno punti migliori vicini, non
  identici.
- Con 500 prove il punto migliore si adatta un po' al val. Il test si usa una volta sola, alla fine.
- R e Q sono unici per fonte. Il file `rq` riporta R per classe: se le differenze contano, servono parametri per classe
  (oggi il codice ne ha un solo set).

## Risultati (val, corsa del 23–24/11/2026, intervalli ampi, 1.000 prove)

| Studio | F1: detector → Kalman | F2 droni: detector → Kalman | Aerei/elicotteri chiamati drone (F2) |
|---|---|---|---|
| Svanström | 0,568 → 0,580 | 0,428 → 0,422 | 3,2% → 2,6% |
| Anti-UAV | 0,822 → 0,854 | 0,776 → 0,823 | 0 → 0 |
| tutte e due | 0,697 → 0,711 | 0,666 → 0,702 | 3,2% → 2,6% |

- Rispetto alla prima corsa: F2 di `tutte` da 0,687 a 0,702; su Svanström il Kalman da solo non batte più il
  detector con l'F2 (0,422 contro 0,428).
- F2 `tutte`: `hi` 0,045, `lo` 0,001, `min_hits` 1, `max_age` 5, `show_age` 0, `gate` 194, `v0` 466. `lo` e `gate`
  sono al limite della ricerca; `max_age` scende da 8–10 a 5.
- Sui video (`results/valuta_val.txt`) la fase 4 usa i parametri degli studi F1. Con lo studio `tutte` il crop resta
  sopra il detector: totale mAP50-95 0,322 contro 0,299. I parametri tarati insieme al crop sono in
  `../taratura_kalman_crop`.

## Risultati (val, 23/11/2026, prima corsa)

**Attenzione**: questi numeri vengono dalla prima corsa, con intervalli più stretti (`hi` da 0,03, `lo` da 0,01,
`max_age` fino a 20, `gate` fino a 50, 500 prove) e detection salvate da 0,01. I file sono in `runs/vecchi_23nov/`.
La corsa del 23–24/11 usa gli intervalli della tabella sopra e 1.000 prove: i risultati nuovi sostituiranno questi.


**R e Q** (`results/rq_val.json`): r_pos 0,036 / 0,047, r_size 0,141 / 0,120 (Svanström / Anti-UAV), circa 3 volte i
vecchi default per le dimensioni. Q: `q_pos` 16 per Svanström, 64 per Anti-UAV e per le due insieme (default 4).

**Optuna** (500 prove per studio, `results/kalman_f1_<studio>_val.json` e `.csv`): F1 del Kalman tarato contro il
detector alla sua conf migliore e il Kalman di partenza.

| Studio | detector | Kalman di partenza | Kalman tarato |
|---|---|---|---|
| Svanström | 0,568 | 0,530 | 0,578 |
| Anti-UAV | 0,822 | 0,712 | 0,846 |
| tutte e due | 0,697 | 0,634 | 0,710 |

- Convergenza: dopo 100 prove il guadagno è sotto 0,01; le 10 prove migliori stanno entro 0,0015 di F1.
- Su Svanström P- trova meno della metà degli oggetti a fotogramma intero (recall 0,47 con la classe giusta).
- Con l'F1 per fotogramma Optuna sceglie `min_hits` = 1, `show_age` = 0 e `max_age` 1–3: mostrare solo le detection
  vere e mai le predizioni. Il Kalman resta quasi un filtro di classe e di box: l'obiettivo non premia la continuità
  del track. Per il tracker con crop la taratura va fatta sul tracker intero (`../taratura_kalman_crop`).
- Segnali di limite: `gate` spesso vicino a 50 (limite superiore), `lo` di Anti-UAV a 0,01 (detection salvate da 0,01).

**Obiettivo della torretta** (`optuna_kalman_f2.py`, `results/kalman_f2_<studio>_val.json` e `.csv`, 500 prove per
studio): F2 sui droni con il vincolo che aerei ed elicotteri chiamati drone non superino il detector da solo. Studio
`tutte`:

| | F2 droni | Recall | Precisione | Aerei/elicotteri chiamati drone |
|---|---|---|---|---|
| detector da solo (conf 0,02) | 0,666 | 0,647 | 0,755 | 3,2% |
| Kalman di partenza | 0,573 | 0,533 | 0,821 | 2,4% |
| Kalman tarato con F2 | 0,687 | 0,656 | 0,844 | 2,5% |

Per fonte (studi separati): Svanström 0,433 contro 0,428 del detector, Anti-UAV 0,808 contro 0,776.
- Con l'F2 `max_age` sale a 8–10 (con l'F1 era 1–3): il track sopravvive agli scatti e mantiene l'ID, e la spia
  "DRONE" di `tools/predict_video.py` si accende. Restano `min_hits` = 1 e `show_age` = 0.
- `hi` (0,033) e `lo` (0,01) sono al limite basso della ricerca e delle detection salvate: per scendere ancora serve
  rifare `detection.py` con una soglia più bassa.

**Sui video** (`results/valuta_val.txt`, parametri dello studio `tutte`, detector alla sua conf migliore 0,2, un crop):

| Fonte | Modo | mAP50 | mAP50-95 | Recall | Falsi/fotogramma |
|---|---|---|---|---|---|
| Svanström | detector | 0,521 | 0,262 | 0,603 | 0,097 |
| | Kalman | 0,521 | 0,261 | 0,639 | 0,078 |
| | crop | 0,572 | 0,293 | 0,676 | 0,061 |
| Anti-UAV | detector | 0,739 | 0,414 | 0,697 | 0,013 |
| | Kalman | 0,721 | 0,406 | 0,723 | 0,014 |
| | crop | 0,702 | 0,401 | 0,727 | 0,005 |
| Totale | detector | 0,580 | 0,299 | 0,653 | 0,049 |
| | Kalman | 0,578 | 0,297 | 0,684 | 0,041 |
| | crop | 0,623 | 0,323 | 0,703 | 0,029 |

Recall, falsi allarmi e persi sono per fotogramma, IoU >= 0,5, senza classe; la mAP conta la classe. Il crop è in
TRACK nel 66% dei fotogrammi. Il Kalman da solo alza un po' il recall e lascia la mAP uguale; il crop con S- dà il
guadagno, soprattutto su Svanström. Su Anti-UAV il crop dimezza i falsi allarmi ma perde un po' di mAP: il drone fa
scatti e il crop lo perde.
