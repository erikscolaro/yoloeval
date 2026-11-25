# Taratura del tracker con crop

**Scopo**: trovare insieme i parametri del Kalman e del crop per il tracker del deploy (`src/crop_tracker.py`): SCAN
con P- sul fotogramma intero, un crop 256×256 con S- in TRACK. Risultati separati per Svanström e Anti-UAV.

**Comando unico**:

```bash
bash experiments/tracking/taratura_kalman_crop/rifai.sh [split=val] [prove=1000] [processi=12]
```

Usa i dati di `../taratura_kalman` (detection di P-, R e Q, parametri del Kalman da solo con F2): se mancano, li fa.

## Fasi

| Fase | Script | Uscita | Dove gira |
|---|---|---|---|
| 0. Dati del Kalman da solo | `../taratura_kalman/*.py` | `../taratura_kalman/{runs,results}/` | GPU e CPU |
| 1. R e confidenza di S- | `rs.py` | `results/rs_<split>.json` | GPU, circa 1 min |
| 2. Optuna | `optuna_crop.py` | `results/kalman_crop_<obiettivo>_<split>.json` e `.csv`, `runs/kalman_crop_<obiettivo>_<split>.db` | GPU, più processi |
| 3. Valutazione sui video | `../valuta_tracker_video/run.py` | `results/valuta_<split>.txt` | GPU |

`base.py` contiene le parti comuni: dati, percorsi delle immagini, fotogramma decodificato solo quando serve.

## Le scelte automatiche e perché

**1. Tarare tutto insieme.** La taratura del Kalman da solo, con l'F1 per fotogramma, sceglie `min_hits` = 1,
`show_age` = 0 e `max_age` 1–3: non premia la continuità del track. Sul tracker con crop la continuità si paga da sé:
se il track resta vivo, il crop resta sul bersaglio e S- lo vede anche quando P- no. Per questo Kalman e crop si
cercano insieme, sul tracker vero.

**2. Un solo crop** (`max_crops` = 1). Con 2 crop S- costa 44,4 ms sul Pi, oltre il budget di 31,4 ms
(`../../latenza/track_modelli_batch_crop256`). Nei video quasi tutti i fotogrammi hanno un oggetto.

**3. Lo SCAN dalle detection salvate.** In SCAN il fotogramma si riduce a 640 come in `../taratura_kalman/detection.py`,
quindi le detection di P- sono le stesse e non serve rifarle. S- gira dal vivo, perché il ritaglio dipende dai
parametri. Il fotogramma si decodifica solo in TRACK: in SCAN non serve. La decodifica di un fotogramma 1920×1080 costa
6,4 ms, più di S-; una cache di tutti i fotogrammi in RAM non ci sta (112 GB).

**4. R di S- dai ritagli, non cercata.** `rs.py` ritaglia intorno a ogni oggetto vero (un fotogramma su 5), con il lato
e lo spostamento che il tracker usa, e misura lo scarto come `../taratura_kalman/rq.py`. Q resta quella di P-: dipende
dal moto, non dal modello.

**5. Soglie separate per il TRACK.** S- sui ritagli ha un'altra scala di confidenza: la sua soglia con F1 massimo sui
droni è 0,01, contro 0,2 di P-. `CropTracker` usa nei fotogrammi TRACK `hi_track`, `lo_track` e la R di S-; il ciclo di
vita del track resta unico.

**6. L'obiettivo è l'F2 sui droni.** La torretta deve trovare i droni: un drone perso è peggio di un falso allarme, e
l'F2 pesa il recall il doppio della precisione. Contano solo i droni: un track drone su un drone vero (IoU ≥ 0,5) è
giusto, un track drone senza drone è un falso allarme, un drone senza track drone è perso. Gli errori fra uccelli,
aerei ed elicotteri non contano.

**7. Il vincolo su aerei ed elicotteri.** Colpire un aereo o un elicottero è l'errore peggiore. Si conta la quota di
aerei ed elicotteri veri con un track drone sopra (IoU ≥ 0,5); non deve superare quella del detector da solo, cioè P-
alla sua soglia con F2 massimo, sugli stessi fotogrammi. È un vincolo relativo: nel val ci sono solo 18 sequenze con
aerei o elicotteri, troppo poche per un tetto assoluto. Le prove oltre il vincolo restano nel database, ma Optuna non le
sceglie.

**8. Spazio di ricerca** (ampio, per una corsa di circa 20 ore). Kalman: `hi` 0,005–0,8, `lo` 0,001–`hi`,
`min_hits` 1–8, `max_age` 1–30, `show_age` 0–`max_age`, `gate` 2–200, `v0` 1–500, `iou_dup` 0,2–0,9, `vote` 0–0,99.
TRACK: `hi_track` 0,001–0,8, `lo_track` 0,001–`hi_track`. Crop: `k` 0,5–6, `ctx` 0–0,8, `fill` 0,2–0,95, `rescan`
0–120 a passi di 1 (0: mai, resta il re-scan dopo un mancato), `max_side` 256–512 a passi di 16 (oltre 512 il costo
della riduzione non è misurato). Le detection di P- sono salvate da conf 0,001. La prima prova sono i parametri del
Kalman da solo tarato con F2 (stesso obiettivo), con i default del crop. 1.000 prove.

**9. Più processi sulla GPU.** Una prova sul val completo usa poco la GPU: più processi lavorano sullo stesso studio e
condividono le prove nel database. Il numero di processi viene da una prova di velocità (vedi Risultati).

## Limiti

- Le prove in parallelo rendono la ricerca non ripetibile prova per prova.
- Un solo studio per le due fonti insieme, perché ogni prova costa minuti. I risultati restano separati per fonte.
- Il vincolo relativo dipende dal detector sugli stessi pochi aerei: va riletto sul test.
- La valutazione finale (fase 3) conta la classe nella mAP, ma recall e falsi allarmi senza classe (come
  `../valuta_tracker_video`); l'obiettivo di Optuna invece conta solo i droni.

## Risultati

**Prova di velocità** (23/11/2026, workstation con 24 core e GPU da 49 GB; 600 fotogrammi per processo, metà Anti-UAV e
metà Svanström; in parallelo girava un'altra ricerca sulla CPU a priorità bassa):

| Processi | Fotogrammi/s in totale | Una prova sul val (28.400 fotogrammi) ogni |
|---|---|---|
| 2 | 194 | 146 s |
| 4 | 275 | 103 s |
| 8 | 445 | 64 s |
| 12 | 475 | 60 s |
| 16 | 510 | 56 s |

Oltre 8 processi il guadagno è piccolo: il limite è la CPU (decodifica dei JPEG e Python del tracker), non la GPU. Si
usano 12 processi: con più prove in parallelo il TPE sceglie le prove successive con meno informazione.

**Ricerca** (val, corsa del 23–24/11/2026: 1.000 prove, 12 processi, circa 8 ore; `results/kalman_crop_f2_val.json`
e `.csv`). F2 sui droni con il vincolo su aerei ed elicotteri; detector da solo a conf 0,02:

| Fonte | detector F2 | tracker con crop F2 | Recall | Precisione | Aerei/elicotteri chiamati drone |
|---|---|---|---|---|---|
| Svanström | 0,428 | 0,461 | 0,415 | 0,817 | 0,13% (detector 3,2%) |
| Anti-UAV | 0,775 | 0,837 | 0,842 | 0,817 | 0 (nessun aereo) |
| Totale | 0,666 | 0,718 | 0,697 | 0,817 | 0,13% (detector 3,2%) |

Il Kalman da solo con F2 (`../taratura_kalman`, stessa corsa) arriva a 0,702 sul totale. Parametri della prova 805:
`hi` 0,050, `lo` 0,001, `min_hits` 1, `max_age` 10, `show_age` 1, `gate` 197, `v0` 357, `iou_dup` 0,21, `vote` 0,51,
`hi_track` 0,20, `lo_track` 0,18, `k` 2,6, `ctx` 0,09, `fill` 0,67, `rescan` 60, `max_side` 464. TRACK nel 68% dei
fotogrammi.
- Al limite della ricerca: `gate` (200), `lo` (0,001, soglia delle detection salvate) e `iou_dup` (0,2). Un `gate`
  così alto vuol dire abbinamento quasi libero: con un bersaglio solo basta.
- `hi_track` (0,20) è molto sopra la conf migliore di S- da sola (0,01): nel crop conviene far nascere track solo
  con detection sicure.

**Sui video** (`results/valuta_val.txt`, detector a conf 0,02, un crop; recall e falsi senza classe, la mAP conta la
classe):

| Fonte | Modo | mAP50-95 | Recall | Falsi/fotogramma |
|---|---|---|---|---|
| Svanström | detector | 0,265 | 0,700 | 0,433 |
| | Kalman | 0,258 | 0,716 | 0,322 |
| | crop | 0,296 | 0,734 | 0,260 |
| Anti-UAV | detector | 0,451 | 0,783 | 0,266 |
| | Kalman | 0,443 | 0,848 | 0,324 |
| | crop | 0,450 | 0,848 | 0,187 |
| Totale | detector | 0,309 | 0,744 | 0,338 |
| | Kalman | 0,301 | 0,787 | 0,323 |
| | crop | 0,333 | 0,794 | 0,218 |
| Totale 1 oggetto | crop (detector) | 0,334 (0,310) | 0,804 (0,749) | 0,199 (0,324) |
| Totale 2+ oggetti | crop (detector) | 0,237 (0,234) | 0,716 (0,705) | 0,562 (0,586) |

Il crop dà il guadagno sui fotogrammi con un oggetto (il caso del deploy): +0,024 di mAP50-95, +5 punti di recall,
falsi allarmi −39%. Con 2+ oggetti è pari al detector: in TRACK solo il 19% dei fotogrammi.

Il confronto con S- da solo e con i crop 128 e 64 (stessi parametri) è in `../confronto_p_s_crop`.
