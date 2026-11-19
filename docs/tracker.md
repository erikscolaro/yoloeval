# Tracker con predittore di Kalman

> **In breve.** Un filtro di Kalman per ogni oggetto predice dove sarà nel fotogramma successivo, e YOLO guarda solo
> un ritaglio attorno a quella predizione invece del fotogramma intero. Sul Pi 5, con P- e un oggetto agganciato, il
> fotogramma passa da 26 ms (640 intero) a circa 6–7 ms. Il fotogramma intero serve solo ogni N fotogrammi, quindi
> diventa possibile il deploy a 1080p (207 ms a fotogramma intero) restando nel budget di 33 ms.

**Stato**: piano, non avviato (19/11/2026). Viene dalla discussione "Kalman predictor per detection" del 28/09/2026,
che usava uno scenario marino (barche, gabbiani); qui è riportato al nostro dominio, oggetti in cielo.

## 1. Il filtro

- **Stato** per ogni track: `[cx, cy, w, h, vx, vy, vw, vh]`, modello a velocità costante. YOLO è il sensore, il
  modello di moto è il predittore.
- **Ciclo** a ogni fotogramma: predict (stato avanti, incertezza `P` cresce) → associazione con le detection (IoU,
  algoritmo ungherese) → update se abbinato (`P` cala), altrimenti solo predict (il track "scivola"); dopo
  `max_age` fotogrammi senza detection il track si cancella.
- **R, rumore di misura** (jitter di YOLO): proporzionale alla box, circa `σ = h/20` sulla posizione (come DeepSORT e
  ByteTrack). Si misura su un oggetto fermo o, meglio, come `cov(box_YOLO − box_GT)` su video annotati. Con il crop
  va scalata: `σ_frame ≈ σ_crop × roi/imgsz`.
- **Q, rumore di processo**: si sceglie un solo numero, l'accelerazione tipica `σ_a`, e la matrice segue dal modello
  white-noise acceleration, per ogni asse:

  ```
  Q(dt) = σ_a² · | dt⁴/4  dt³/2 |
                 | dt³/2  dt²   |
  ```

  `σ_a` dai dati: `std(diff(diff(x)))` sulle traiettorie GT (con le box di YOLO al posto della GT,
  `σ_a² ≈ var(a) − 6·σ_R²`). Anche `σ_a` proporzionale alla box (circa `h/160` sulla velocità).
- **Per classe**: `σ_a`, R, `P₀` sulla velocità e `max_age` diversi. Conta la manovrabilità, non la velocità:

  | classe | velocità | manovrabilità (`σ_a`) |
  |---|---|---|
  | aereo | alta | bassa (traiettorie lisce) |
  | elicottero | media | medio-alta (si ferma, gira) |
  | uccello | media | alta |
  | drone | media | alta (cambi bruschi) |

  La classe del track si decide a voto sulla sua storia, oppure `Q = Σ p(classe) · Q_classe`. Se i track si perdono
  nelle manovre, il passo successivo è l'IMM (2–3 modelli in parallelo: dritto, in manovra, in virata).
- **Tempo**: `dt` reale dai timestamp di acquisizione (non di fine inferenza), velocità in px/s e `σ_a` in px/s²
  (`σ_a[px/s²] = σ_a[px/frame²] × fps²`). Il framerate variabile va bene: F(dt) e Q(dt) si ricalcolano a ogni passo.
  Con dt grande l'incertezza cresce come dt³, la ROI esplode: oltre una certa frazione del fotogramma si torna allo
  scan. Le misure vanno applicate in ordine di tempo.
- **Verifica della taratura**: l'innovazione (misura − predizione) normalizzata, il NIS, deve valere in media circa 4
  (le variabili misurate).

## 2. Il crop guidato dal predict

- **ROI** = box predetta + margine dall'incertezza, `k = 3` (3σ):

  ```
  roi_w = w + 2·k·σx + k·σw        roi_h = h + 2·k·σy + k·σh        (σ dalla diagonale di P predetta)
  ```

  più almeno il 20% di contesto. Track stabile: ROI stretta, crop piccolo. Track nuovo, in manovra o con fotogrammi
  persi: ROI larga.
- **Crop**: dimensione fissa multipla di 32 (192–256), con l'oggetto al 50–70% del crop (un oggetto che riempie il
  crop è fuori distribuzione, le box ai bordi sono rare nel training). Riduzione con `cv2.INTER_AREA` (con
  `INTER_LINEAR` aliasing). ROI fuori dal fotogramma: padding grigio 114, come il letterbox. Riproiezione:
  `x_frame = roi.x0 + (x_crop − pad_x) / scala`.
- **Stati**:

  | stato | ingresso | quando |
  |---|---|---|
  | SCAN | fotogramma intero, alta risoluzione | nessun track confermato |
  | TRACK | crop dal Kalman | track confermato: `hits` consecutivi (es. 5), confidenza media, traccia di `P` sotto soglia |
  | RE-SCAN | fotogramma intero | ogni N fotogrammi, o subito al primo miss |
  | LOST | torna a SCAN | miss oltre `max_age` |

  Isteresi fra gli stati, per non oscillare.
- **Più oggetti**: un crop per track, in batch. Il costo cresce con i pixel: conviene finché `N × imgsz²` resta sotto
  circa il 60–70% del costo del fotogramma intero, poi si torna al fotogramma intero. ROI vicine si uniscono; dopo la
  riproiezione un NMS globale toglie i doppioni. Su CPU del Pi il batch non dà parallelismo: il costo è lineare.

## 3. Oggetti nuovi e piccoli

Finché si è agganciati, un oggetto nuovo si vede solo al re-scan. Per ridurre il ritardo:

- **Variazione di entropia per blocco** rispetto a una baseline adattiva: EMA di `H` e della sua deviazione per ogni
  blocco, candidato se `z = |H_t − μ_b|/σ_b > 3` per 2–3 fotogrammi. Robusta ai cambi di luce uniformi; blocchi 2–4
  volte l'oggetto più piccolo, sovrapposti a metà; i blocchi già coperti dai track si mascherano. Variante più
  sensibile alla forma: entropia dell'istogramma dei gradienti.
- **Priorità ai bordi** (gli oggetti entrano soprattutto da lì), ma non da soli: un anello di tile da 128 in 1080p è
  il 34% del fotogramma, e lo spessore deve superare `v_max × dt`.
- **Scan a tasselli a rotazione** come rete di sicurezza per gli oggetti lenti o fermi: un tassello diverso a ogni
  fotogramma, costo costante.
- **Oggetti piccoli all'orizzonte**: fascia ad alta risoluzione con upsample 2×, candidati da filtri classici
  (top-hat, Difference of Gaussians, Local Contrast Measure, con CLAHE prima), conferma M di N con Q bassa, classe
  generica finché l'oggetto è piccolo. Più avanti, track-before-detect.

## 4. I modelli

Un modello unico per i tre regimi (scan, crop di tracking, fascia dell'orizzonte), con augmentation che li
riproducono (crop attorno all'oggetto con resize, oggetti piccoli ingranditi), valutato separatamente per regime.
Specializzare solo dove serve: il tracking è il compito più facile e quello che gira più spesso, quindi è dove una
rete potata piccola rende di più; l'orizzonte è il più difficile (eventualmente testa P2).

## Livelli di lavoro

| livello | cosa | difficoltà | tempo |
|---|---|---|---|
| A | ByteTrack o BoT-SORT di Ultralytics nel predict: ID e scie, nessun crop | 1/10 | mezz'ora |
| B | il nostro Kalman con il crop guidato e gli stati SCAN/TRACK/RE-SCAN/LOST, uno o più oggetti | 4/10 | 1–2 giorni |
| C | taratura di Q e R per classe dai dati, validazione con metriche di tracking (HOTA, MOTA) | 6/10 | qualche giorno, servono video annotati |
| D | entropia, fascia dell'orizzonte, IMM | 8/10 | settimane |

Il punto di partenza proposto è B, nel predict (`tools/predict_video.py`), prima sui video di `predict/` e poi sul Pi.

## Latenza sul Pi 5

P-, OpenVINO fp16, 4 core a 2,4 GHz, mediane delle misure in `results/` (`deploy=crop*` e risoluzioni intere):

| ingresso del modello | P- |
|---|---|
| crop 128 | 3,0 ms |
| crop 256 | 5,6 ms |
| crop 512 | 16,6 ms |
| 640 intero | 25,8 ms |
| 720p intero | 63,9 ms |
| 1080p intero | 206,7 ms |

Il tracker aggiunge circa 0,1 ms per track (Kalman 8×8 in numpy) e circa 1 ms di ritaglio e resize: stime, da
misurare.

- **Un oggetto, crop 256**: circa 6–7 ms a fotogramma, contro 26 del 640 intero.
- **Più oggetti**: circa 6 ms in più per oggetto; oltre 3–4 oggetti conviene il fotogramma intero.
- **Re-scan**: crop 256 con un re-scan a 640 ogni 10 fotogrammi fa in media circa 9 ms; con un re-scan a 1080p ogni
  30 fotogrammi circa 13–14 ms.
- **Shape fisse**: OpenVINO vuole un modello compilato per ogni dimensione di crop, quindi 1–2 dimensioni fisse.

## Cosa otteniamo

1. **Deploy ad alta risoluzione nel budget**: il 1080p solo per il re-scan, il resto sui crop a risoluzione nativa.
   Si vedono i droni lontani senza pagare 207 ms a fotogramma.
2. **Oggetti piccoli trovati meglio**: nel crop l'oggetto resta grande in pixel come nel training (prova del 19/11:
   con il crop a 640 P- trova un drone piccolo a conf 0,46, che a fotogramma intero si vedeva appena).
3. **Identità e traiettoria**: ID, velocità e posizione predetta per ogni oggetto, cioè quello che serve per puntare.
4. **Box e classe più stabili**: jitter filtrato, classe a voto sulla storia del track, il track sopravvive a qualche
   detection mancata.

**Prezzi**: gli oggetti nuovi si vedono fino a N fotogrammi dopo (lo riduce il livello D); la mAP non basta più, servono
metriche di tracking su video annotati; una o due dimensioni di crop fisse.
