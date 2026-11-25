# Tracker con predittore di Kalman

> **In breve.** Un filtro di Kalman per ogni oggetto predice dove sarà nel fotogramma successivo, e YOLO guarda solo
> un ritaglio attorno a quella predizione invece del fotogramma intero. Sul Pi 5, con P- e un oggetto agganciato, il
> fotogramma passa da 26 ms (640 intero) a circa 6–7 ms. Il fotogramma intero serve solo ogni N fotogrammi, quindi
> diventa possibile il deploy a 1080p (207 ms a fotogramma intero) restando nel budget di 33 ms.

**Stato** (25/11/2026): fatto e tarato sul val. Codice in `src/tracking.py` e `src/crop_tracker.py`; taratura e
confronti in `experiments/tracking/` (`taratura_kalman`, `taratura_kalman_crop`, `confronto_p_s_crop`). Con un crop
256 di S- l'F2 sui droni è 0,718 sul totale (Svanström 0,461, Anti-UAV 0,837), contro 0,666 di P- da solo. Cose da
fare in `docs/todo.md`. Piano iniziale del 19/11. Viene dalla discussione "Kalman predictor per detection" del 28/09/2026,
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
  Nessun ingrandimento digitale (`max_up = 1`, 23/11/2026): un oggetto piccolo si taglia a scala nativa, senza
  interpolare. Lo zoom digitale aggiunge pochi dettagli (`experiments/tracking/crop_zoom_map`: a 16–32 px guadagna 0,06 di
  mAP50 con P- e niente con S-) e non è uno zoom ottico.
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

### Scelta dei modelli, un track (23/11/2026)

- **Regola**: il modello sta nel budget a 30 fps nel caso peggiore, usando come p99 il massimo di ogni passata sul Pi
  (`benchmark_app` non dà il p99). Il budget è 33,3 ms meno il costo fuori dal modello nel caso peggiore.
- **SCAN: P-** (mAP50-95 0,570, val 640), un solo file per tutte le camere: l'IR si porta alla forma della camera
  all'avvio (`model.reshape` di OpenVINO). È l'unico modello nel budget con tutte e tre le camere:
  - Svanström 640×512 nativo: 20,5 ms, massimo 25,3 ms, budget circa 31,1 ms (`experiments/latenza/scan_modelli_640x512`);
  - camera HD → 640×384 con AREA: 15,6 ms, budget 30,4 ms;
  - camera FHD → 640×384 con AREA: 15,6 ms, massimo 23,9 ms, budget 23,9 ms (margine nullo, il massimo è un picco isolato).
  PN- (0,587) non sta nel budget a 640×512 (32,8 ms) né con camera FHD e AREA
  (`experiments/latenza/scan_modelli_per_camera_e_ingresso`).
- **Ingresso dello SCAN alla risoluzione della camera**: schiacciare 640×512 a 640×384 costa 0,064–0,074 di mAP50-95, il
  letterbox 0,029, soprattutto sugli oggetti da 8 a 32 px (`experiments/tracking/scan_map_per_ingresso`).
- **TRACK: S-** (mAP50-95 0,613), crop 256², batch 1: 24,4 ms, massimo 25,2 ms, budget 31,2 ms. Con la regola del pari
  merito vincerebbe NS- (16,1 ms, 0,608): si tiene S- per la mAP, che nel budget ha 6 ms di margine.
- **Un solo crop** (`max_crops` = 1, 23/11/2026): con 2 crop S- costa 44,4 ms contro un budget di 31,4 ms
  (`experiments/latenza/track_modelli_batch_crop256`). Con 2 o più track confermati il fotogramma va in SCAN. Nei
  video di val e test quasi tutti i fotogrammi hanno un oggetto; nessuno sciame.
- **Due modelli in memoria**: 212 MB; il cambio SCAN ↔ TRACK costa 0,1–0,2 ms; la prima inferenza va fatta all'avvio
  (`experiments/latenza/modelli_carico_e_cambio`).
- **Non verificato**: la mAP sui crop 256² e sui video.

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

Fuori dal modello, un fotogramma TRACK costa circa 0,93 ms per crop (ritaglio, Kalman, pre e post), misurato in
`experiments/latenza/track_budget_n_crop`: con N crop il budget del modello a 30 fps è 32,3 / 31,4 / 29,6 / 25,9 ms
(N = 1 / 2 / 4 / 8).

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

## Limiti del prototipo: cosa serve per un sistema production grade

Il tracker è un prototipo: i tempi di oggi bastano per la tesi e non si ottimizzano. Per portarlo a un sistema
production grade restano questi punti (20/11/2026).

- **Codice fuori dal modello in Python.** Un fotogramma TRACK costa circa 0,93 ms per crop fuori dal modello
  (`experiments/latenza/track_budget_n_crop`), quasi tutto overhead di Python su matrici piccole. Stima, non misurata:
  in C circa 0,15–0,25 ms per crop, cioè circa 5–6 ms in più di budget con 8 crop. Senza C: pre-processing dentro il
  modello OpenVINO e Kalman vettoriale su tutti i track.
- **Ritaglio misurato sul crop più piccolo.** Il budget per N crop usa crop vicini al minimo (128 → 256). Con oggetti
  grandi il ritaglio costa fino a 3,8 ms per crop (720 → 256 con INTER_AREA, `experiments/latenza/track_costo_ritaglio`),
  e il budget TRACK è più stretto.
- **Nessun lato massimo del crop.** Con un oggetto molto grande o un'incertezza alta il crop può crescere senza limite.
- **Interpolazione: INTER_AREA ovunque** (scelta dell'utente, 23/11/2026). È una scelta conservativa, per preservare la
  qualità dell'immagine, e i dati la sostengono. Nel SCAN con LINEAR i droni di 16–64 px perdono 5–8 punti di mAP50
  (`experiments/tracking/crop_interpolazione_map`, riduzione 3× da 1080p); nel crop AREA e LINEAR danno la stessa mAP
  agli zoom usati, ma AREA è più sicura con riduzioni forti. Costo a 720p con un track: SCAN 2,8 ms con AREA contro 3,0 ms
  con LINEAR; crop da 1,0 a circa 2,0 ms, meno di 1 ms in più rispetto a LINEAR. A 1080p lo SCAN con AREA costa 11,5 ms
  contro 2,9 ms. Nel codice `CropTracker.fit` riduce il fotogramma SCAN con AREA (prima lo faceva il letterbox di
  Ultralytics con LINEAR) e `cut` usa AREA quando il crop si riduce.
- **Budget dello SCAN su video.** Con un video 720p o 1080p, oltre a pre e post, va tolto il resize a 640.
- **Precisione nei crop.** Il Pareto per batch (`experiments/latenza/track_modelli_batch_crop256`) usa la mAP del val a 640, non nei
  crop; M non è addestrata.
- **Riavvii del Pi.** Il Pi si è riavviato due volte sotto carico con batch 8 (SM- e SM+); la causa non è nota.
- **Coda della latenza.** `benchmark_app` riporta solo la mediana: per dimensionare il margine del crop serve il caso
  peggiore, oggi visto solo dal massimo di ogni passata.
- **Taratura del Kalman.** Non fatta: la procedura è quella della sezione "Livelli di lavoro", livello C. Il costo
  del Kalman non dipende dai parametri, quindi i tempi di oggi restano validi dopo la taratura.

## Sviluppi possibili

Idee valutate e non implementate (23/11/2026):

- **Lati del crop fissi.** Portare il lato calcolato a 256, 384 o 512 px (il modello riceve sempre 256×256, un solo
  export). Il resize con AREA costa 0,06 ms a 512 e circa 1,3 ms a 384. La perdita stimata è al massimo 0,03 di
  mAP50 con P- per gli oggetti di 154–307 px (da `experiments/tracking/crop_zoom_map`, interpolata, non misurata).
- **Resize AREA più veloce** (`experiments/latenza/track_costo_ritaglio`, sezione "Perché AREA costa di più"). AREA è
  veloce solo con fattore 2 (SIMD) e 4; con gli altri fattori costa da 1 a 9 ms e cresce con i pixel. Tre rimedi:
  (a) lato vicino a 256 o 512 portato a quel valore (tolleranza 10%, oggetto cambiato di meno del 10%);
  (b) due passi, AREA 2× esatto e poi LINEAR (0,2–0,3 ms fino a 768 px, qualità non misurata);
  (c) i thread non servono (l'uscita 256×256 è una sola striscia di lavoro). Il guadagno è di circa 1–2 ms con un
  track e di circa 4 ms con 4 crop; con un track non serve.
- **Crop rettangolari.** Un oggetto allungato occupa meglio un crop 256×128, con meno latenza. Servono un export per
  forma, una forma sola per batch e la mAP rettangolare, che non è misurata.
- **Oggetti molto grandi.** Oltre `max_side` il fotogramma è uno SCAN (oggi 512 px, non tarato). Si potrebbe ridurre
  l'oggetto nel crop, con costo di resize e grigio da misurare.
- **Ingresso 128×128 per i ritagli piccoli** (opzione, da valutare dopo la taratura con il solo 256). Due modelli
  compilati: ritaglio fino a 128 px → ingresso 128 (1:1); da 129 a 256 px → ritaglio allargato a 256 (1:1); da 257 a 512
  px → riduzione a 256 con AREA. Con un bersaglio piccolo e stabile S- costa 10,2 ms invece di 24,4 ms: cala la latenza
  media, non il caso peggiore. Da misurare: la mAP su ritagli 128×128 e le tracce perse (poco margine per l'errore del
  Kalman). Nel codice basta una lista di ingressi (128, 256) al posto di `crop`.
- **Zoom digitale.** Escluso: aggiunge pochi dettagli (vedi sopra). Resta lo zoom ottico.

- **Zoom ottico reale.** Una camera con zoom ottico (PTZ) aumenta i pixel dell'oggetto, cosa che lo zoom digitale
  non fa. Il predittore di Kalman dà posizione e dimensione attese dell'oggetto: può comandare il brandeggio e lo
  zoom della camera. Così il tracker mantiene l'oggetto a una grandezza adatta al modello (circa 96–160 px, vedi
  `experiments/tracking/crop_zoom_map`). Non è stato progettato né misurato.
- **Ripresa dell'ID dopo una perdita.** Oggi un track perso per più di `max_age` fotogrammi rinasce con un ID nuovo:
  la spia "DRONE" riparte da zero, e i cambi di ID non sono misurati. Idea: se un track nuovo nasce entro X
  fotogrammi nella zona prevista dal Kalman per quello perso (incertezza allargata) e con dimensione simile, eredita
  il vecchio ID; una verifica dell'aspetto conferma che è lo stesso oggetto. Per la verifica basta un hash percettivo
  (pHash o dHash, distanza di Hamming) calcolato sulla box e non sul crop intero, dove il cielo domina; costa
  pochissimo e si fa solo alla rinascita. Sotto i 20 px l'aspetto distingue poco (un drone lontano somiglia a un
  uccello), e conta di più la geometria. DeepSORT confronta l'aspetto a ogni abbinamento, con una rete di
  re-identificazione: sul Pi costerebbe budget.
- **Moto proprio della camera.** Nella torretta la camera si muove per seguire il bersaglio; in Anti-UAV i
  riorientamenti del pan-tilt spostano il drone di 100–130 px in 4–5 fotogrammi e il Kalman a velocità costante resta
  indietro (Q 4 volte più alta che su Svanström). Gli angoli pan/tilt della torretta, passati al Kalman come ingresso
  di controllo, compenserebbero questo moto. Non si può valutare qui: Svanström e Anti-UAV non hanno gli angoli della
  camera.
