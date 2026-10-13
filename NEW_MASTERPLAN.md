# NEW MASTERPLAN — detection di oggetti aerei su dispositivi edge, guidata dall'hardware

**Convenzione.** Il testo fra `<esempio>` e `</esempio>` è un'**ipotesi**, un'attesa o un esempio con numeri
inventati: non è ancora verificato. Tutto il resto è una **decisione presa** o un **fatto misurato**; per i numeri
misurati è indicato dove (per esempio "sul Pi", "sulla RTX 6000"). Le stime di tempo sono sempre indicate come tali.

## 1. L'idea in breve

Abbiamo tre dispositivi edge molto diversi fra loro:

- un **Raspberry Pi 5**, che calcola con la sua CPU;
- una **Jetson AGX Orin**, che calcola con una GPU;
- un **Raspberry Pi 5 con l'acceleratore DX-M1**, che calcola con un NPU collegato via PCIe.

Su ciascuno vogliamo far girare un detector per quattro classi (**drone, aereo, elicottero, uccello**) che regga
**30 fotogrammi al secondo**. La tesi risponde a due domande:

1. **Su ogni dispositivo, qual è il detector più preciso che sta nei 30 fps?**
2. **Dove il budget è stretto, conviene prendere un modello grande e potarlo, oppure basta usare fin dall'inizio un
   modello più piccolo?**

Quello che abbiamo già visto con le prove sulla pipeline attuale suggerisce che la risposta cambia da dispositivo a
dispositivo: sulla CPU del Pi la latenza segue i calcoli; sulla GPU della workstation (RTX 6000) segue soprattutto il
numero di strati e l'allineamento dei canali; \<esempio>sull'NPU probabilmente la granularità con cui elabora i
canali\</esempio>. Il messaggio atteso della tesi è quindi:

> \<esempio>*"Su ogni dispositivo edge conta una cosa diversa: ecco quale, e quando la potatura conviene davvero."*\</esempio>

---

### Struttura: due blocchi

La tesi è divisa in due blocchi, per non rischiare di finire il tempo prima di avere un risultato completo.

- **Blocco 1 — il nucleo (circa 7 settimane)**: fasi 0, 1 (versione di base), 2 e 3, con le due incognite X1 e X2
  risolte subito prima della fase che ne ha bisogno. Risponde alla
  domanda 1: *su ogni dispositivo e a ogni risoluzione di deploy, qual è la ricetta migliore (larghezza, profondità,
  precisione) e perché*. **Da solo è già una tesi completa.**
- **Blocco 2 — la potatura (circa 4 settimane, se c'è tempo)**: fase 4, con lo **stimatore di latenza** e la potatura
  guidata sia dai calcoli sia dallo stimatore. Risponde alla domanda 2.

Il test (fase 5) si apre **una volta sola, alla fine di tutto**, sui modelli che esistono in quel momento. Si fissa una
**data limite**: se a quella data il blocco 2 non è finito, il test si fa senza i potati. Nei momenti in cui la GPU è
libera durante il blocco 1 si allenano già con gli altri seed i modelli da cui partirebbe la potatura (2e): se il
blocco 2 si fa sono pronti, se non si fa restano punti confermati del fronte.

### Le fasi in una tabella: cosa testiamo, che conoscenza produciamo, come la usiamo

| fase | cosa testiamo | conoscenza che produciamo | come la usiamo dopo |
|---|---|---|---|
| 0 | se i dati sono adatti | dataset custom a 640×512 (Svanström più aggiunte) con **box corretti** e bilanciato su classe × fascia di grandezza, split pulito; set di val alle risoluzioni di deploy | il dataset e la ricetta valgono per tutta la tesi; i set di val di deploy danno la mAP usata nella scelta dell'ottimo (2c) |
| X1 (prima della fase 1) | se le nostre varianti girano sulla DX-M1 | operatori supportati; tempo di compilazione | decide se DX resta nel piano e quanto costano le misure sulla DX |
| 1 (base) | quanto sono ripetibili le misure; quanto dei 33 ms resta al modello; come si comporta ogni dispositivo | la **soglia di rumore** della latenza; il **budget del modello**; i tratti di base di ogni dispositivo (gradini, costo per strato, granularità N) | la soglia dice quando una differenza di latenza è vera; il budget è il vincolo dell'ottimo (2c) e della decisione (2d); i tratti spiegano i trend della fase 2 e sono il punto di partenza dello stimatore (4a) |
| X2 (prima della fase 2, **risolta**) | se l'int8 funziona sulla testa one-to-one, e come va sul PI | causa della mAP 0 (coda della testa quantizzata); ricetta int8; prova preliminare: sul PI l'int8 non è più veloce dell'fp16 | sul PI: quali nodi quantizzare nella 2a e nella fase 3, quale software usare per l'int8 sul PI; l'attesa da verificare in 2a e fase 3 |
| 2 (2a–2b) | quanto sono precisi e veloci i modelli normali | i **trend**: mAP e latenza in funzione di larghezza e profondità, a ogni risoluzione di deploy, per dispositivo | materia prima della fase 2c |
| **2 (2c)** | **qual è la configurazione migliore nel budget** | **tasso di cambio** di ogni leva (mAP per ms), un **modello di precisione**, l'**ottimo predetto e verificato** per dispositivo e risoluzione di deploy | l'ottimo è la **baseline** da battere; la leva più redditizia dice **su cosa deve lavorare la potatura**; il gap fra ottimo e modello migliore senza budget dice **se la potatura serve** (2d) |
| 3 | quanto rende l'int8 | perdita di mAP e guadagno di latenza dell'int8, per modello e dispositivo | si ricalcola l'ottimo (2c) con le latenze int8; chiude il blocco 1; si sceglie la precisione della fase 4 |
| 4a (blocco 2) | se una formula predice la latenza di una rete potata | lo **stimatore di latenza** per dispositivo e precisione, con il suo errore; i suoi parametri descrivono il dispositivo | è il costo che guida la potatura in 4b |
| 4b (blocco 2) | se potare batte lo scaling ottimizzato | quanto rende la potatura rispetto all'ottimo scalato; se conta la forma o i pesi ereditati; se lo stimatore rende più dei calcoli | sceglie il modello finale per dispositivo e il metodo da raccomandare |
| 5 (alla fine di tutto) | se i risultati reggono su dati mai visti | lo scarto fra val e test; la robustezza a bassa potenza; la mAP per classe e per grandezza del box | i numeri e le conclusioni della tesi; un esempio di distanza massima di rilevamento |


---

## 2. Regole fisse

Queste regole valgono per tutta la tesi. Servono a rendere i confronti onesti e a non doverli rifare.

### Dataset

- **Un solo dataset custom**, con le quattro classi, usato per tutta la tesi. Si crea **quando il piano è finito**,
  seguendo la fase 0. **Tutte le immagini sono 640×512, come Svanström**: non ci sono fonti ad alta risoluzione.
  **Non è solo Svanström**: è Svanström più alcune aggiunte che ne coprono i buchi (dettagli in fase 0, ricerca delle
  fonti in `docs/ricerca_dataset.md`).
  - **Base: Svanström** (285 video visibili, 640×512, ripresi da terra verso il cielo). È l'unico dataset trovato con
    **tutte e quattro le classi etichettate negli stessi video e con la stessa camera**: ogni classe vi compare,
    quindi nessuna classe viene da una fonte sola e il modello non può riconoscere la fonte invece dell'oggetto.
  - **Aggiunte**: fonti scelte quando si crea il dataset, portate a 640×512 ed etichettate con la stessa regola.
    AOD4 non si usa.
- Diviso in train / val / test **per video o per scena**, non per singola immagine: fotogrammi vicini dello stesso
  video sono quasi identici, e se finiscono uno nel train e uno nel val le mAP risultano gonfiate (è il problema
  già incontrato con AOD4).
- Le immagini quasi duplicate si eliminano con l'hash semantico: dentro ogni video (fase 0, passo 6) e fra split
  diversi dopo lo split (passo 7).
- **Bilanciato fra le classi e fra le fasce di grandezza** (16–32, 32–64, 64–96, sopra 96 px): un minimo di istanze
  e di video distinti per ogni cella classe × fascia (fase 0).
- **Box aderenti all'oggetto**, con la stessa regola per tutte le fonti. I box originali di Svanström sono larghi e
  incoerenti (un box di 16 px contiene un oggetto di 4–6 px) e **si correggono** in fase 0, prima di qualunque
  conteggio o training.
- **Il test si apre una volta sola**, alla fine (fase 5). Tutte le scelte si fanno guardando il val.

### Budget di latenza

- Il budget è **30 fps, cioè 33 ms end-to-end**, uguale per tutti i dispositivi.
- **End-to-end** vuol dire: dal fotogramma uint8 **già alla risoluzione di deploy** fino ai box finali. Comprende:
  - il pre-processing (uint8 → float, divisione per 255, cambio di layout);
  - i trasferimenti verso l'acceleratore (verso la GPU sulla Jetson, sul PCIe per la DX-M1);
  - l'inferenza;
  - il post-processing (decodifica della testa, soglia di confidenza, box in coordinate dell'immagine).
- **Restano fuori** l'acquisizione dalla camera e il ridimensionamento: si assume che la camera lavori già alla
  risoluzione di deploy.
- Pre-processing, post-processing e trasferimenti **non dipendono dal modello** (a parità di risoluzione): si
  misurano una volta e si sottraggono dal budget. Quello che resta è il **budget del modello**:
  `budget del modello = 33 ms − (pre + post + trasferimenti)`.

### Risoluzione

**Training a risoluzione fissa, deploy a risoluzione variabile.**

- **Training**: tutti i modelli si allenano a **640×640**, la scala di Svanström (nativo 640×512). Non si allena a
  nessun'altra risoluzione, e non si ingrandisce mai un'immagine (non aggiunge dettagli).
- **Deploy**: camera **16:9** a tre risoluzioni. Il modello riceve il fotogramma intero; l'ingresso va arrotondato a
  un multiplo di 32 (lo stride massimo della rete):

  | | fotogramma | ingresso del modello |
  |---|---|---|
  | 480p | 854×480 | 864×480 |
  | 720p | 1280×720 | 1280×736 |
  | 1080p | 1920×1080 | 1920×1088 |

- **Perché funziona**: un detector riconosce un oggetto in base a **quanti pixel occupa**, non a quanto è grande
  l'immagine. Un riquadro 640×640 preso da un fotogramma 1080p contiene oggetti grandi in pixel quanto quelli del
  training: il fotogramma intero è come più riquadri affiancati. Con la stessa camera, una risoluzione più alta non
  rende il modello più preciso sul singolo oggetto: gli fa vedere **droni più lontani**, che a risoluzione più bassa
  sarebbero troppo piccoli.
- **Cosa può cambiare**: \<esempio>il **recall** (quanti oggetti trova) dovrebbe restare uguale\</esempio>; la **precisione** può scendere, perché un
  fotogramma più grande ha più cielo e quindi più occasioni di falso allarme (da 640×640 a 1920×1088 il cielo è circa
  5 volte più grande). Per questo la mAP alla risoluzione di deploy **non si assume uguale: si misura**, su set di
  val costruiti apposta (fase 0, passo 10). \<esempio>Ci aspettiamo che cambi poco: lo studio già fatto sulle immagini di
  background mostra che i modelli piccoli non sbagliano sul cielo vuoto; il loro problema sono le detection multiple
  sullo stesso oggetto, che non dipendono dalla grandezza del fotogramma.\</esempio>
- **La risoluzione non è una leva da allenare, è una condizione di deploy**: cambia la latenza (in modo circa
  quadratico) e la mAP misurata sui set di deploy, mai il training.

### Training e seed

- **Una sola ricetta di training per tutti i training della tesi**, fissata in fase 0 e mai più cambiata
  (ottimizzatore, learning rate, augmentation, batch **e numero di epoche**), uguale per tutti i modelli:
  - **MuSGD**, `lr0` 0.01, `lrf` 0.01 (lr finale 1e-4, decadimento lineare): la combinazione migliore dei training
    da zero del progetto vecchio (ablation sull'ottimizzatore, pico su AOD4: MuSGD 0,649 di mAP50-95, SGD 0,643,
    Adam 0,637); 0.01 è anche il valore che `optimizer=auto` sceglie per MuSGD;
  - momentum, weight decay e warmup di default (0.937, 5e-4, 3 epoche);
  - **batch** scelto in base alla GPU, lo stesso per tutti i modelli: Ultralytics accumula i gradienti fino a un
    batch nominale di 64 (`nbs`), quindi con batch ≤ 64 l'aggiornamento dei pesi resta equivalente;
  - **augmentation**: default di Ultralytics con mosaic acceso (il mosaic non cambia la grandezza degli oggetti);
    **`scale` 0.25** invece di 0.5 (zoom casuale fra 0,75 e 1,25): con 0.5 un oggetto di 16 px scenderebbe a 8 px,
    fuori dal campo della tesi e sotto il passo 8 della testa; con 0.25 al massimo a 12 px, e resta un po' di
    ingrandimento per le celle con pochi oggetti grandi;
  - seed fisso e `nms=False` (testa). Niente ricette ridotte per
  esplorare: a poche epoche i modelli piccoli sono più indietro dei grandi e i trend risulterebbero falsati. Così le
  differenze fra modelli dipendono solo dal modello, e ogni training della mappatura vale anche come seed 0 dei
  confronti finali.
- **Lunghezza: 1000 epoche, con early stopping a patience 200** come rete di sicurezza. Ferma solo i modelli
  chiaramente in overfitting (su un dataset piccolo come Svanström il massimo arriva verso l'epoca 15–30), senza
  tagliare i modelli piccoli che migliorano lentamente (il pico su AOD4 guadagnava ancora 0,02 fra l'epoca 300 e la
  900). Il mosaic si spegne nelle ultime 10 epoche (`close_mosaic` di Ultralytics): i run fermati prima dall'early
  stopping non fanno questa parte, ed è un limite da dichiarare. `best.pt` si sceglie comunque sul val. Per ogni run si registra a che
  epoca si è fermata: un modello della griglia che si ferma presto è il segnale che il dataset lo limita, e va
  riportato.
- I training esplorativi usano **un solo seed**. I modelli che entrano nei confronti finali si allenano con
  **3 seed** e si riportano come media ± deviazione standard: con un seed solo, una differenza di 0,5–1 punto di
  mAP può essere solo fortuna.

### Dispositivi

- **Una sola configurazione per dispositivo, quella di deploy**: tutti i core, frequenza massima, profilo di potenza
  massimo sulla Jetson. Core e frequenze non si variano: la tesi studia il modello, non il dispositivo.
- Alla fine (fase 5) un solo controllo di robustezza: i modelli scelti rimisurati a bassa potenza.

### Misure

- Latenza = mediana di 3 passate, alternando i modelli fra una passata e l'altra (così un rallentamento temporaneo
  del dispositivo non colpisce sempre lo stesso modello).
- A ogni misura si registrano le condizioni: frequenze, temperatura, governor, clock della GPU.

### Precisioni

- In fase 2a si misura la latenza in **tutte le precisioni supportate** da ogni dispositivo (PI e AGX: fp32, fp16,
  int8; DX: solo int8). Da lì in poi il **deploy considera solo fp16 e int8**: \<esempio>l'fp32 non conviene mai rispetto
  all'fp16 (stessa precisione del modello, latenza uguale o peggiore)\</esempio>, e la fase 2a lo documenta una volta per tutte.
- **fp16 contro int8 è un risultato, non una premessa.** Su ogni dispositivo si misurano latenza (2a) e mAP (fase 3)
  in entrambe le precisioni, e la scelta fra le due esce dai dati. \<esempio>Sul PI ci aspettiamo che l'int8 venga
  scartato: latenza simile all'fp16 e precisione più bassa.\</esempio> Lo suggerisce la prova preliminare di X2
  (sezione 8, modelli del progetto vecchio): sul pico l'fp16 di OpenVINO impiega 28,5 ms con mAP 0,657, la migliore
  variante int8 29,4 ms con mAP 0,630. Il motivo atteso è hardware e software: il Cortex-A76 calcola in fp16
  nativamente ma non ha le istruzioni int8 veloci (i8mm); OpenVINO su ARM non esegue le conv in int8, oppure lo fa ma
  rallenta la SiLU; in onnxruntime la SiLU, circa un terzo del tempo, costa uguale in int8 e in fp16.

### Testa

- **Testa one-to-one, senza NMS, ovunque** (decisa il 13/10/2026): validazione durante il training (`nms=False`,
  quindi `best.pt` scelto sulla one-to-one), valutazione, export, misura di latenza e quantizzazione. Il training è lo
  stesso con le due teste (si allenano insieme); la scelta decide cosa si valuta, si esporta e si misura. Perché:
  - il post-processing ha un costo fisso (un numero fisso di box, `export_max_det` = 64), mentre l'NMS costa in base ai box che passano la
    soglia, cioè dipende dai pesi e dall'immagine: con la one-to-one la latenza misurata con pesi casuali (fasi 1 e 2a)
    è quella vera;
  - nel budget end-to-end l'NMS andrebbe implementata e misurata su ogni dispositivo (sulla DX girerebbe sulla CPU
    del Pi);
  - si valuta lo stesso modello che si esporta;
  - in precisione le due teste sono pari (progetto vecchio, mAP50-95 one-to-many / one-to-one: pico 0,492 / 0,491,
    nano 0,528 / 0,526).
- Il costo sono le detection doppie, che restano senza NMS: si contano come metrica. Come riferimento, ogni E2 valuta
  anche la one-to-many con NMS dagli stessi pesi.

### Metriche di precisione

- **Campo della tesi: oggetti di almeno 16 px.** La grandezza di un oggetto è la radice dell'area del suo box in
  pixel, √(w·h), come in COCO. Gli oggetti sotto i 16 px si escludono **costruendo il dataset**: le immagini che
  contengono un oggetto sotto i 16 px **non entrano in val e test** (fase 0, passo 7), così le metriche standard di
  Ultralytics valgono già solo per oggetti ≥ 16 px, senza valutazioni personalizzate. Nel train restano, con la loro
  label (togliere solo la label insegnerebbe al modello che lì non c'è niente). Limite da dichiarare: val e test non
  contengono droni più lontani di così, che in deploy invece ci saranno.
  **Perché 16 px:**
  - **limite dell'architettura**: la testa più fine di YOLO26 lavora a passo 8, cioè una cella ogni 8×8 pixel
    d'ingresso, e ogni cella può proporre un oggetto. Sotto i 16 px un oggetto copre al massimo una o due celle per
    lato: la rete ha pochissima informazione per riconoscerlo e per distinguerlo dalle altre classi. Ultralytics
    allarga a 8 px in training i box più piccoli per dargli almeno una cella (`ultralytics/utils/tal.py`), ma non
    può aggiungere dettaglio;
  - **letteratura**: sui benchmark di oggetti minuscoli (AI-TOD, Wang et al. 2021) i detector come YOLO trovano
    quasi niente sotto gli 8 px e poco, con precisione bassa, fra 8 e 16 px; diventano usabili sopra i 16 e normali
    sopra i 32;
  - **affidabilità delle label**: un box di pochi pixel è difficile da disegnare in modo preciso e coerente, e
    l'errore di qualche pixel pesa molto sull'IoU; le metriche sotto i 16 px misurerebbero soprattutto il rumore delle
    annotazioni;
  - **leggibilità dei confronti**: con pochi oggetti minuscoli, molto difficili, la mAP complessiva oscillerebbe per
    effetti di quella fascia invece che per differenze reali fra i modelli.
- mAP50 e mAP50-95, come al solito, sugli oggetti ≥ 16 px, sul val a 640 e sui tre set di val alle risoluzioni di
  deploy. È la **metrica principale** di tutte le scelte;
- **recall sugli oggetti piccoli** (16–32 px), sul test in fase 5: sono i droni lontani, il caso più difficile;
- **falsi positivi per fotogramma** sulle immagini di solo cielo e sui set di deploy: un detector che vede droni
  nelle nuvole è inutile, e più grande è il fotogramma più occasioni ha di sbagliare;
- **detection multiple sullo stesso oggetto** (box in più sopra un oggetto già trovato): con la testa one-to-one non
  c'è NMS a toglierle, e sono il problema noto dei modelli piccoli;
- **Soglia di confidenza.** La mAP si calcola con `conf=0.001` (default del `val` di Ultralytics) e non dipende
  dalla soglia: tutte le scelte delle fasi 2–4 la usano. La **soglia di deploy** è quella che **massimizza l'F1** sul
  val (Ultralytics calcola già la curva F1): **una sola soglia per tutte le classi**, una per modello e risoluzione di
  deploy (su un fotogramma più grande l'ottimo può salire). Il confronto con le soglie ottime per classe si fa **una
  volta sola, alla fine**: sui modelli scelti, sul val, subito prima di aprire il test (fase 5). Se si discostano poco
  dalla generale (\<esempio>entro ±0,05\</esempio>) si tiene la generale e lo si riporta, altrimenti per quei modelli si usa
  una soglia per classe. Con la testa one-to-one la soglia è l'unico post-processing, e massimizzare l'F1 riduce anche le detection
  doppie (contano come falsi positivi). La soglia serve alle metriche operative (precision, recall, falsi positivi
  per fotogramma); si sceglie **solo sul val** e in fase 5 si applica al test senza ricalcolarla.
- **mAP per classe e per fascia di grandezza del box**, con fasce **16–32, 32–64, 64–96, sopra 96 px**: 16–32 e
  32–64 come "small" e "medium" di AI-TOD, 64–96 e sopra 96 come "medium" e "large" di COCO. Dice fino a che grandezza
  il modello riconosce ancora un oggetto. Si riporta sul test in fase 5.

---

## 3. Dispositivi

| sigla | dispositivo | dove calcola | software di inferenza | precisioni | configurazione |
|---|---|---|---|---|---|
| **PI** | Raspberry Pi 5 | CPU Cortex-A76, 4 core | OpenVINO (fp16), onnxruntime (int8, X2) | fp16, int8 | 4 core, 2.4 GHz |
| **AGX** | Jetson AGX Orin | GPU Ampere | TensorRT | fp16, int8 | profilo MAXN, clock fissati |
| **DX** | Raspberry Pi 5 + DX-M1 | NPU, collegato via PCIe | DX-COM (compilatore) e DX-RT (runtime) | solo int8 | host come PI |

Cosa ci aspettiamo da ciascuno, in base alle prove già fatte:

- **PI**: la latenza cresce quasi in proporzione ai calcoli, più un costo fisso interno al modello. Dalle misure
  attuali (OpenVINO fp16, 4 core): circa 16,6 ms + 15,2 ms per ogni GMAC. Il pico sta a ~28 ms, il nano a ~57 ms.
  L'allineamento dei canali non conta.
- **AGX**: \<esempio>per modelli piccoli la GPU è quasi vuota e la latenza dipende soprattutto dal **numero di
  strati** (ogni strato ha un costo minimo di qualche µs), e i canali "tondi" (multipli di 8 o 16) contano\</esempio>.
  È quello che succede sulla RTX 6000 della workstation, misurato: il probe dà N=16 e un gradino netto a 64 canali, e
  il pico potato canale per canale è più lento del pico intero. Sulla AGX va verificato.
- **DX**: un NPU elabora i canali a blocchi fissi; \<esempio>ci aspettiamo una latenza a gradini, con la dimensione dei blocchi
  da scoprire. Il trasferimento dell'immagine sul PCIe potrebbe pesare quanto l'inferenza per i modelli piccoli.\</esempio>

---

## 4. Modelli

Si usa la famiglia **YOLO26** con la testa **one-to-one** (senza NMS). Nel training si variano due cose:

- la **larghezza** (quanti canali ha ogni strato);
- la **profondità**, cioè il **numero di strati convoluzionali** della rete (definizione sotto).

La **risoluzione** non si varia nel training (sempre 640×640): è la condizione di deploy (480p, 720p, 1080p, regole
fisse), su cui si misurano latenza e mAP.

| sigla | nome | larghezza | profondità | ruolo |
|---|---|---|---|---|
| F | femto | 0.0625 | D126 | il più piccolo, sotto il pico |
| P | pico | 0.125 | D126 | candidato per PI |
| N- | nano corto | 0.25 | D101 | prova della profondità |
| **N** | **nano** | **0.25** | **D126** | **centro della griglia**; \<esempio>previsto come modello da potare su PI (lo decide la 2d)\</esempio> |
| N+ | nano profondo | 0.25 | D170 | prova della profondità |
| S- | small corto | 0.5 | D101 | prova della profondità |
| S | small | 0.5 | D126 | candidato per AGX e DX; secondo centro |
| M | medium | 1.0 | D126 | il più grande |

### Profondità: definizione operativa

**La profondità è il numero di strati convoluzionali (conv) della rete.** I livelli si chiamano con quel numero:
D101, D126, D170. Il numero di conv **non dipende dalla larghezza** (verificato per F, N e S: stessi conteggi), quindi
ogni livello è la stessa topologia a ogni larghezza.

| livello | conv | bottleneck | come si ottiene in Ultralytics | parametri del nano |
|---|---|---|---|---|
| **D101** (N-, S-) | 101 | 8 | moltiplicatore di profondità 0.5, **tutti** i blocchi C3k2 con `c3k=False` | 2,44 M (−3%) |
| **D126** (standard) | 126 | 13 | moltiplicatore 0.5, YAML di YOLO26: `c3k=False` nei primi due C3k2, `True` negli altri | 2,51 M |
| **D170** (N+) | 170 | 25 | moltiplicatore 1.0 | 3,03 M (+21%) |

Perché servono tre livelli costruiti così: **la profondità non si riduce con il moltiplicatore.** In Ultralytics le
ripetizioni di ogni blocco valgono `max(round(n × profondità), 1)` e nella YOLO26 i blocchi hanno `n = 2`: a 0.5 si
ottiene già una ripetizione, il minimo. Sotto 0.5 il modello resta identico (verificato: nano a 0.5, 0.33, 0.25 e
0.1 ha sempre 2,505 M parametri e 13 bottleneck).

D101 è anche un esperimento pulito sul **numero di strati**: toglie un quinto degli strati quasi senza cambiare
parametri e calcoli, quindi dice direttamente quanto pesa il costo per strato su ogni dispositivo.

**M si costruisce con la topologia D126, non con la scala `m` ufficiale.** Ultralytics, per le scale m, l e x, forza
`c3k=True` in tutti i blocchi C3k2 (`ultralytics/nn/tasks.py`, `parse_model`): la m ufficiale ha 136 conv e 15
bottleneck. Con quella, il passo da S a M cambierebbe larghezza **e** profondità insieme. Il nostro M ha larghezza
1.0, tetto di 512 canali, 126 conv e 21,7 M parametri (la m ufficiale: 21,8 M): la curva della larghezza resta una
variazione della sola larghezza.

Perché il **nano** è il centro: le larghezze 0.0625 / 0.125 / 0.25 / 0.5 / 1.0 raddoppiano ogni volta, e il nano è
proprio quella in mezzo. Le prove di profondità si ripetono anche su **S**, perché S è il modello di
riferimento per AGX e DX, mentre il nano lo è per PI.

Nota sulle larghezze piccole: Ultralytics arrotonda i canali in uscita di ogni modulo a multipli di 8 con un minimo
di 8, mentre i canali interni dei blocchi possono scendere a 1–4. Sotto il pico la riduzione quindi non è uniforme
(il femto ha il 37% dei calcoli del pico, non il 25%), e \<esempio>i canali interni minuscoli sono pessimi per GPU e NPU\</esempio>. È un
risultato da riportare, non un problema da aggirare.

### Valori ammessi: scala discreta di larghezze, risoluzioni di deploy

La larghezza **non è continua**. Costruendo il modello per 121 larghezze fra 0.0625 e 1 (passo 1/128):

- **44 non sono valide**: il forward fallisce perché il blocco di attenzione (C2PSA) riceve un numero di canali
  incompatibile con le sue teste (per esempio tutta la fascia 0.38–0.48);
- le 77 valide danno solo **57 architetture distinte**: larghezze vicine finiscono sugli stessi canali, per
  l'arrotondamento a multipli di 8.

Per questo si usa una **scala discreta di 9 larghezze**, a passi di circa 1,4 volte, tutte valide. Sopra 0.5 si usa
lo stesso tetto di canali della m ufficiale (512 invece di 1024), altrimenti 0.75 avrebbe più parametri della M.

| larghezza | 0.0625 (F) | 0.09375 | 0.125 (P) | 0.1875 | 0.25 (N) | 0.375 | 0.5 (S) | 0.75 | 1.0 (M) |
|---|---|---|---|---|---|---|---|---|---|
| tetto canali | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 512 | 512 |
| parametri | 0,20 M | 0,40 M | 0,68 M | 1,45 M | 2,5 M | 5,6 M | 10 M | 12,3 M | 21,7 M |
| conv a multipli di 16 | 59% | 40% | 87% | 60% | 99% | 87% | 100% | ~100% | 100% |

Le 5 larghezze con nome (F, P, N, S, M) sono quelle della griglia di training; le 4 intermedie servono alla ricerca
dell'ottimo (se ne misura la latenza, la loro mAP si predice, l'ottimo si allena). **Le larghezze intermedie danno
canali multipli di 3** (3, 6, 12, 24, 48, 96): lo 0.09375 ha solo il 62% di conv a multipli di 8 e il 40% a multipli
di 16, e femto e 0.09375 hanno 9 conv sotto gli 8 canali. \<esempio>Su GPU e NPU, che lavorano a blocchi di canali, queste
larghezze rischiano di costare più delle potenze di 2 a parità di calcoli\</esempio>: la griglia di latenza della 2a lo misura,
ed è un risultato da riportare.

Le risoluzioni sono le tre di deploy: **480p, 720p, 1080p** (ingressi 864×480, 1280×736, 1920×1088; regole fisse).

Ogni configurazione valutata (larghezza, profondità, risoluzione di deploy, precisione) è quindi **un modello che
esiste**, e l'ottimo della fase 2c si sceglie solo fra queste.

### Notazione dei run

Ogni run ha un codice `<tipo><fase>-<numero>`, per esempio `T2-03` = training 3 della fase 2.

| lettera | tipo di run |
|---|---|
| T | training |
| E | valutazione della precisione sul val (o sul test in fase 5) |
| B | misura di latenza |
| P | probe dei canali (reti finte) |
| L | stima e verifica dello stimatore di latenza |
| Q | quantizzazione int8 |
| S | search di potatura |
| FT | finetune dopo la potatura |
| R | riaddestramento da zero di un'architettura potata |
| X | prova preliminare sulle incognite |

---

## 5. Fase 0 — Preparare il terreno (circa 2,5 settimane)

> **Cosa testiamo**: se il dataset copre ogni classe a ogni fascia di grandezza, con box corretti, ed è diviso senza
> leakage.
>
> **Come**: regole fissate prima di toccare i dati; correzione dei box di Svanström; conteggio sulla matrice classe ×
> fascia; aggiunte per riempire le celle scoperte, annotate con la stessa regola; split per video e bilanciamento;
> set di val alle risoluzioni di deploy.
>
> **Conoscenza che produciamo**: la matrice del dataset (istanze e video per classe e fascia); quanto erano larghi i
> box originali di Svanström; i set di val di deploy.
>
> **Come la usiamo**: il dataset e la ricetta valgono per tutta la tesi; le celle sotto il minimo si dichiarano come
> limite; i set di val di deploy danno la mAP con cui si sceglie l'ottimo (2c).

**Domanda**: i dati sono adatti, e il protocollo è definito?

Le misure sui dispositivi della fase 1 e la griglia di latenza della 2a usano pesi casuali, quindi **non aspettano il
dataset**: girano in parallelo a questa fase.

### Costruzione del dataset: l'ordine delle operazioni

Ogni passo dipende da quello prima.

1. **Fissare le regole, prima di toccare i dati.**
   - Classi: aereo, uccello, drone, elicottero.
   - Fasce sulla √area del **box aderente**, alla scala di training (640): 16–32, 32–64, 64–96, sopra 96 px. Le
     immagini con almeno un oggetto sotto i 16 px vanno solo nel train (regole fisse, metriche).
   - **Regola di annotazione: box aderente all'oggetto**, uguale per tutte le fonti.
   - **Obiettivo di bilanciamento**: una matrice classe × fascia (4 × 4 = 16 celle) con un **minimo di istanze e un
     minimo di video distinti per cella** (i valori si fissano qui). Contano anche i video: mille frame dello stesso
     video valgono poco più di pochi frame. Un bilanciamento perfetto non esiste (uccelli grandi ripresi da terra
     sono rari in natura): l'obiettivo è un minimo per cella, non celle uguali.
   - Requisiti delle fonti: a colori, da terra verso il cielo, ID dei video, licenza adatta a una tesi.
2. **Correggere i box di Svanström (indispensabile).** Le label originali sono larghe e incoerenti: un box di 16 px
   contiene un oggetto di 4–6 px (verificato a occhio su uccello, drone ed elicottero). Con box così le fasce di
   grandezza non dicono niente, la mAP50-95 penalizza alle IoU alte anche un box preciso, e mescolare fonti con box
   aderenti insegnerebbe al modello due modi di disegnare il box.
   - Ogni box si dà come prompt a **SAM 2**, che restituisce la sagoma dell'oggetto; dalla sagoma si ricava il box
     aderente. Per gli oggetti su cielo uniforme si confronta con una soglia di contrasto locale dentro il box.
   - Revisione a mano di un campione, e di tutti i casi critici: sfondi complessi (edifici, alberi, pali), box che
     cambiano molto, sagome vuote o spezzate.
   - Si misura e si riporta **quanto erano larghi i box originali** (rapporto fra √area originale e aderente, per
     classe e fascia): è un risultato sul dataset.
   - Nello stesso passaggio si cercano gli **oggetti visibili non etichettati** (per esempio gli stormi) e si
     etichettano. Lo stesso vale per le aggiunte (passo 5). Si riporta quanti oggetti mancavano per fonte.
3. **Contare.** Si riempie la matrice con Svanström corretto: istanze e video per cella. Va fatto **dopo** il passo
   2, perché le fasce dipendono dalla grandezza dei box. \<esempio>Con i box originali (fasce COCO, 640×512):
   elicotteri grandi abbondanti, uccelli grandi zero, droni grandi 37.\</esempio>
4. **Elenco dei buchi**: le celle sotto il minimo diventano la lista di cosa cercare.
5. **Riempire i buchi**, con le regole del passo 1.
   - Fonti scelte quando si crea il dataset (candidate in `docs/ricerca_dataset.md`), a colori, da terra verso il
     cielo; le immagini entrano a **640×512**, come Svanström. Box proposti da un detector a vocabolario aperto,
     resi aderenti con SAM 2 e rivisti a mano.
6. **Selezione dei frame per video**: via i quasi-duplicati con l'hash semantico (come in `tools/svanstrom.py`, ora nella repo `~/Desktop/torretta-dataset`).
7. **Split per video** (70/20/10), su tutte le fonti insieme, bilanciato sulla matrice classe × fascia (lo schema dello stesso
   script). Va fatto **prima** del bilanciamento finale, così nessun video finisce in due split. Dai
   video di val e test si tolgono le immagini con oggetti sotto i 16 px (restano solo nei video di train). Poi un
   controllo fra split con lo stesso hash: i quasi-duplicati di immagini del train (video diversi della stessa
   scena) si tolgono da val e test.
8. **Bilanciamento dentro ogni split**: tetti per cella e per video (per esempio i droni, oggi molto più numerosi delle
   altre classi). Anche val e test si bilanciano con un minimo per cella: la mAP di una cella con 5 oggetti non vuol
   dire niente.
9. **Background**: circa il 10% di frame senza oggetti, dagli stessi video.
10. **Set di deploy, statistiche, congelamento.**
    - Set di val alle risoluzioni di deploy (`val_480p`, `val_720p`, `val_1080p`), costruiti una volta su disco e
      uguali per tutti i modelli. Ogni fotogramma di deploy è un **mosaico**: un frame di val con il suo oggetto, in
      posizione casuale a seed fisso, e intorno frame di **solo sfondo dello stesso video** (oggetto cancellato con
      l'inpainting che lo script del dataset usa già per il background), così il cielo è lo stesso e non ci sono
      giunture fra cieli diversi. Per il 1080p: 3×2 frame da 640×512 (1920×1024) più una striscia di sfondo fino a
      1088; per 720p e 480p si ritaglia di conseguenza (a 480p il frame con l'oggetto, alto 512, si taglia in modo che
      l'oggetto resti intero). Nessun ridimensionamento: gli oggetti hanno la stessa grandezza in pixel a ogni
      risoluzione. Il frame con l'oggetto è **uno solo**: come un fotogramma vero
      della torretta (un drone, o nessuno, e tanto cielo vuoto); con più frame pieni i falsi allarmi peserebbero
      meno del vero e la mAP sarebbe ottimista. Si valutano con il normale `val` a `imgsz` di deploy, sui modelli
      PyTorch e sugli export; gli export si valutano con il runtime vero sul dispositivo. Lo stesso si prepara per il test (`test_480p`, …), chiuso fino alla fase 5. La
      validazione di Ultralytics non fa mosaic (è solo augmentation di training, casuale e quadrata), per questo i set
      si costruiscono a parte.
    - Statistiche finali: la matrice classe × fascia per split, quota di ogni fonte per classe, immagini senza
      oggetti; quanti oggetti sotto i 16 px (solo per sapere quanti sono).
    - **Manifest** (fonti, parametri, frame scelti, hash) e **congelamento**: da qui il dataset non cambia più, e il
      test si chiude.

### Il resto del protocollo

- **Valutazione**: il `val` standard di Ultralytics (mAP e AP per classe, quindi anche l'AP dei droni). La mAP per
  fascia si ottiene con i sottoinsiemi del dataset per fascia, come i sottoinsiemi `small`/`medium`/`large` che
  lo script del dataset costruisce già. Falsi positivi per fotogramma e detection multiple si contano dalle
  predizioni salvate del `val`, con un'analisi breve.
- **YAML dei modelli**: F, P, N, S, M e le larghezze intermedie con un nome di scala proprio (mai `m`, `l`, `x`,
  che forzano `c3k=True`, sezione 4); D101 (`c3k=False` in tutti i C3k2) e D170 (moltiplicatore 1.0). Controllo
  automatico del numero di conv (101, 126, 170) a ogni costruzione.
- **Correzioni del codice di misura (fatte il 13/10/2026, da verificare sui dispositivi)**: TensorRT ora misura i
  trasferimenti (tolto `--noDataTransfers=false`, che trtexec leggeva come `--noDataTransfers`; il parser segnala un
  errore se H2D vale 0); OpenVINO int8 usa `infer_precision` f16 su aarch64 (con f32 sul PI cadeva nei kernel di
  riferimento, 2 s per inferenza) e f32 su x86. Prima di B1-2: una run su AGX per controllare che H2D e D2H non
  siano 0 (con la memoria unificata potrebbero esserlo davvero, e allora il controllo del parser va rivisto).
- **Ricetta di training** fissata una volta per tutte (regole fisse): MuSGD, `lr0` 0.01, `lrf` 0.01, batch secondo la
  GPU, augmentation di default con mosaic e `scale` 0.25, 1000 epoche con patience 200. Il controllo sulla lunghezza è gratis: è la curva del nano
  della mappatura (T2-03).
- **Misura end-to-end** sui tre dispositivi, secondo la definizione delle regole fisse, con il controllo automatico
  delle condizioni (frequenze, temperatura, clock).

### Run

Nessuna misura: la fase 0 è lavoro sui dati e sul protocollo. Le misure di ripetibilità della latenza sono in fase 1
(B1-0).

### Cosa ne esce

Il dataset congelato con box aderenti e la sua matrice classe × fascia, il rapporto sulla larghezza dei box originali,
i set di val (e di test) alle risoluzioni di deploy, la ricetta di training.

---

## 6. Incognita X1 — La DX-M1 riesce a eseguire i nostri modelli? (prima della fase 1)

> **Cosa testiamo**: se le nostre varianti di YOLO26 girano interamente sulla DX-M1.
>
> **Come**: esportando e compilando le varianti con DX-COM (pesi casuali) ed eseguendole con DX-RT.
>
> **Conoscenza che produciamo**: se le varianti girano interamente sull'NPU; quanto dura una compilazione.
>
> **Come la usiamo**: se la DX-M1 resta nel piano o diventa descrittiva; il tempo di compilazione decide quanto
> costano le misure sulla DX (probe, griglia di latenza della 2a, stimatore del blocco 2).

**Perché qui**: dalla fase 1 in poi si misura anche sulla DX. Se le nostre varianti non girano sull'NPU, quelle misure
non si possono fare.

La DX-M1 ha un compilatore proprio (DX-COM) che accetta solo certi operatori. YOLO26 contiene blocchi (per esempio
l'attenzione del C2PSA e la testa) che potrebbero non essere supportati: in quel caso finiscono sulla CPU del Pi, e la
latenza crolla.

- **Già noto**: YOLO26 è nel model zoo di DEEPX, quindi i modelli standard si compilano. Resta da verificare (mezza
  giornata) che compilino anche **le nostre varianti**: larghezze sotto il pico (canali interni da 1–4), 4 classi,
  testa one-to-one, profondità D101 (`c3k=False`).
- **Cosa si fa**: si esportano F, N- ed S (pesi casuali: per questa prova i pesi non contano), si compilano con
  DX-COM e si eseguono con DX-RT.
- **Cosa si ottiene**: conferma che le varianti girano interamente sull'NPU; tempo di compilazione per rete.
- **Se fallisce**: si sostituiscono gli operatori problematici (attivazioni, blocchi di attenzione) e si ricompila;
  se non è possibile, la DX-M1 diventa un risultato descrittivo ("questi modelli non girano interamente sull'NPU").

---

## 7. Fase 1 — Conoscere i dispositivi, versione di base (circa 3 giorni, in parallelo alla fase 0)

> **Cosa testiamo**: quanto sono ripetibili le misure di latenza; quanto dei 33 ms resta al modello su ogni
> dispositivo; il comportamento di base di ogni
> dispositivo: se la latenza fa gradini al variare dei canali, quanto costa uno strato anche quando calcola poco,
> quali sono le prestazioni massime reali.
>
> **Come**: misurando lo stesso modello 5 volte su ogni dispositivo; misurando a parte pre-processing,
> post-processing e trasferimenti alle tre risoluzioni di deploy; con un
> probe ridotto (un solo tipo di strato, due dimensioni di mappa); con catene di strati minuscoli di lunghezza
> crescente; con una misura di roofline. Tutto con pesi casuali, nessun training.
>
> **Conoscenza che produciamo**: la **soglia di rumore** della latenza per dispositivo; il **budget del modello** per
> dispositivo e risoluzione di deploy; i **tratti di base** di
> ogni dispositivo: granularità dei canali (N), costo per strato, picco di calcolo e banda.
>
> **Come la usiamo**: la soglia dice in tutte le fasi dopo se una differenza di latenza è vera o rumore; il budget è
> il vincolo della ricerca dell'ottimo (2c) e della decisione (2d); i tratti di base
> servono a **spiegare** i trend della fase 2 (per esempio perché sull'AGX la larghezza costa poco); se si fa il
> blocco 2, sono il punto di partenza dello stimatore di latenza (4a).

**Domanda**: quanto sono affidabili le misure, quanto tempo resta al modello su ogni dispositivo, e come si comporta
ogni dispositivo?

Questa è la versione minima: serve al blocco 1. La caratterizzazione completa (tutti i tipi di strato, ingressi e
uscite diversi, reti potate a caso) serve solo alla potatura, e si fa nel blocco 2 (fase 4a).

### Run

| run | cosa misura | dove | cosa si varia |
|---|---|---|---|
| B1-0 | ripetibilità: lo stesso modello (N, pesi casuali) misurato 5 volte | PI fp16, AGX fp16, DX int8 | niente: è sempre la stessa misura |
| B1-2 | pre-processing, post-processing e trasferimenti | PI, AGX, DX | risoluzione di deploy: 480p, 720p, 1080p |
| B1-1 | costo minimo di uno strato: catene di strati identici e piccoli (conv 3×3, 16 canali, 54×30) | PI, AGX, DX | numero di strati: 1, 2, 4, 8, 16, 32 |
| P1-1 | probe ridotto: 8 conv 3×3 uguali, canali da 1 a 64 a passo 1 e da 72 a 128 a passo 8, mappe del deploy a passo 8 e 16 del 480p e del 1080p (108×60, 54×30, 240×136, 120×68), 3 passate | PI fp16, AGX fp16, DX int8 | numero di canali |
| B1-3 | prestazioni massime reali: picco di calcolo e banda di memoria (roofline) | PI, AGX, DX | precisioni del dispositivo |

**A cosa serve B1-0.** Misura lo stesso modello 5 volte su ogni dispositivo. La variazione fra le passate è la
**soglia di rumore** della latenza: sotto quella soglia una differenza fra due modelli non si distingue dal caso. Per
esempio, se le passate dello stesso modello variano dell'1%, una differenza del 3% fra due modelli è vera, una dello
0,5% no. La soglia si usa in tutte le fasi dopo: nei confronti di latenza della fase 2, nella scelta dell'ottimo (due
candidate entro la soglia sono equivalenti) e nei confronti della potatura. Si fa per prima, perché dice anche se
il dispositivo è pronto per misurare (frequenze, temperatura, processi in background).

Il probe ridotto sono 288 reti per dispositivo (72 canali × 4 mappe): \<esempio>circa 1 ora sul Pi, 4–6 ore di build
sull'AGX\</esempio>, da verificare
sulla DX (dipende dal compilatore, si scopre con X1).

### Cosa ci aspettiamo

\<esempio>

- Variazione fra le passate dello stesso modello sotto il 2% su ogni dispositivo.
- **PI**: nessun gradino; latenza quasi proporzionale ai calcoli.
- **AGX**: gradini (attorno ai 64 canali), costo per strato di qualche µs che pesa molto sui modelli piccoli; N = 8–16
  (sulla RTX 6000 viene 16).
- **DX**: gradini alla granularità dell'NPU; trasferimento PCIe importante per i modelli piccoli.

\</esempio>

### Cosa ne esce

- La **soglia di rumore** della latenza per dispositivo.
- Il **budget del modello** per dispositivo e risoluzione di deploy: `33 ms − (pre + post + trasferimenti)`.
- Una scheda per dispositivo: granularità N, costo per strato, picco di calcolo e banda.

---

## 8. Incognita X2 — L'int8 funziona sulla testa one-to-one, e conviene sul Pi? (risolta il 13/10/2026)

> **Cosa testiamo**: se l'int8 dà una mAP sensata con la testa one-to-one, con quale ricetta, e se sul PI è più
> veloce dell'fp16.
>
> **Come**: quantizzando pico e nano già allenati (progetto vecchio, `.old/`, testa one-to-one) con varie ricette;
> mAP sul val e latenza sul PI (4 core, 2,4 GHz, governor performance), con profilo per strato.
>
> **Conoscenza che produciamo**: la causa della mAP 0; una ricetta int8 che funziona; se l'int8 conviene sul PI.
>
> **Come la usiamo**: sul PI la ricetta decide quali nodi sono quantizzati nella latenza int8 della 2a e nella fase 3
> (AGX e DX usano la quantizzazione del loro strumento), e dice quale software usare per l'int8 (onnxruntime) e cosa aspettarsi. La conclusione
> sull'int8 del PI si trae dai modelli nuovi, in 2a e fase 3.

**Esito.**

**1. Causa della mAP 0.** Alla fine della testa, coordinate dei box (0–716 px) e punteggi (0–1) stanno nello stesso
tensore (`/model.23/Concat_3`) e vengono quantizzati con una sola scala (5,64): tutti i punteggi si arrotondano a 0.
Lasciando in float la **coda della testa** (regex `HEAD_TAIL`, già in `int8_full`) il punteggio torna corretto (0,959
contro 0,964 dell'fp32). Escludere solo dal Concat in poi non basta: la decodifica dei box in int8 dà mAP 0,353.

**2. Prova preliminare sul PI: l'int8 non è più veloce dell'fp16.** Latenza sul PI e mAP50-95 sul val (AOD4, modelli
del progetto vecchio):

| modello | variante | latenza PI | mAP50-95 |
|---|---|---|---|
| P | **OpenVINO fp16 (riferimento)** | **28,5 ms** | **0,657** |
| P | onnxruntime int8, testa, ramo di classe e attention in float | 29,4 ms | 0,630 |
| P | onnxruntime int8, tutto quantizzato tranne la coda della testa | 28,4 ms | 0,605 |
| P | OpenVINO int8 (NNCF) | 41,1 ms | 0,626 |
| N | **OpenVINO fp16 (riferimento)** | **56,5 ms** | **0,668** |
| N | onnxruntime int8, tutto quantizzato tranne la coda della testa | 52,6 ms (−7%) | 0,607 |
| N | onnxruntime int8, testa, ramo di classe e attention in float | 56,6 ms | 0,656 |

Perché, dal profilo per strato:

- **OpenVINO su ARM non esegue davvero l'int8**: con NNCF le conv restano in fp16 con i pesi dequantizzati e i nodi
  di quantizzazione costano in più; con attivazioni simmetriche le conv vanno in int8, ma la SiLU con
  dequantizzazione e riquantizzazione diventa lentissima.
- **onnxruntime int8 costa quanto l'fp16**: il Cortex-A76 ha l'fp16 nativo ma non le istruzioni int8 veloci (i8mm), e
  la SiLU, circa un terzo del tempo, costa uguale in int8 e in fp16. Ogni nodo lasciato in float per la precisione
  aggiunge latenza.

**Cosa ne segue**:

- **ricetta int8 del PI**: onnxruntime (con OpenVINO l'int8 su ARM è più lento: 41,1 ms contro 29,4 sul pico), con
  testa, ramo di classe e attention in float (il compromesso migliore fra latenza e mAP);
- **AGX e DX** usano la quantizzazione del loro strumento: la ricetta standard di Ultralytics per TensorRT, la
  conversione di DX-COM. Il controllo è la mAP int8 misurata sul dispositivo in fase 3 (una mAP vicina a 0 vorrebbe
  dire che anche lì la coda della testa è quantizzata);
- **attesa da verificare**: sul PI l'int8 ha latenza simile all'fp16 e mAP più bassa, quindi verrà scartato. Non si
  decide qui: la griglia di latenza (2a) e la fase 3 lo misurano sui modelli nuovi, e la tesi lo riporta come
  risultato.

**Strade non provate** (solo se un giorno servisse l'int8 sul PI): ReLU al posto della SiLU, che onnxruntime fonde
nella conv int8; XNNPACK come esecutore.

---

## 9. Fase 2 — Modelli normali di varie taglie (circa 2–2,5 settimane)

> **Cosa testiamo**: quanto sono precisi e quanto sono veloci, su ogni dispositivo, i modelli normali al variare di
> larghezza e profondità, alle tre risoluzioni di deploy.
>
> **Come**: misurando subito la latenza di tutti i modelli con pesi casuali, alle risoluzioni di deploy; allenando da
> zero a 640, con una ricetta fissa, modelli che differiscono dal nano per una sola cosa alla volta (e ripetendo la
> profondità su small); valutando ogni modello sul val a 640 e sui tre set di val di deploy; confermando con 3 seed i
> modelli che servono ai confronti finali.
>
> **Conoscenza che produciamo**: i **trend** (come cambiano mAP e latenza con larghezza e profondità, a ogni
> risoluzione di deploy, su ogni dispositivo); quanto costa in ms e in mAP passare da 480p a 720p a 1080p; da
> questi, in 2c, il **tasso di cambio** di ogni leva (punti di mAP per ms), un **modello di precisione** che predice
> la mAP di configurazioni mai allenate, e l'**ottimo per dispositivo e risoluzione di deploy**, verificato con un
> training.
>
> **Come la usiamo**: l'ottimo verificato è la **baseline** che la potatura deve battere in fase 4; la leva con il
> tasso di cambio più alto dice **su cosa deve lavorare la potatura** (togliere canali o togliere blocchi); il confronto fra l'ottimo nel budget e il modello migliore senza
> budget dice **se** la potatura serve su quel dispositivo (2d) e **da quale modello** partire.

**Domanda**: senza trucchi, quanto sono precisi e quanto sono veloci i modelli di varie taglie, su ogni dispositivo?
E su quali dispositivi il budget è davvero stretto?

Questa fase costruisce il **fronte**: per ogni dispositivo, la curva "precisione in funzione della latenza" dei
modelli normali. È il riferimento che la potatura dovrà battere.

### 2a. Prima la latenza, in tutte le precisioni (subito, con pesi casuali)

La latenza non dipende dai pesi, quindi tutti i modelli della griglia si misurano **subito**, mentre i training sono
ancora in corso, in **tutte le precisioni** che ogni dispositivo supporta. E siccome misurare non richiede training,
la latenza non si limita ai modelli che alleniamo: si misura la **griglia discreta** completa dei valori ammessi
(sezione 4): 9 larghezze × 3 profondità × 3 risoluzioni di deploy = **81 configurazioni** per dispositivo, in fp16
e in int8. L'fp32 si misura solo sui modelli della mappatura, quanto basta per mostrare che non conviene. La griglia
comprende tutti i modelli della mappatura e tutte le candidate all'ottimo: è una **mappa della latenza misurata**,
senza bisogno di interpolare. \<esempio>Costo indicativo: sul Pi qualche ora per precisione; sull'AGX la build di ogni engine
richiede minuti, quindi circa un giorno di macchina senza presidio\</esempio>; sulla DX dipende dal tempo di compilazione (X1). Per l'int8 i modelli si quantizzano con
**la ricetta del dispositivo**: sul PI quella di X2 (onnxruntime, testa, ramo di classe e attention in float); sull'AGX
la ricetta standard di Ultralytics per TensorRT (`format=engine`, `int8=True`); sulla DX la conversione di DX-COM.
Calibrazione su pesi casuali: per la latenza conta solo quali nodi sono quantizzati, non i valori. Già da qui si sa quali modelli stanno nel budget su ogni dispositivo, in ogni precisione e a ogni
risoluzione di deploy (figura A, sezione 13). \<esempio>Sul Pi probabilmente solo a 480p: a 720p il fotogramma ha circa 2,3
volte i pixel di un 640×640, a 1080p circa 5 volte, e la latenza sul Pi segue i calcoli.\</esempio>

| run | modelli | dove | precisioni |
|---|---|---|---|
| B2-1 | griglia discreta di latenza | PI | fp16 (OpenVINO), int8 ricetta X2 (onnxruntime) (fp32 sui modelli della mappatura) |
| B2-2 | griglia discreta di latenza | AGX | fp16, int8 Ultralytics (fp32 sui modelli della mappatura) |
| B2-3 | griglia discreta di latenza | DX | int8 DX-COM |

**Due risultati che escono da qui:**

1. **L'fp32 non conviene mai.** \<esempio>Ci aspettiamo che l'fp16 sia sempre veloce almeno quanto l'fp32\</esempio>: sul Pi oggi sono
   uguali (pico 27,9 contro 28,1 ms con OpenVINO), sulla GPU l'fp16 è nettamente più veloce (pico con TensorRT
   sulla RTX 6000: 0,55 contro 0,74 ms). Con la stessa precisione del modello, non c'è motivo di usare l'fp32. **Da
   qui in poi il deploy considera solo fp16 e int8.**
2. **fp16 contro int8 sul PI**: l'int8 si misura con onnxruntime e la ricetta di X2 (sezione 8), l'fp16 con
   OpenVINO. \<esempio>Ci aspettiamo latenze simili, come nella prova preliminare (pico: 28,5 ms fp16, 29,4 ms int8).\</esempio>
   La mAP int8 arriva dalla fase 3: se a latenza simile è più bassa, sul PI l'int8 si scarta, e la tesi lo motiva con
   queste misure e con il profilo per strato di X2.

### 2b. Training di mappatura (un seed)

Si cambia **una cosa alla volta rispetto al nano**. Così, se il modello cambia precisione, si sa a cosa è dovuto.
Le prove di profondità si ripetono su S per vedere se l'effetto cambia con la larghezza. Tutti i training sono a
**640×640**: la risoluzione non è una leva di training (regole fisse).

Si chiama "griglia non fattoriale": provare tutte le combinazioni (5 larghezze × 3 profondità) vorrebbe dire 15
training; così ne bastano 8, più 2–3 "fuori asse". Le combinazioni non allenate si **predicono** in 2c.

| run | modello | larghezza | profondità | a cosa serve |
|---|---|---|---|---|
| T2-01 | F | 0.0625 | D126 | curva della larghezza |
| T2-02 | P | 0.125 | D126 | curva della larghezza |
| **T2-03** | **N** | **0.25** | **D126** | **centro** |
| T2-04 | S | 0.5 | D126 | curva della larghezza |
| T2-05 | M | 1.0 | D126 (non la scala `m`, sezione 4) | curva della larghezza |
| T2-06 | N- | 0.25 | D101 | effetto della profondità |
| T2-07 | S- | 0.5 | D101 | profondità su S |
| T2-08 | N+ | 0.25 | D170 | profondità verso l'alto |

**Ogni training si valuta quattro volte**: sul val a 640 (quello del training) e sui tre set di val di deploy
(`val_480p`, `val_720p`, `val_1080p`, fase 0). Sono solo validazioni, pochi minuti per modello. Così ogni modello ha
una mAP **misurata** per ogni risoluzione di deploy: `E2-01` per `T2-01` e così via.

**Limite dei mosaici.** I set di deploy sono mosaici: le giunture fra un frame e l'altro sono bordi artificiali
(attenuati dallo sfondo dello stesso video). Senza fotogrammi veri ad alta risoluzione non si possono confrontare
con un fotogramma vero: si dichiara come limite.

**Training fuori asse (T2-F1…F3).** Con 8 punti il modello di precisione \<esempio>regge bene dentro la griglia\</esempio>, ma è
inaffidabile se l'ottimo cade in una combinazione lontana da tutti i punti allenati (per esempio larghezza
intermedia + profondità D101: due cambiamenti insieme, mai visti). Dopo la griglia di latenza (2a) si vede dove
passano i confini del budget di ogni dispositivo; lì si allenano **2–3 combinazioni fuori asse**, nella zona dove
probabilmente cadrà l'ottimo \<esempio>(per il Pi, per esempio, una larghezza fra pico e nano a profondità D101)\</esempio>. Costano
circa un giorno di GPU e mettono punti veri dove servono.

| run | modello | a cosa serve |
|---|---|---|
| T2-F1…F3 | combinazioni scelte dopo la 2a, vicino ai confini del budget | punti veri dove cadrà l'ottimo |

### 2c. Dai trend alla ricetta ottima per dispositivo

I trend da soli non bastano: dire "la larghezza pesa più della profondità" non sceglie un modello. Questo passo li
trasforma in una scelta, in quattro passi, **per ogni dispositivo e per ogni risoluzione di deploy**.

**Passo 1 — Il tasso di cambio di ogni leva.** Per ogni dispositivo, risoluzione di deploy e leva (larghezza,
profondità) si calcola quanta precisione si guadagna per ogni ms di latenza in più, fra punti vicini della griglia:
`Δ mAP50-95 / Δ ms`. \<esempio>Esempio con numeri inventati, su PI a 480p:
larghezza +2 punti ogni 10 ms, profondità +0,5. Vuol dire che su PI i ms conviene spenderli in larghezza. Su AGX il
conto sarà diverso, perché lì la larghezza costa poco finché la GPU resta mezza vuota.\</esempio> Il tasso si calcola attorno
a entrambi i centri (N e S), per vedere se cambia con la taglia. A parte si riporta **quanto costa salire di
risoluzione di deploy**: quanti ms in più, e quanta mAP in più o in meno, da 480p a 720p a 1080p.

**Passo 2 — Due modelli che predicono.**

- **Precisione**: per ogni risoluzione di deploy, `mAP(w, d)` stimata dai training della griglia valutati sul set di
  val di quella risoluzione, con una curva per leva (per esempio lineare nel logaritmo della larghezza) più un
  termine di interazione stimato dagli incroci su S. Dipende solo dal dataset, quindi è uguale per tutti i
  dispositivi. Si verifica togliendo un punto alla volta e predicendolo dagli altri (\<esempio>errore atteso sotto 1
  punto di mAP\</esempio>).
- **Latenza**: non serve predirla: è già **misurata** per ogni configurazione ammessa nella griglia discreta della 2a.

**Passo 3 — L'ottimo sotto il budget.** Per ogni dispositivo, precisione e risoluzione di deploy si prendono le
configurazioni della griglia discreta che stanno nel budget del modello (latenza misurata in 2a) e fra queste si
sceglie quella con la mAP predetta più alta (figura B, sezione 13). È sempre un modello che esiste. Sui dispositivi a
gradini (AGX, DX) le misure mostrano da sole quali larghezze sprecano un gradino \<esempio>(per esempio quelle
intermedie, con canali multipli di 3)\</esempio>.

**Criterio di scelta, valido in tutta la tesi** (ottimo, candidate, modello da potare): conta la **mAP50-95
complessiva** (media sulle 4 classi, oggetti ≥ 16 px, sul set di val della risoluzione di deploy). A pari merito,
cioè con una differenza **sotto 1 punto di mAP50-95** (soglia fissa, la stessa in tutta la tesi), vince il modello
migliore **sui droni** (AP della classe drone). Metriche
di una sola classe come criterio principale cambierebbero la domanda della tesi da "il detector migliore" a "il
miglior rilevatore di droni".

**Passo 4 — Verifica.** Si allena l'ottimo predetto con 3 seed. Gli ottimi sono uno per dispositivo e risoluzione di
deploy (fino a 9), ma il training non dipende dalla risoluzione: **si allenano solo le architetture `(w, d)`
distinte** \<esempio>(probabilmente 3–5)\</esempio>. Se l'ottimo coincide con un punto della griglia, il seed 0 c'è già e servono
solo i seed 1 e 2. Quali ottimi confermare con 3 seed (tutti, o uno per dispositivo) è la scelta 8 (sezione 15).

| run | cosa | seed / ripetizioni |
|---|---|---|
| T2-O1 | ottimo predetto per PI | 0, 1, 2 |
| T2-O2 | ottimo predetto per AGX | 0, 1, 2 |
| T2-O3 | ottimo predetto per DX | 0, 1, 2 |

Se più dispositivi o risoluzioni hanno lo stesso ottimo, i suoi training si fanno una volta sola.

> **Nota, da decidere (pending): quali candidate allenare oltre all'ottimo.** Proposta: si allena prima l'ottimo
> di ogni risoluzione del dispositivo (seed 0) e si misura l'errore assoluto fra mAP predetta e vera. L'intorno è il più grande
> fra questo errore e quello del test "togli un punto e predicilo" del passo 2 (un solo errore può uscire piccolo per
> caso). Poi si allenano (seed 0) le altre configurazioni nel budget con mAP predetta dentro l'intorno; la migliore
> misurata prende i seed 1 e 2. Le righe int8 non si allenano: sono la quantizzazione degli stessi modelli.
> \<esempio>Ottimo predetto 0,461, misurato 0,452: errore 0,9 punti; errore del passo 2: 0,6; intorno 0,9. Si allenano
> le candidate nel budget con mAP predetta ≥ 0,452.\</esempio>

**Cosa ci aspettiamo**: \<esempio>la mAP vera dell'ottimo entro circa 1 punto da quella predetta; ottimi diversi fra
dispositivi (su PI un modello stretto e solo a 480p, su AGX uno più largo anche a 1080p, su DX canali allineati ai
blocchi dell'NPU); l'ottimo migliore di qualunque taglia standard (F, P, N, S, M) che sta nel budget.\</esempio>

**Conoscenza che produciamo**: il tasso di cambio delle leve per dispositivo; la **ricetta** `(w, d)` ottima per
dispositivo, risoluzione di deploy e budget; quanto costa salire di risoluzione; quanto è affidabile scegliere un
modello senza allenarlo.

**Come la usiamo**:

- l'ottimo verificato (T2-O) è la **baseline** della fase 4: la potatura deve battere il miglior modello scalato
  possibile, non una taglia standard scelta a occhio;
- la leva con il tasso di cambio più alto dice **su cosa deve lavorare la potatura**: se costano i canali si tolgono
  canali, se costano i blocchi si tolgono blocchi;
- il modello di precisione dice **quanto si perde** a stare nel budget rispetto al modello migliore senza budget: è
  il dato su cui si basa la decisione 2d.

### 2d. Decisione per dispositivo

Per ogni dispositivo (figura C, sezione 13) si confronta l'ottimo nel budget (2c) con il **tetto**, cioè il modello
più preciso della griglia senza vincoli di latenza, alla stessa risoluzione di deploy. Il dato è la differenza di mAP
predetta fra i due (criterio di scelta della 2c: mAP50-95 complessiva, a pari merito i droni):

| caso | cosa vuol dire | cosa si fa |
|---|---|---|
| **stretto** | il tetto non sta nel budget, e l'ottimo nel budget perde più di 1 punto di mAP50-95 | fase 4: si pota partendo dal tetto (o dal modello più piccolo che lo eguaglia) |
| **libero** | il tetto sta nel budget, oppure la differenza è sotto 1 punto | l'ottimo è il modello finale; il dispositivo diventa un risultato descrittivo |

Si sceglie sempre il modello con la mAP migliore fra quelli che rispettano la latenza: la distanza di rilevamento non
entra nelle scelte, si considera solo alla fine. \<esempio>Previsione: **PI stretto** (a 480p il pico ci sta, il nano
no; a 720p forse nessuno), **AGX e DX** probabilmente "liberi".\</esempio>

### 2e. Conferma con 3 seed

Le baseline sono già allenate con 3 seed (T2-O, in 2c). Resta da allenare il **modello da cui parte la potatura**,
solo sui dispositivi "stretti". Se è un punto della griglia (per esempio N, cioè T2-03), il seed 0 è
il suo training di mappatura e servono solo i seed 1 e 2.

| run | modello | seed |
|---|---|---|
| T2-C1…C3 | modello da potare su PI \<esempio>(previsto: N)\</esempio> | 0, 1, 2 |
| T2-C4…C6 | modello da potare su AGX, se serve | 0, 1, 2 |
| T2-C7…C9 | modello da potare su DX, se serve | 0, 1, 2 |

Se due dispositivi condividono lo stesso modello, i suoi training si fanno una volta sola.

Questi training servono solo al blocco 2: si lanciano **quando la GPU è libera** durante il blocco 1, senza
farli diventare un collo di bottiglia. Se il blocco 2 non si fa, restano punti del fronte confermati con 3 seed.

### Cosa ci aspettiamo

\<esempio>

- Oltre una certa taglia la precisione smette di salire: con quattro classi potrebbe valere S ≈ M.
- La mAP ai set di deploy cambia poco con la risoluzione (regole fisse); la differenza fra modelli la fanno le
  detection multiple sullo stesso oggetto, più frequenti nei modelli piccoli.
- Sotto il pico la precisione cala in fretta (canali interni minuscoli).
- **PI**: nel budget stanno solo i modelli piccoli (circa fino al pico), e probabilmente solo a 480p.
- **AGX e DX**: probabilmente anche S sta nel budget, almeno a 720p.

\</esempio>

---

## 10. Fase 3 — Int8 (circa 1 settimana)

> **Cosa testiamo**: quanta precisione si perde passando da fp16 a int8, modello per modello. La velocità
> guadagnata è già misurata in fase 2a.
>
> **Come**: quantizzando dopo il training i modelli del fronte con la ricetta int8 del dispositivo (la stessa della
> latenza in 2a), e misurando la mAP sul val **con il runtime vero sul dispositivo**
> (OpenVINO/onnxruntime sul PI, TensorRT sull'AGX, DX-RT sulla DX); con il quantization-aware training solo dove la
> perdita supera 1 punto.
>
> **Conoscenza che produciamo**: per ogni modello e dispositivo, quanti punti di mAP costa l'int8 e quanti ms fa
> risparmiare; quale variante di quantizzazione funziona.
>
> **Come la usiamo**: si **ricalcola l'ottimo della fase 2c con le latenze int8** (il budget "compra" di più) e si
> ricontrolla la decisione 2d (un dispositivo "stretto" può diventare "libero"); si sceglie in quale precisione fare
> la potatura su ogni dispositivo e con quale variante quantizzare i potati.

**Domanda**: usando numeri a 8 bit invece che a 16, quanto si guadagna in velocità e quanto si perde in precisione?

Perché serve: sulla DX-M1 l'int8 è **obbligatorio**; su PI e AGX può spostare il fronte, facendo entrare nel budget
modelli che in fp16 non ci stavano.

### Run

Modelli: gli ottimi T2-O (seed 0), i modelli da potare (2e) e i punti della griglia, per ricalcolare l'ottimo in int8.

| run | variante | cosa si misura |
|---|---|---|
| Q3-1 | PI: ricetta di X2 (onnxruntime, testa, ramo di classe e attention in float), 500 immagini di calibrazione dal train | mAP sul val a 640 e sui set di deploy, sul dispositivo (la latenza viene dalla 2a) |
| Q3-2 | AGX: ricetta standard di Ultralytics per TensorRT (`format=engine`, `int8=True`), calibrazione dal train | come sopra |
| Q3-3 | DX: conversione di DX-COM, con la sua calibrazione | come sopra |
| Q3-4 | quantization-aware training (30–50 epoche partendo dal modello già allenato) | solo per i modelli che con Q3-1…Q3-3 perdono più di 1 punto di mAP50-95 |

### Cosa ci aspettiamo

\<esempio>

- 0,5–2 punti di mAP persi con la quantizzazione dopo il training, recuperati dal quantization-aware training.
- **PI**: latenza simile all'fp16 e mAP più bassa (prova preliminare di X2), quindi int8 scartato; è la conferma sui
  modelli nuovi che la tesi riporta.
- **AGX**: guadagno modesto, perché per i modelli piccoli pesa il costo minimo per strato.
- **DX**: è l'unico modo di farli girare.

\</esempio>

### Cosa ne esce

Il fronte in int8. Si **ricalcola l'ottimo 2c con le latenze int8** e si ricontrolla la decisione 2d: un modello che
in fp16 non stava nel budget potrebbe starci in int8, e un dispositivo "stretto" potrebbe diventare "libero". Se
l'ottimo int8 cambia, lo si verifica con un training come in 2c.

---

## 11. Blocco 2 — Fase 4: potatura con lo stimatore di latenza (circa 4 settimane, se c'è tempo)

Si fa **solo dopo aver chiuso il blocco 1**, e **solo sui dispositivi "stretti"** della decisione 2d.
Sugli altri non c'è niente da ottimizzare. Ha due parti: prima lo stimatore (4a), poi la potatura (4b).

### 4a. Stimatore di latenza (circa 2 settimane)

> **Cosa testiamo**: se una formula semplice predice la latenza di una rete potata su ogni dispositivo, guardando solo
> com'è fatta (quanti strati, di che tipo, con quanti canali in ingresso e in uscita, a che dimensione di mappa).
>
> **Come**: misurando reti finte (probe completi, ingressi e uscite diversi, catene di strati) per stimare i
> parametri, e circa 100 reti potate a caso per verificare la formula.
>
> **Conoscenza che produciamo**: lo **stimatore di latenza** per dispositivo e precisione, con il suo errore; i suoi
> parametri, che descrivono il dispositivo in pochi numeri (quanto costa uno strato, quanto costa un calcolo per tipo
> di strato, a che blocchi di canali lavora).
>
> **Come la usiamo**: è il costo che guida la potatura in 4b, al posto del conteggio dei calcoli; i parametri sono un
> risultato della tesi di per sé.

**La formula**:

```
latenza = c0 + (numero di strati × α) + Σ (calcoli di ogni strato, a canali arrotondati a g) × β del suo tipo
```

| parametro | cosa rappresenta | come si misura | cosa dice alla potatura |
|---|---|---|---|
| `c0` | la parte fissa della rete: testa, attenzione, concatenazioni | dalle reti intere, è quello che resta | il pavimento: sotto non si scende |
| `α` | il costo di uno strato che esiste, anche se calcola poco | catene di strati minuscoli | togliere canali non lo riduce, togliere strati sì |
| `β` (per tipo: conv 3×3, conv 1×1, depthwise) | il prezzo di un calcolo | pendenza dei probe | **dove** conviene togliere canali |
| `g` | a che blocchi il dispositivo elabora i canali | gradini dei probe | **di quanto** potare alla volta |
| `γ` (facoltativo) | il costo di spostare dati (concat, upsample, somme) | dalle reti intere, se serve | \<esempio>di solito poco\</esempio> |

\<esempio>Esempio sul Pi (numeri inventati, tranne i 28 ms): il pico si scompone in circa 12 ms di parte fissa, circa 5 ms di
costo degli strati e circa 11 ms di calcoli. Il pico potato viene predetto a 23 ms, misurato 22. Si legge che circa
17 dei 28 ms non dipendono dai canali: per questo sul Pi la potatura rende poco.\</esempio>

**Run**:

| run | cosa misura | dove |
|---|---|---|
| P4-1 | probe completi: conv 3×3, conv 1×1, depthwise, a 3–4 dimensioni di mappa prese dalle risoluzioni usate (r/2 … r/32), canali fino a 128 | PI, AGX, DX nelle loro precisioni |
| P4-2 | probe fuori diagonale: canali in ingresso fissi e in uscita variabili, e viceversa | come sopra |
| B4-R | circa 100 reti potate a caso partendo dai modelli da potare (2e), pesi casuali, misurate intere | come sopra |
| L4-1 | stima dei parametri (regressione) e verifica: errore della latenza predetta sulle reti B4-R | — |
| L4-2 | (solo se l'errore supera il 10%) un piccolo MLP che corregge il residuo della formula | — |

**Integrazione nella potatura**: il costo della search passa da PLiNIO, che ha già modelli di latenza per hardware
specifici (con l'arrotondamento dei canali e il gradiente adatto); si aggiunge un costo `latency` sullo stesso
modello e un obiettivo DUCCIO espresso in ms invece che in percentuale di calcoli.

**Cosa ci aspettiamo**: \<esempio>errore sotto il 10% su PI; più difficile su AGX, dove le fusioni e le scelte di TensorRT sono
meno prevedibili (lì potrebbe servire L4-2).\</esempio>

### 4b. Potatura (circa 2 settimane)

> **Cosa testiamo**: se un modello grande potato fino a stare nel budget è più preciso dell'ottimo scalato (T2-O);
> se conta la forma trovata dalla potatura o i pesi ereditati; se guidare la potatura con lo stimatore rende più che
> contare i calcoli.
>
> **Come**: potando il modello di partenza (2e) con due costi diversi (calcoli o stimatore), riallenandolo, e
> confrontandolo alla stessa latenza end-to-end con l'ottimo scalato e con la stessa architettura allenata da zero,
> tutti con 3 seed.
>
> **Conoscenza che produciamo**: quanti punti di mAP dà (o toglie) la potatura rispetto all'ottimo scalato, alla
> stessa latenza; se il guadagno viene dalla forma o dai pesi ereditati; se conoscere l'hardware serve per potare bene.
>
> **Come la usiamo**: si sceglie il modello finale per ogni dispositivo (potato o ottimo scalato) e il metodo da
> raccomandare per ciascun tipo di dispositivo: è la risposta alla domanda 2 della tesi.

**Come si pota**:

- Si parte dal modello indicato dalla decisione 2d (il tetto, o il più piccolo che lo eguaglia), già allenato con 3
  seed in 2e. \<esempio>Su PI è previsto il **nano**: per portarlo alla latenza del pico va tolto circa il 70% dei calcoli
  (stima dalla retta 16,6 ms + 15,2 ms/GMAC).\</esempio>
- **La potatura lavora sulla leva che rende di più (tasso di cambio, 2c)**, alla risoluzione di deploy a cui il
  dispositivo è stretto.
- Il costo che guida la potatura (PIT) è o il **numero di calcoli** (il metodo classico) o lo **stimatore** del
  dispositivo (4a), con canali potati a blocchi di `g`.
- Dopo la potatura il modello si riallena per recuperare precisione (finetune).

**Run: potatura e finetune (seed 0)**. Ogni search `S4-xx` è seguita dal suo finetune `FT4-xx`. `B` è il budget del
modello del dispositivo; `B'` è un budget più stretto (per esempio 20 fps), per vedere se la tendenza regge.

| run | dispositivo | si parte da | costo che guida la potatura | budget | nota |
|---|---|---|---|---|---|
| S4-01 | PI | N | calcoli | `B` | metodo classico |
| S4-02 | PI | N | stimatore di PI | `B` | |
| S4-03 | PI | N | stimatore di PI | `B'` | budget più stretto |
| S4-04 | AGX | dalla decisione 2d | calcoli | `B` | solo se AGX è stretto |
| S4-05 | AGX | dalla decisione 2d | stimatore di AGX | come sopra | |
| S4-06 | DX | dalla decisione 2d | calcoli | `B` | solo se DX è stretto |
| S4-07 | DX | dalla decisione 2d | stimatore di DX | `B` | |

**Run: i confronti (3 seed)**, sulla variante di potatura migliore di ogni dispositivo (scelta sul val con il seed 0):

| run | cosa | seed |
|---|---|---|
| S4-xx + FT4-xx | la stessa potatura e lo stesso finetune, con altri 2 seed | 1, 2 |
| R4-xx | la **stessa architettura trovata dalla potatura, ma allenata da zero** (pesi casuali, ricetta completa) | 0, 1, 2 |
| già fatto in 2c (T2-O) | l'ottimo scalato per il dispositivo | 0, 1, 2 |

Per ogni potato si misurano la latenza end-to-end sul suo dispositivo (`B4-xx`) e la precisione sul val (`E4-xx`).

**Le tre domande a cui rispondono i confronti**:

1. **Potato contro ottimo scalato (T2-O)**: la potatura conviene, anche contro il miglior modello che si ottiene solo
   scegliendo bene larghezza e profondità?
2. **Potato contro la stessa architettura allenata da zero**: conta la forma trovata dalla potatura o i pesi
   ereditati dal modello grande?
3. **Calcoli contro stimatore** (S4-01 contro S4-02, ecc.): serve sapere com'è fatto l'hardware per potare bene?

**Cosa ci aspettiamo**:

\<esempio>

- **PI**: il nano potato batte l'ottimo scalato di poco, o pareggia (il costo fisso limita il guadagno); calcoli e
  stimatore danno risultati simili, perché sul Pi la latenza segue i calcoli.
- **AGX**: togliere canali serve poco; lo stimatore evita i canali non allineati.
- **DX**: lo stimatore allinea i canali ai blocchi dell'NPU, con un guadagno netto rispetto ai calcoli.

\</esempio>
- Se la potatura perde contro l'ottimo scalato, è comunque un risultato: "su questo dispositivo conviene scegliere
  bene il modello piccolo".
- La potatura in profondità (togliere blocchi interi) resta fuori dal piano: si valuta solo se lo stimatore mostra
  che sull'AGX il costo per strato domina.
- **Nota, se c'è tempo: allineamento dei canali su TensorRT.** Lo stimatore (4a) tiene già conto dei blocchi di
  canali (`g`). Come verifica diretta, sull'AGX si confronta un potato con i suoi canali così come escono dalla
  potatura e lo stesso potato con i canali arrotondati a multipli di `g`. Sul Pi l'arrotondamento non dava speedup;
  su GPU HALP (Shen et al., NeurIPS 2022) mostra latenza a gradini e guadagno con canali allineati: è il contrappunto
  da citare.

---

## 12. Fase 5 — Esame finale (circa mezza settimana)

> **Cosa testiamo**: se precisione e latenza dei modelli scelti reggono su dati mai visti e a bassa potenza.
>
> **Come**: aprendo il test una volta sola sui
> modelli scelti con il val, a 640 e alle risoluzioni di deploy; rimisurando latenza ed energia nelle condizioni
> finali; rimisurando a 15 W sull'AGX e alla frequenza sostenibile sul Pi.
>
> **Conoscenza che produciamo**: lo scarto fra val e test (quanto erano ottimistiche le scelte); se ottimi e classifica
> reggono a bassa potenza; la mAP per classe e per grandezza del box, cioè fino a che grandezza il modello vede ancora
> un oggetto.
>
> **Come la usiamo**: nessuna scelta di modello (sono già fatte sul val): dà i numeri e le conclusioni della tesi, e
> dice fin dove valgono; la mAP per grandezza del box diventa un esempio di distanza massima di rilevamento.

**Domanda**: i risultati reggono sui dati mai visti?

Si fa **alla fine di tutto**, una volta sola: dopo il blocco 2, oppure alla **data limite** se il blocco 2 non è
finito (in quel caso senza i potati).

Si prendono i modelli scelti **solo sul val**: per ogni dispositivo l'ottimo scalato (T2-O) e, se c'è, il miglior
potato.

**Prima di aprire il test: soglia di confidenza.** Per i modelli scelti, sul val, si confrontano la soglia generale
(massimo F1) e le soglie ottime per classe, e si fissa la soglia definitiva (regole fisse, metriche).

| run | cosa | ripetizioni |
|---|---|---|
| E5-1 | precisione sul test di ogni modello scelto (3 seed), a 640 e sui set di test alle risoluzioni di deploy: mAP, e precision, recall e falsi positivi per fotogramma alla soglia di deploy scelta sul val | una volta sola |
| E5-2 | mAP sul test per classe e per fascia di grandezza del box (16–32, 32–64, 64–96, sopra 96 px) | una volta sola, dalle stesse predizioni di E5-1 |
| B5-1 | latenza end-to-end finale sul proprio dispositivo | 5 passate, condizioni registrate |
| B5-2 | energia per inferenza sull'AGX (sensori di bordo) | 3 passate |
| B5-3 | robustezza: i modelli scelti rimisurati sull'AGX a 15 W e sul PI alla frequenza sostenibile senza throttling | 3 passate |

B5-3 serve solo a dire se la decisione 2d e la classifica dei modelli reggono a bassa potenza: se sì, basta una
frase nella tesi; se no, è un risultato da riportare.

**Risoluzione e distanza (alla fine).** Nelle fasi 2–4 la risoluzione non si sceglie: per ognuna c'è il suo ottimo,
e la mAP non premia la risoluzione alta: gli oggetti dei set di deploy hanno la stessa grandezza in pixel a ogni
risoluzione. È voluto, per due motivi:

- con la stessa camera, a risoluzione più alta lo stesso oggetto occupa più pixel e il modello per forza va meglio:
  la mAP mescolerebbe l'effetto del modello con quello della grandezza dell'oggetto. A grandezza invariata la mAP
  confronta solo i modelli, e il vantaggio della risoluzione si esprime in modo pulito come **raggio di rilevamento
  utile** più grande;
- dati veri a 1080p per tutte le classi non ci sono.

Il raggio di rilevamento si valuta solo qui, con una tabella per dispositivo:

| risoluzione | miglior modello nel budget | mAP50-95 | `p_min` di quel modello | distanza |
|---|---|---|---|---|

`p_min` è la fascia di grandezza più piccola in cui l'AP di quel modello resta accettabile (E5-2): un modello più
piccolo può vedere bene solo da 32 px in su. La tabella dice se a 1080p, con un modello più piccolo, si vede ancora
più lontano che a 480p con uno più grande. La distanza dipende dalla camera, quindi non è una metrica della tesi: per
una camera ipotetica con campo visivo orizzontale `FOV` e larghezza del fotogramma `W` pixel, un drone di grandezza
`S` metri si vede fino a circa

```
distanza ≈ S · W / (2 · p_min · tan(FOV / 2))
```

\<esempio>Drone di 0,35 m, camera 1080p (W = 1920) con FOV di 30°, p_min = 16 px: circa 80 m; a 720p circa 50 m. Se a
1080p il modello nel budget vede bene solo da 32 px, la distanza scende a circa 40 m: meno che a 720p.\</esempio>

### Cosa ci aspettiamo

\<esempio>Il test qualche punto sotto il val (lo split per video è più severo), ma con lo stesso ordinamento fra i metodi.\</esempio>

---

## 13. Figure di sintesi

Tre figure riassumono il blocco 1 e l'ingresso al blocco 2. Sono tutte a **piccoli multipli**: un pannello per ogni
combinazione di precisione (righe: fp16, int8) e profondità (colonne: D101, D126, D170), e dentro ogni
pannello la griglia discreta larghezza × risoluzione di deploy (480p, 720p, 1080p; sezione 4). Se ne fa **una per dispositivo**; affiancate, mostrano
come il dispositivo cambia la risposta.

| figura | cosa mostra | quando si fa | a quale scelta serve |
|---|---|---|---|
| **A. Chi sta nel budget** | ogni configurazione come casella colorata: entro il budget, entro ±5% dal budget, oltre il budget | dopo la griglia di latenza (2a) | dove passano i confini del budget; dove mettere i training fuori asse (2b); quanto l'int8 e la profondità spostano il confine |
| **B. Dove sta l'ottimo** | le configurazioni nel budget come pallini colorati per mAP predetta (sul set di val di quella risoluzione di deploy), fuori budget in grigio; stella = ottimo di ogni risoluzione (il migliore fra tutte le precisioni e profondità a quella risoluzione). Non c'è un ottimo globale: la mAP non premia la risoluzione, il confronto fra risoluzioni si fa con la distanza in fase 5 | dopo il modello di precisione (2c); si rifà dopo la fase 3 con le latenze int8 | quale configurazione allenare come ottimo (e quali candidate dentro l'intorno di errore, vedi la nota pending in 2c) |
| **C. Candidati alla potatura** | come B, più gli anelli: configurazioni fuori budget ma più precise dell'ottimo della stessa risoluzione di più di 1 punto e con un taglio di latenza realistico (≤ 50%); rombo = miglior candidato di ogni risoluzione (più mAP per punto di taglio), con il taglio da fare | alla decisione 2d, prima del blocco 2 | se la potatura ha senso su quel dispositivo (ci sono anelli?) e da quale modello partire |

Le figure si usano anche nella tesi: A e B nel capitolo dei risultati del blocco 1, C come ponte verso la potatura.
Per spiegare l'idea nel capitolo del metodo basta un solo pannello di B (una precisione, una profondità).

Lo script è `tools/figure_sintesi.py` (`python tools/figure_sintesi.py OUT_DIR`), oggi con dati di esempio
inventati: quando arrivano i dati veri si sostituiscono le funzioni `lat` (fase 2a) e `mAP` (fase 2c). Parametri da fissare: il margine "a pelo" (±5%) e il taglio massimo per la potatura (50%); l'intorno sulla mAP è
1 punto (criterio di scelta, 2c).

---

## 14. Rischi

| rischio | quando lo si scopre | piano B |
|---|---|---|
| il tempo non basta per il blocco 2 | fine del blocco 1 | la tesi è il blocco 1; il test si fa alla data limite senza i potati |
| operatori di YOLO26 non supportati dalla DX-M1 | X1 | sostituirli; altrimenti DX diventa descrittivo |
| AGX non disponibile | fase 1 | si lavora su PI e DX |
| aggiunte al dataset (uccelli e aerei grandi) non trovate o troppo lente da etichettare | fase 0, passo 5 | si usa quello che c'è; le celle scoperte (per esempio uccelli grandi) si dichiarano come limite |
| SAM 2 non rende aderenti i box degli oggetti minuscoli (sagome vuote o sbagliate) | fase 0, passo 2 | soglia di contrasto locale su cielo uniforme; revisione a mano dei casi rimasti |
| sul Pi non sta nessun modello a 720p e 1080p | 2a | sul Pi solo 480p: è un risultato |
| lo stimatore sbaglia più del 10% sull'AGX | 4a (L4-1) | MLP sul residuo (L4-2); altrimenti sull'AGX si pota solo con i calcoli |
| compilazione DX-COM troppo lenta per i probe completi | X1 | probe e reti potate a caso ridotti sulla DX |

---

## 15. Scelte da confermare: stato

| # | scelta | stato |
|---|---|---|
| 1 | **ricetta di training** | **decisa**: una ricetta unica per tutti i training, niente versione ridotta per la mappatura; 1000 epoche con early stopping a patience 200 |
| 2 | **software di inferenza per dispositivo** | **decisa** con X2: sul PI OpenVINO per l'fp16 e onnxruntime per l'int8, perché OpenVINO int8 su ARM è più lento (41,1 contro 29,4 ms sul pico); AGX TensorRT (int8 con la ricetta standard di Ultralytics), DX DX-COM/DX-RT |
| 3 | **ottimo predetto al posto del riempimento** (2c) | **accettata** |
| 4 | **secondo budget più stretto** (`B'`, es. 20 fps) in fase 4 | aperta, si decide all'inizio del blocco 2 |
| 5 | **latenza misurata con pesi casuali**, subito (fasi 1 e 2) | **accettata**, insieme alla 3: la latenza delle candidate dell'ottimo si misura invece di predirla |
| 6 | **alta risoluzione su S solo se sul nano aiuta** | **superata**: la risoluzione non si allena più, è una condizione di deploy |
| 7 | **aggiunte al dataset come ritagli 640×640 a scala nativa** (fase 0) | **superata**: niente fonti ad alta risoluzione, tutte le immagini sono 640×512 come Svanström |
| 8 | **quali ottimi confermare con 3 seed**: tutti quelli distinti per dispositivo e risoluzione di deploy, o uno per dispositivo | **aperta**, si decide in 2c quando si sa quanti sono |
| 9 | **ottimizzatore, learning rate e augmentation** della ricetta unica | **decisi** MuSGD, `lr0` 0.01, `lrf` 0.01, batch secondo la GPU, mosaic acceso, `scale` 0.25 (regole fisse, training e seed) |
| 10 | **soglia di confidenza di deploy** | **decisa**: quella che massimizza l'F1 sul val, una per tutte le classi, per modello e risoluzione di deploy; applicata al test in fase 5 senza ricalcolarla (regole fisse, metriche) |
| 11 | **minimo di istanze e di video per cella** della matrice classe × fascia (fase 0, passo 1) | **aperta**, in fase 0 |

### Sulla scelta 1: la ricetta

- **Riferimento Ultralytics**: il default è 100 epoche (pensato per il fine-tuning da pesi COCO); per i training da
  zero la guida di Ultralytics suggerisce di partire da 300 e allungare se non c'è overfitting. Gli altri default:
  batch 16, `lr0` 0.01, `lrf` 0.01, momentum 0.937, weight decay 5e-4, warmup 3 epoche, mosaic 1.0 con
  `close_mosaic` 10, `scale` 0.5, `translate` 0.1, flip orizzontale 0.5, mixup 0. Con `optimizer=auto` sceglie MuSGD
  (lr 0.01) sopra le 10 000 iterazioni, altrimenti AdamW.
- **Perché una ricetta sola**: i modelli piccoli convergono più lentamente dei grandi. Una mappatura più corta li
  penalizzerebbe di più, renderebbe il trend della larghezza più ripido del vero e sposterebbe l'ottimo verso modelli
  più grandi: cioè falserebbe il risultato principale del blocco 1. \<esempio>Il risparmio sarebbe di 2–3 giorni su un piano di
  10–14 settimane\</esempio>, e con la ricetta unica i training della mappatura si riusano come seed 0 dei confronti.
- **Lunghezza**: 1000 epoche con patience 200. Su AOD4 il pico arrivava al 99% del suo massimo verso l'epoca 620 e il
  nano verso la 400–430; su Svanström tutti e tre i modelli piccoli toccavano il massimo verso l'epoca 15–30 e poi
  calavano. 1000 epoche coprono il caso lento, la patience evita di sprecare GPU nel caso veloce.

### Da zero o da COCO

Tutto **da zero**. I pesi COCO esistono solo per n, s, m: le nostre varianti (F, P, profondità D101 o D170, gli ottimi
con larghezze intermedie) non li hanno. Mescolare modelli preaddestrati e non mescolerebbe l'effetto dell'architettura
con quello del preaddestramento, e i trend della fase 2 non sarebbero più leggibili. Preaddestrare tutto su COCO
costerebbe giorni di GPU per modello. \<esempio>Con un dataset grande e un training lungo, partire da zero si avvicina al
preaddestrato.\</esempio> Il guadagno del preaddestramento si può citare come limite, o misurare con un solo run (lo stesso
modello da zero e da COCO) se resta tempo.
