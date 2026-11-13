# NEW MASTERPLAN — detection di oggetti aerei su dispositivi edge, guidata dall'hardware

**Convenzione.** Il testo fra `<esempio>` e `</esempio>` è un'**ipotesi**, un'attesa o un esempio con numeri
inventati: non è ancora verificato. Tutto il resto è una **decisione presa** o un **fatto misurato**; per i numeri
misurati è indicato dove (per esempio "sul Pi", "sulla RTX 6000"). Le stime di tempo sono sempre indicate come tali.

**Perimetro.** Si lavora solo sul **Raspberry Pi 5 (PI)**. AGX e DX non fanno parte del percorso principale: quello che le riguarda sta nella sezione 16, «Se abbiamo tempo», e dove resta una menzione è marcata *(se abbiamo tempo)*.

## 1. L'idea in breve

Abbiamo tre dispositivi edge molto diversi fra loro; la tesi si fa sul primo, gli altri due sono *se abbiamo tempo*:

- un **Raspberry Pi 5**, che calcola con la sua CPU;
- una **Jetson AGX Orin**, che calcola con una GPU (se abbiamo tempo);
- un **Raspberry Pi 5 con l'acceleratore DX-M1**, che calcola con un NPU collegato via PCIe (se abbiamo tempo).

Sul PI (e, se abbiamo tempo, sugli altri due) vogliamo far girare un detector per quattro classi (**drone, aereo, elicottero, uccello**) che regga
**30 fotogrammi al secondo**. La tesi risponde a due domande:

1. **Sul PI, qual è il detector più preciso che sta nei 30 fps?**
2. **Dove il budget è stretto, conviene prendere un modello grande e potarlo, oppure basta usare fin dall'inizio un
   modello più piccolo?**

Quello che abbiamo già visto con le prove sulla pipeline attuale suggerisce che la risposta cambia da dispositivo a
dispositivo: sulla CPU del Pi la latenza segue i calcoli; sulla GPU della workstation (RTX 6000) segue soprattutto il
numero di strati e l'allineamento dei canali; \<esempio>sull'NPU probabilmente la granularità con cui elabora i
canali\</esempio>. Il messaggio atteso della tesi è quindi:

> \<esempio>*"Su ogni dispositivo edge conta una cosa diversa: ecco quale, e quando la potatura conviene davvero."*\</esempio>

Con il solo PI il messaggio si restringe a quello che si vede sul PI; il confronto fra dispositivi è in «Se abbiamo tempo» (sezione 16).

---

### Struttura: due blocchi

La tesi è divisa in due blocchi, per non rischiare di finire il tempo prima di avere un risultato completo.

- **Blocco 1 — il nucleo (circa 7 settimane)**: fasi 0, 1 (versione di base), 2 e 3, con le due incognite X1 e X2
  risolte subito prima della fase che ne ha bisogno. Risponde alla
  domanda 1: *sul PI e a ogni risoluzione di deploy, qual è la ricetta migliore (larghezza, profondità,
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
| 0 | se i dati sono adatti | dataset custom a 640×512 (VEDETTA, repo vedetta) con **box corretti** e bilanciato su classe e grandezza, split per gruppi di sessioni; set di val alle risoluzioni di deploy | il dataset e la ricetta valgono per tutta la tesi; i set di val di deploy danno la mAP usata nella scelta dell'ottimo (2c) |
| X1 (prima della fase 1, **se abbiamo tempo**) | se le nostre varianti girano sulla DX-M1 | operatori supportati; tempo di compilazione | decide se DX resta nel piano e quanto costano le misure sulla DX |
| 1 (base) | quanto sono ripetibili le misure; quanto dei 33 ms resta al modello; come si comporta ogni dispositivo | la **soglia di rumore** della latenza; il **budget del modello**; i tratti di base di ogni dispositivo (gradini, costo per strato, granularità N) | la soglia dice quando una differenza di latenza è vera; il budget è il vincolo dell'ottimo (2c) e della decisione (2d); i tratti spiegano i trend della fase 2 e sono il punto di partenza dello stimatore (4a) |
| X2 (prima della fase 2, **risolta**) | se l'int8 funziona sulla testa one-to-one, e come va sul PI | causa della mAP 0 (coda della testa quantizzata); ricetta int8; prova preliminare: sul PI l'int8 non è più veloce dell'fp16 | sul PI: quali nodi quantizzare nella 2a e nella fase 3, quale software usare per l'int8 sul PI; l'attesa da verificare in 2a e fase 3 |
| 2 (2a–2b) | quanto sono precisi e veloci i modelli normali | i **trend**: mAP e latenza in funzione di larghezza e profondità, a ogni risoluzione di deploy, per dispositivo | materia prima della fase 2c |
| **2 (2c)** | **qual è la configurazione migliore nel budget** | **tasso di cambio** di ogni leva (mAP per ms), un **modello di precisione**, l'**ottimo predetto e verificato** per dispositivo e risoluzione di deploy | l'ottimo è la **baseline** da battere; la leva più redditizia dice **su cosa deve lavorare la potatura**; il gap fra ottimo e modello migliore senza budget dice **se la potatura serve** (2d) |
| 3 | quanto rende l'int8 | perdita di mAP e guadagno di latenza dell'int8, per modello e dispositivo | si ricalcola l'ottimo (2c) con le latenze int8; chiude il blocco 1; si sceglie la precisione della fase 4 |
| 4a (blocco 2) | se una formula predice la latenza di una rete potata | lo **stimatore di latenza** del PI (LUT per strato e composizione nella rete), con il suo errore | è il costo che guida la potatura in 4b |
| 4b (blocco 2) | se potare batte lo scaling ottimizzato | quanto rende la potatura rispetto all'ottimo scalato; se conta la forma o i pesi ereditati; se lo stimatore rende più dei calcoli | sceglie il modello finale per dispositivo e il metodo da raccomandare |
| 5 (alla fine di tutto) | se i risultati reggono su dati mai visti | lo scarto fra val e test; la robustezza a bassa potenza; la mAP per classe e per grandezza del box | i numeri e le conclusioni della tesi; un esempio di distanza massima di rilevamento |


---

## 2. Regole fisse

Queste regole valgono per tutta la tesi. Servono a rendere i confronti onesti e a non doverli rifare.

### Dataset

- **Un solo dataset**, con le quattro classi, usato per tutta la tesi: quello congelato della repo
  `~/Desktop/vedetta` (`data/output`, qui in `data/air-detection/`, nome storico). Regole, fonti, procedura, split e limiti
  sono nel README di quella repo (sezioni Regole, Fonti e licenze, Procedura, Limiti dichiarati) e non si ripetono
  qui.
- In breve: immagini 640×512 da sei fonti pubbliche di oggetti più due di background, box aderenti rivisti a mano,
  split per gruppi di sessioni di ripresa (nessuna ripresa in due split), stessa distribuzione obiettivo delle
  grandezze per tutte le classi.
- **Il test si apre una volta sola**, alla fine (fase 5). Tutte le scelte si fanno guardando il val.

### Budget di latenza

- Il budget è **30 fps, cioè 33 ms end-to-end**, uguale per tutti i dispositivi.
- **End-to-end** vuol dire: dal fotogramma uint8 **già alla risoluzione di deploy** fino ai box finali. Comprende:
  - il pre-processing (uint8 → float, divisione per 255, cambio di layout);
  - i trasferimenti verso l'acceleratore (nessuno sul PI; verso la GPU sulla Jetson e sul PCIe per la DX-M1, se abbiamo tempo);
  - l'inferenza;
  - il post-processing (decodifica della testa, soglia di confidenza, box in coordinate dell'immagine).
- **Restano fuori** l'acquisizione dalla camera e il ridimensionamento: si assume che la camera lavori già alla
  risoluzione di deploy.
- Pre-processing, post-processing e trasferimenti **non dipendono dal modello** (a parità di risoluzione): si
  misurano una volta e si sottraggono dal budget. Quello che resta è il **budget del modello**:
  `budget del modello = 33 ms − (pre + post + trasferimenti)`.
- **Sul PI il percorso principale è a 640×640** (vedi "Risoluzione"). Il budget del modello a 640 è **da fissare**:
  lo sceglie l'utente dalla tabella `reports/budget_latenza_rpi5` (`experiments/latenza/griglia_rpi5/rifai.sh`). Il valore 28 ms che compare
  negli esempi è quello ottenuto a 480p: vale solo come esempio.

### Risoluzione

**Training a risoluzione fissa, deploy a risoluzione variabile.**

- **Training**: tutti i modelli si allenano a **640×640**, la scala del dataset (nativo 640×512). Non si allena a
  nessun'altra risoluzione, e non si ingrandisce mai un'immagine (non aggiunge dettagli).
- **Deploy principale: 640×640, risoluzione fittizia** (decisione dell'utente). È una scelta di comodo: LUT, B4-R,
  pavimento e stimatore di latenza esistono solo a 640, e a 640 il fotogramma ha la stessa grandezza del training,
  quindi il `val` standard basta. Tutte le decisioni (2c, 2d, fase 4) si prendono a 640 sul PI.
- **Deploy a più risoluzioni (estensione, se abbiamo tempo)**: camera **16:9** a tre risoluzioni. Il modello riceve il fotogramma intero; l'ingresso va arrotondato a
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
  val costruiti apposta (fase 0, Set di deploy). \<esempio>Ci aspettiamo che cambi poco: lo studio già fatto sulle immagini di
  background mostra che i modelli piccoli non sbagliano sul cielo vuoto; il loro problema sono le detection multiple
  sullo stesso oggetto, che non dipendono dalla grandezza del fotogramma.\</esempio>
- **La risoluzione non è una leva da allenare, è una condizione di deploy**: cambia la latenza (in modo circa
  quadratico) e la mAP misurata sui set di deploy, mai il training.
- **Ritagli quadrati del tracking: 128×128, 256×256, 512×512.** Il detector gira sul fotogramma intero per agganciare
  il bersaglio; poi, a ogni fotogramma, si taglia un quadrato **a risoluzione nativa** attorno al soggetto e lo si dà
  al modello: l'oggetto resta grande in pixel come nel training, il calcolo scende (il 128 ha circa 1/25 dei pixel
  del 640×640). Per ora entrano **solo nella griglia di latenza** (2a); la mAP sui ritagli e il ritorno al fotogramma
  intero quando il bersaglio si perde sono da definire. \<esempio>Il ritaglio regge oggetti fino a circa metà del suo
  lato (64 px nel 128); il 512 non sta in un fotogramma 480p (alto 480), quindi serve dal 720p in su.\</esempio>

### Training e seed

- **Una sola ricetta di training per tutti i training della tesi**, fissata in fase 0 e mai più cambiata
  (ottimizzatore, learning rate, augmentation, batch **e numero di epoche**), uguale per tutti i modelli:
  - **MuSGD**, `lr0` 0.01, `lrf` 0.01 (lr finale 1e-4, decadimento lineare): la combinazione migliore dei training
    da zero del progetto vecchio (ablation sull'ottimizzatore, pico su AOD4: MuSGD 0,649 di mAP50-95, SGD 0,643,
    Adam 0,637); 0.01 è anche il valore che `optimizer=auto` sceglie per MuSGD;
  - momentum, weight decay e warmup di default (0.937, 5e-4, 3 epoche);
  - **batch** scelto in base alla GPU, lo stesso per tutti i modelli: Ultralytics accumula i gradienti fino a un
    batch nominale di 64 (`nbs`), quindi con batch ≤ 64 l'aggiornamento dei pesi resta equivalente;
  - **augmentation** (dal 28/10/2026, `conf/train/default.yaml`): i valori di `yolo26n.pt` (train_args, COCO), fra
    cui mosaic 0.91, `scale` 0.56, `degrees` 1.1, `shear` 1.5, `hsv_v` 0.57, flip verticale 0.06; in più `cutmix`
    0.1 e il **degrado della camera** con Albumentations (motion blur e blur gaussiano 0.1, rumore 0.1, JPEG 50–95
    0.2, kernel fino a 5 px per non cancellare gli uccelli da 8–16 px). Sul pico (`59d8534a`) mAP50 0,842 contro
    0,819 della ricetta precedente;
  - seed fisso e `nms=False` (testa). Niente ricette ridotte per
  esplorare: a poche epoche i modelli piccoli sono più indietro dei grandi e i trend risulterebbero falsati. Così le
  differenze fra modelli dipendono solo dal modello, e ogni training della mappatura vale anche come seed 0 dei
  confronti finali.
- **Lunghezza: 300 epoche, con early stopping a patience 200** come rete di sicurezza. 300 epoche sono un
  compromesso con il tempo di GPU (il pico circa 4,5 ore, lo small circa 6,5 sulla RTX 6000), e sul dataset del
  28/10/2026 bastano: il massimo sul val arriva all'epoca 225 per il pico, 233 per il nano e 187 per lo small, che
  dopo cala un poco. Limite da dichiarare: su AOD4 il pico guadagnava ancora 0,02 fra l'epoca 300 e la 900. Il mosaic
  si spegne nelle ultime 10 epoche (`close_mosaic` di Ultralytics): i run fermati prima dall'early stopping non fanno
  questa parte, ed è un limite da dichiarare. `best.pt` si sceglie comunque sul val. Per ogni run si registra a che
  epoca ha toccato il massimo: un modello della griglia che ci arriva presto è il segnale che il dataset lo limita, e
  va riportato.
- I training esplorativi usano **un solo seed**. I modelli che entrano nei confronti finali si allenano con
  **3 seed** e si riportano come media ± deviazione standard: con un seed solo, una differenza di 0,5–1 punto di
  mAP può essere solo fortuna.

### Dispositivi

- **Una sola configurazione per dispositivo, quella di deploy**: tutti i core, frequenza massima, profilo di potenza
  massimo sulla Jetson (se abbiamo tempo). Core e frequenze non si variano: la tesi studia il modello, non il dispositivo.
- Alla fine (fase 5) un solo controllo di robustezza: i modelli scelti rimisurati a bassa potenza.

### Misure

- Latenza = mediana di 3 passate, alternando i modelli fra una passata e l'altra (così un rallentamento temporaneo
  del dispositivo non colpisce sempre lo stesso modello).
- A ogni misura si registrano le condizioni: frequenze, temperatura, governor, clock della GPU.

### Precisioni

- In fase 2a si misura la latenza in **tutte le precisioni supportate** da ogni dispositivo (PI: fp32, fp16,
  int8; AGX e DX, se abbiamo tempo: sezione 16). Da lì in poi il **deploy considera solo fp16 e int8**: \<esempio>l'fp32 non conviene mai rispetto
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
  - nel budget end-to-end l'NMS andrebbe implementata e misurata su ogni dispositivo (sulla DX, se abbiamo tempo, girerebbe sulla CPU
    del Pi);
  - si valuta lo stesso modello che si esporta;
  - in precisione le due teste sono pari (progetto vecchio, mAP50-95 one-to-many / one-to-one: pico 0,492 / 0,491,
    nano 0,528 / 0,526).
- Il costo sono le detection doppie, che restano senza NMS: si contano come metrica. Come riferimento, ogni E2 valuta
  anche la one-to-many con NMS dagli stessi pesi.

### Metriche di precisione

- **Campo della tesi: oggetti di almeno 8 px.** La grandezza di un oggetto è la radice dell'area del suo box in
  pixel, √(w·h), come in COCO. Le immagini che contengono un oggetto sotto gli 8 px si tolgono da **tutti gli split**
  costruendo il dataset (togliere solo la label insegnerebbe al modello che lì non c'è niente), così le metriche
  standard di Ultralytics valgono già solo per oggetti ≥ 8 px, senza valutazioni personalizzate. Limite da
  dichiarare: val e test non contengono droni più lontani di così, che in deploy invece ci saranno.
  **Perché 8 px:**
  - **limite dell'architettura**: la testa più fine di YOLO26 lavora a passo 8, cioè una cella ogni 8×8 pixel
    d'ingresso, e ogni cella può proporre un oggetto. Sotto gli 8 px un oggetto non copre nemmeno una cella;
    Ultralytics allarga a 8 px in training i box più piccoli per dargli almeno una cella (`ultralytics/utils/tal.py`),
    ma non può aggiungere dettaglio;
  - **letteratura**: sui benchmark di oggetti minuscoli (AI-TOD, Wang et al. 2021) i detector come YOLO trovano
    quasi niente sotto gli 8 px e poco, con precisione bassa, fra 8 e 16 px; diventano usabili sopra i 16 e normali
    sopra i 32;
  - **affidabilità delle label**: un box di pochi pixel è difficile da disegnare in modo preciso e coerente, e
    l'errore di qualche pixel pesa molto sull'IoU;
  - **perché anche 8–16 px in val e test** (decisione del 19/10/2026, prima il campo partiva da 16 px): sono i droni
    lontani, il caso che conta per la torretta. Il loro peso nella mAP è fissato dalla distribuzione obiettivo del
    dataset (densità ∝ s^−0,3 fra 8 e 320 px, uguale per tutte le classi; `vedetta/README.md`, Regole).
- mAP50 e mAP50-95, come al solito, sugli oggetti ≥ 8 px, sul val a 640 e sui tre set di val alle risoluzioni di
  deploy. È la **metrica principale** di tutte le scelte;
- **recall sugli oggetti piccoli** (8–32 px), sul test in fase 5: sono i droni lontani, il caso più difficile;
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
- **mAP per classe e per fascia di grandezza del box**, con fasce **8–16, 16–32, 32–64, 64–96, sopra 96 px** (le
  stesse del dataset): 8–16, 16–32 e 32–64 come "tiny", "small" e "medium" di AI-TOD, 64–96 e sopra 96 come "medium" e "large" di COCO. Dice fino a che grandezza
  il modello riconosce ancora un oggetto. Si riporta sul test in fase 5.

---

## 3. Dispositivi

| sigla | dispositivo | dove calcola | software di inferenza | precisioni | configurazione |
|---|---|---|---|---|---|
| **PI** | Raspberry Pi 5 | CPU Cortex-A76, 4 core | OpenVINO (fp16), onnxruntime (int8, X2) | fp16, int8 | 4 core, 2.4 GHz |

Cosa ci aspettiamo da ciascuno, in base alle prove già fatte:

- **PI**: la latenza cresce quasi in proporzione ai calcoli, più un costo fisso interno al modello. Dalle misure
  attuali (OpenVINO fp16, 4 core): circa 16,6 ms + 15,2 ms per ogni GMAC. Il pico sta a ~28 ms, il nano a ~57 ms.
  L'allineamento dei canali non conta.
AGX e DX (se abbiamo tempo): sezione 16.

---

## 4. Modelli

Si usa la famiglia **YOLO26** con la testa **one-to-one** (senza NMS). Nel training si variano due cose:

- la **larghezza** (quanti canali ha ogni strato);
- la **profondità**, cioè il **numero di strati convoluzionali** della rete (definizione sotto).

La **risoluzione** non si varia nel training (sempre 640×640): è la condizione di deploy (640×640 principale; 480p, 720p, 1080p, regole
fisse), su cui si misurano latenza e mAP. Il percorso principale è a 640×640; le tre risoluzioni di deploy sono un'estensione.

**Sigle**: ogni modello ha una sigla = lettera della larghezza + segno della profondità. Lettere: F 0.0625, FP 0.09375,
P 0.125, PN 0.1875, N 0.25, NS 0.375, S 0.5, SM 0.75, M 1.0 (le intermedie prendono le lettere delle due vicine). Segno:
`-` D101, nessuno D126, `+` D170. Esempi: N- = 0.25 D101, NS+ = 0.375 D170. I modelli usati più spesso:

| sigla | nome | larghezza | profondità | ruolo |
|---|---|---|---|---|
| F | femto | 0.0625 | D126 | il più piccolo, sotto il pico |
| P | pico | 0.125 | D126 | candidato per PI |
| N- | nano corto | 0.25 | D101 | prova della profondità |
| **N** | **nano** | **0.25** | **D126** | **centro della griglia**; \<esempio>previsto come modello da potare su PI (lo decide la 2d)\</esempio> |
| N+ | nano profondo | 0.25 | D170 | prova della profondità |
| S- | small corto | 0.5 | D101 | prova della profondità |
| S | small | 0.5 | D126 | secondo centro; candidato per AGX e DX (se abbiamo tempo) |
| M | medium | 1.0 | D126 | il più grande; solo latenza (fuori budget sul PI, 2b) |

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
riferimento per AGX e DX (se abbiamo tempo), mentre il nano lo è per PI.

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

La griglia di training usa le prime 8 larghezze, fino a 0.75; la M (1.0) si misura solo in latenza, perché sul PI
è fuori budget di un ordine di grandezza (2b). **Le larghezze intermedie danno
canali multipli di 3** (3, 6, 12, 24, 48, 96): lo 0.09375 ha solo il 62% di conv a multipli di 8 e il 40% a multipli
di 16, e femto e 0.09375 hanno 9 conv sotto gli 8 canali. \<esempio>Su GPU e NPU, che lavorano a blocchi di canali, queste
larghezze rischiano di costare più delle potenze di 2 a parità di calcoli\</esempio>: la griglia di latenza della 2a lo misura,
ed è un risultato da riportare.

Le risoluzioni sono **640×640** (deploy principale, fittizio) e le tre dell'estensione: **480p, 720p, 1080p** (ingressi 864×480, 1280×736, 1920×1088; regole fisse),
più i tre ritagli quadrati del tracking (**`crop128`, `crop256`, `crop512`**), per ora solo nella griglia di latenza.

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

> **Cosa testiamo**: se il dataset copre ogni classe a ogni grandezza, con box corretti, ed è diviso senza leakage.
>
> **Come**: il dataset VEDETTA si costruisce nella repo vedetta (README, Procedura); qui si preparano i set di val alle
> risoluzioni di deploy e il resto del protocollo.
>
> **Conoscenza che produciamo**: la distribuzione del dataset per classe, grandezza e split; i set di val di deploy.
>
> **Come la usiamo**: il dataset e la ricetta valgono per tutta la tesi; i buchi si dichiarano come limite; i set di
> val di deploy danno la mAP con cui si sceglie l'ottimo (2c).

**Domanda**: i dati sono adatti, e il protocollo è definito?

Le misure sui dispositivi della fase 1 e la griglia di latenza della 2a usano pesi casuali, quindi **non aspettano il
dataset**: girano in parallelo a questa fase.

### Il dataset

Costruito e congelato nella repo `~/Desktop/vedetta`: l'ordine delle operazioni è nel suo README
(Procedura), la procedura generale in `docs/procedura_ideale.md`. Qui resta solo quello che riguarda yoloeval.

### Set di deploy

A 640 il set di deploy è il `val` standard. Per l'estensione servono set di val alle risoluzioni di deploy (`val_480p`, `val_720p`, `val_1080p`), costruiti una volta su disco e uguali per
tutti i modelli. Ogni fotogramma di deploy è un **mosaico**: un frame di val con il suo oggetto, in posizione casuale
a seed fisso, e intorno frame di **solo sfondo dello stesso video** (oggetto cancellato con l'inpainting), così il
cielo è lo stesso e non ci sono giunture fra cieli diversi. Per il 1080p: 3×2 frame da 640×512 (1920×1024) più una
striscia di sfondo fino a 1088; per 720p e 480p si ritaglia di conseguenza (a 480p il frame con l'oggetto, alto 512,
si taglia in modo che l'oggetto resti intero). Nessun ridimensionamento: gli oggetti hanno la stessa grandezza in
pixel a ogni risoluzione. Il frame con l'oggetto è **uno solo**: come un fotogramma vero della torretta (un drone, o
nessuno, e tanto cielo vuoto); con più frame pieni i falsi allarmi peserebbero meno del vero e la mAP sarebbe
ottimista. Si valutano con il normale `val` a `imgsz` di deploy, sui modelli PyTorch e sugli export; gli export si
valutano con il runtime vero sul dispositivo. Lo stesso si prepara per il test (`test_480p`, …), chiuso fino alla
fase 5. La validazione di Ultralytics non fa mosaic (è solo augmentation di training, casuale e quadrata), per
questo i set si costruiscono a parte.

### Il resto del protocollo

- **Valutazione**: il `val` standard di Ultralytics (mAP e AP per classe, quindi anche l'AP dei droni). La mAP per
  fascia si calcola qui, in yoloeval, dalle predizioni del `val` (il dataset non ha sottoinsiemi per fascia: perdevano
  i box delle formazioni). Falsi positivi per fotogramma e detection multiple si contano dalle
  predizioni salvate del `val`, con un'analisi breve.
- **YAML dei modelli**: F, P, N, S, M e le larghezze intermedie con un nome di scala proprio (mai `m`, `l`, `x`,
  che forzano `c3k=True`, sezione 4); D101 (`c3k=False` in tutti i C3k2) e D170 (moltiplicatore 1.0). Controllo
  automatico del numero di conv (101, 126, 170) a ogni costruzione.
- **Correzioni del codice di misura (fatte il 13/10/2026, da verificare sui dispositivi)**: TensorRT ora misura i
  trasferimenti (tolto `--noDataTransfers=false`, che trtexec leggeva come `--noDataTransfers`; il parser segnala un
  errore se H2D vale 0); OpenVINO int8 usa `infer_precision` f16 su aarch64 (con f32 sul PI cadeva nei kernel di
  riferimento, 2 s per inferenza) e f32 su x86. Se abbiamo tempo, prima di B1-2 sull'AGX: una run per controllare che H2D e D2H non
  siano 0 (con la memoria unificata potrebbero esserlo davvero, e allora il controllo del parser va rivisto).
- **Ricetta di training** fissata una volta per tutte (regole fisse): MuSGD, `lr0` 0.01, `lrf` 0.01, batch secondo la
  GPU, augmentation di `yolo26n.pt` con cutmix e degrado della camera (regole fisse), 300 epoche con patience 200. Il controllo sulla lunghezza è gratis: è la curva del nano
  della mappatura (T2-14).
- **Misura end-to-end** sul PI (AGX e DX: sezione 16), secondo la definizione delle regole fisse, con il controllo automatico
  delle condizioni (frequenze, temperatura, clock).

### Run

Nessuna misura: la fase 0 è lavoro sui dati e sul protocollo. Le misure di ripetibilità della latenza sono in fase 1
(B1-0).

### Cosa ne esce

Il dataset congelato con box aderenti e la sua distribuzione per classe e grandezza,
i set di val (e di test) alle risoluzioni di deploy, la ricetta di training.

---

## 6. Incognita X1 — La DX-M1 riesce a eseguire i nostri modelli? (prima della fase 1; se abbiamo tempo)

**Se abbiamo tempo.** Riguarda solo la DX, fuori dal percorso principale: si fa solo se si decide di usarla.

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
> servono a **spiegare** i trend della fase 2 (per esempio perché sul PI la latenza segue i calcoli); se si fa il
> blocco 2, sono il punto di partenza dello stimatore di latenza (4a).

**Domanda**: quanto sono affidabili le misure, quanto tempo resta al modello su ogni dispositivo, e come si comporta
ogni dispositivo?

Questa è la versione minima: serve al blocco 1. La caratterizzazione completa (tutti i tipi di strato, ingressi e
uscite diversi, reti potate a caso) serve solo alla potatura, e si fa nel blocco 2 (fase 4a).

### Run

| run | cosa misura | dove | cosa si varia |
|---|---|---|---|
| B1-0 | ripetibilità: lo stesso modello (N, pesi casuali) misurato 5 volte | PI fp16 | niente: è sempre la stessa misura |
| B1-2 | pre-processing, post-processing e trasferimenti | PI | risoluzione di deploy: 640×640 (principale), 480p, 720p, 1080p (estensione) |
| B1-1 | costo minimo di uno strato: catene di strati identici e piccoli (conv 3×3, 16 canali, 54×30) | PI | numero di strati: 1, 2, 4, 8, 16, 32 |
| P1-1 | probe ridotto: 8 conv 3×3 uguali, canali da 1 a 64 a passo 1 e da 72 a 128 a passo 8, mappe del deploy a passo 8 e 16 del 480p e del 1080p (108×60, 54×30, 240×136, 120×68), 3 passate | PI fp16 | numero di canali |
| B1-3 | prestazioni massime reali: picco di calcolo e banda di memoria (roofline) | PI | precisioni del dispositivo |

**A cosa serve B1-0.** Misura lo stesso modello 5 volte su ogni dispositivo. La variazione fra le passate è la
**soglia di rumore** della latenza: sotto quella soglia una differenza fra due modelli non si distingue dal caso. Per
esempio, se le passate dello stesso modello variano dell'1%, una differenza del 3% fra due modelli è vera, una dello
0,5% no. La soglia si usa in tutte le fasi dopo: nei confronti di latenza della fase 2, nella scelta dell'ottimo (due
candidate entro la soglia sono equivalenti) e nei confronti della potatura. Si fa per prima, perché dice anche se
il dispositivo è pronto per misurare (frequenze, temperatura, processi in background).

Il probe ridotto sono 288 reti (72 canali × 4 mappe): \<esempio>circa 1 ora sul Pi\</esempio>.

### Cosa ci aspettiamo

\<esempio>

- Variazione fra le passate dello stesso modello sotto il 2% sul PI.
- **PI**: nessun gradino; latenza quasi proporzionale ai calcoli.

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
> (AGX e DX, se abbiamo tempo, usano la quantizzazione del loro strumento), e dice quale software usare per l'int8 (onnxruntime) e cosa aspettarsi. La conclusione
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
- **AGX e DX** (se abbiamo tempo) usano la quantizzazione del loro strumento: la ricetta standard di Ultralytics per TensorRT, la
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
> zero a 640, con una ricetta fissa, tutte le combinazioni di larghezza e profondità fino a 0.75; valutando ogni modello sul val a 640 e sui tre set di val di deploy; confermando con 3 seed i
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
(sezione 4): 9 larghezze × 3 profondità × 6 ingressi (3 risoluzioni di deploy e 3 ritagli quadrati del tracking) =
**162 configurazioni** per dispositivo, in fp16
e in int8. L'fp32 si misura solo sui modelli della mappatura, quanto basta per mostrare che non conviene. La griglia
comprende tutti i modelli della mappatura e tutte le candidate all'ottimo: è una **mappa della latenza misurata**,
senza bisogno di interpolare. \<esempio>Costo indicativo: sul Pi qualche ora per precisione\</esempio>. Per l'int8 i modelli si quantizzano con
**la ricetta del dispositivo**: sul PI quella di X2 (onnxruntime, testa, ramo di classe e attention in float) (AGX e DX: sezione 16).
Calibrazione su pesi casuali: per la latenza conta solo quali nodi sono quantizzati, non i valori. Già da qui si sa quali modelli stanno nel budget su ogni dispositivo, in ogni precisione e a ogni
risoluzione di deploy (figura A, sezione 13). \<esempio>Sul Pi, nell'estensione, probabilmente nessun modello oltre i 640: a 720p il fotogramma ha circa 2,3
volte i pixel di un 640×640, a 1080p circa 5 volte, e la latenza sul Pi segue i calcoli.\</esempio>

| run | modelli | dove | precisioni |
|---|---|---|---|
| B2-1 | griglia discreta di latenza | PI | fp16 (OpenVINO), int8 ricetta X2 (onnxruntime) (fp32 sui modelli della mappatura) |

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

Si allenano **tutte le combinazioni** di larghezza e profondità della scala discreta fino a 0.75: 8 larghezze × 3
profondità = **24 training** (griglia fattoriale completa). Ogni effetto si legge così su tutta la griglia, compresa
l'interazione fra le due leve, e il modello di precisione della 2c **interpola** fra punti misurati invece di
estrapolare: non servono training "fuori asse". Tutti i training sono a **640×640**: la risoluzione non è una leva di
training (regole fisse).

**Il limite superiore lo fissa l'hardware.** La M (1.0) non si allena: sul PI, il dispositivo più stretto, in fp16 a
480p (misura storica, non guida più le decisioni) la M D126 impiega 360 ms contro un budget di 28 (13 volte), e già la 0.75 ne impiega 246. Oltre la 0.75 non c'è
nessun candidato da cercare. La latenza della M si misura comunque nella griglia della 2a, perché è il dato che lo
dimostra. In più la precisione arriva a un plateau prima (dalla 0.375 il mAP50-95 sale di circa 1 punto in tutto).

| run | larghezze | profondità | note |
|---|---|---|---|
| T2-01…T2-24 | 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75 | D101, D126, D170 | numerati per larghezza crescente e, a pari larghezza, per profondità: N (0.25 D126) è **T2-14**, P è T2-08, S è T2-20 |

Script: `scripts/train_griglia_larghezza_profondita.sh` (fino a 0.25 due training alla volta, sopra uno alla volta);
figure: `tools/figure_griglia.py OUT_DIR` (mAP50 e mAP50-95 sul val a 640, a barre 3D e a linee).

**Ogni training si valuta quattro volte**: sul val a 640 (quello del training) e sui tre set di val di deploy
(`val_480p`, `val_720p`, `val_1080p`, fase 0, ancora da produrre). Sono solo validazioni, pochi minuti per modello.
Così ogni modello ha una mAP **misurata** per ogni risoluzione di deploy: `E2-01` per `T2-01` e così via.

**Limite dei mosaici.** I set di deploy sono mosaici: le giunture fra un frame e l'altro sono bordi artificiali
(attenuati dallo sfondo dello stesso video). Senza fotogrammi veri ad alta risoluzione non si possono confrontare
con un fotogramma vero: si dichiara come limite.

### 2c. Dai trend alla ricetta ottima per dispositivo

I trend da soli non bastano: dire "la larghezza pesa più della profondità" non sceglie un modello. Questo passo li
trasforma in una scelta, in quattro passi, **per ogni dispositivo e per ogni risoluzione di deploy**.

**Passo 1 — Il tasso di cambio di ogni leva.** Per ogni dispositivo, risoluzione di deploy e leva (larghezza,
profondità) si calcola quanta precisione si guadagna per ogni ms di latenza in più, fra punti vicini della griglia:
`Δ mAP50-95 / Δ ms`. \<esempio>Esempio con numeri inventati, su PI a 640:
larghezza +2 punti ogni 10 ms, profondità +0,5. Vuol dire che su PI i ms conviene spenderli in larghezza.\</esempio> Il tasso si calcola su tutta la
griglia, per vedere come cambia con la taglia. A parte si riporta **quanto costa salire di
risoluzione di deploy**: quanti ms in più, e quanta mAP in più o in meno, da 480p a 720p a 1080p.

**Passo 2 — Due modelli che predicono.**

- **Precisione**: per ogni risoluzione di deploy, `mAP(w, d)` stimata dai training della griglia valutati sul set di
  val di quella risoluzione, con una curva per leva (per esempio lineare nel logaritmo della larghezza) più un
  termine di interazione, misurato direttamente perché la griglia è completa. Dipende solo dal dataset, quindi è uguale per tutti i
  dispositivi. Si verifica togliendo un punto alla volta e predicendolo dagli altri (\<esempio>errore atteso sotto 1
  punto di mAP\</esempio>).
- **Latenza**: non serve predirla: è già **misurata** per ogni configurazione ammessa nella griglia discreta della 2a.

**Passo 3 — L'ottimo sotto il budget.** Per ogni dispositivo, precisione e risoluzione di deploy si prendono le
configurazioni della griglia discreta che stanno nel budget del modello (latenza misurata in 2a) e fra queste si
sceglie quella con la mAP predetta più alta (figura B, sezione 13). È sempre un modello che esiste. Sui dispositivi a
gradini (AGX, DX, se abbiamo tempo) le misure mostrano da sole quali larghezze sprecano un gradino \<esempio>(per esempio quelle
intermedie, con canali multipli di 3)\</esempio>.

**Criterio di scelta, valido in tutta la tesi** (ottimo, candidate, modello da potare): conta la **mAP50-95
complessiva** (media sulle 4 classi, oggetti ≥ 8 px, sul set di val della risoluzione di deploy). A pari merito,
cioè con una differenza **sotto 1 punto di mAP50-95** (soglia fissa, la stessa in tutta la tesi), vince il modello
più veloce.

**Passo 4 — Verifica.** Si allena l'ottimo predetto con 3 seed. Gli ottimi sono uno per dispositivo e risoluzione di
deploy (fino a 9), ma il training non dipende dalla risoluzione: **si allenano solo le architetture `(w, d)`
distinte** \<esempio>(probabilmente 3–5)\</esempio>. Se l'ottimo coincide con un punto della griglia, il seed 0 c'è già e servono
solo i seed 1 e 2. Quali ottimi confermare con 3 seed (tutti, o uno per dispositivo) è la scelta 8 (sezione 15).

| run | cosa | seed / ripetizioni |
|---|---|---|
| T2-O1 | ottimo predetto per PI | 0, 1, 2 |

Se più dispositivi o risoluzioni hanno lo stesso ottimo, i suoi training si fanno una volta sola.

> **Nota, da decidere (pending): quali candidate allenare oltre all'ottimo.** Proposta: si allena prima l'ottimo
> di ogni risoluzione del dispositivo (seed 0) e si misura l'errore assoluto fra mAP predetta e vera. L'intorno è il più grande
> fra questo errore e quello del test "togli un punto e predicilo" del passo 2 (un solo errore può uscire piccolo per
> caso). Poi si allenano (seed 0) le altre configurazioni nel budget con mAP predetta dentro l'intorno; la migliore
> misurata prende i seed 1 e 2. Le righe int8 non si allenano: sono la quantizzazione degli stessi modelli.
> \<esempio>Ottimo predetto 0,461, misurato 0,452: errore 0,9 punti; errore del passo 2: 0,6; intorno 0,9. Si allenano
> le candidate nel budget con mAP predetta ≥ 0,452.\</esempio>

**Cosa ci aspettiamo**: \<esempio>la mAP vera dell'ottimo entro circa 1 punto da quella predetta; un ottimo che cambia
con la risoluzione di deploy (su PI a 640 il budget è stretto; nell'estensione, a risoluzioni più alte, forse nessun modello); l'ottimo migliore di qualunque taglia standard (F, P, N, S, M) che sta nel budget.\</esempio>

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
predetta fra i due (criterio di scelta della 2c: mAP50-95 complessiva, a pari merito il modello più veloce):

| caso | cosa vuol dire | cosa si fa |
|---|---|---|
| **stretto** | il tetto non sta nel budget, e l'ottimo nel budget perde più di 1 punto di mAP50-95 | fase 4: si pota partendo dal tetto (o dal modello più piccolo che lo eguaglia), se passa il test del pavimento (sotto) |
| **libero** | il tetto sta nel budget, oppure la differenza è sotto 1 punto | l'ottimo è il modello finale; il dispositivo diventa un risultato descrittivo |

**Test del pavimento.** Un modello è candidato alla potatura solo se il suo **pavimento** sta sotto l'**85% del
budget `B`**. Il pavimento è la latenza con gli strati potabili al minimo (2 canali). Fanno eccezione 7 conv che non
scendono: sono le uscite congelate che alimentano C2PSA, PSABlock e Detect (`layers.8.cv2`, `9.cv2`, `16.cv2`,
`19.cv2`, `22.cv1b`, `22.m.0.0.cv2`, `22.cv2`), e restano a larghezza piena. Il pavimento è quindi la parte fissa C
(C2PSA, PSABlock, Detect, nodi non conv) più il costo di quelle 7 conv e il costo minimo delle altre. Vicino al
pavimento la rete perde troppa accuratezza; un modello il cui pavimento supera `B` non può proprio arrivarci. `B` e
la soglia dell'85% sono parametri per dispositivo. \<esempio>Sul PI `B` = 28 ms, quindi il pavimento deve stare sotto
23,8 ms (soglia scelta perché il nano deve rientrare).\</esempio>

- Per le architetture coperte dallo stimatore (F…N+, larghezze 0,0625–0,25, D101/D126/D170) il pavimento si calcola
  con LUT + fit, con i 7 strati a larghezza piena.
- Per quelle fuori (NS, S, SM, M) si misura sul Pi **una sola rete** al minimo. S non può arrivare al budget
  nemmeno potata al massimo, quindi l'estensione della LUT a S/SM non serve per il PI.
- Se il tetto non passa il test, si parte dal modello più preciso fra quelli che passano. A parità (differenza sotto
  il rumore tra seed) si prende quello con il pavimento più basso.
- **Scelta manuale.** Il modello di partenza della regola si può forzare con un parametro per dispositivo; la
  scelta forzata va scritta con il motivo, e il test del pavimento resta come avviso.
- **Implementazione.** `python run.py stage=select target=rpi5_640` (regola in
  `src/selection.py`, risultato in `results/select/<profilo>.json`). Budget, soglia, pari merito, scelta manuale e LUT
  stanno nel profilo `conf/target/rpi5_640.yaml`.

Latenze a 640×640 sul PI (ms). Il pavimento è la stima con lo schema esatto della rete al minimo
(`tools/figure_guadagno_latenza.py`; figura della tesi, cap. 8 "Decisione per dispositivo", `fig:ch8-guadagno-latenza`),
salvo dove indicato "misurato" (`experiments/pruning/pavimento_s`). Con soglia 85% e `B` = 28 ms (esempio) il limite è
23,8 ms: passano tutti fino a N compreso; non passano N+ e S.

| arch | pavimento | arch | pavimento | arch | pavimento |
|---|---|---|---|---|---|
| F- | 12,1 | P- | 14,7 | N- | 21,3 |
| F | 13,2 | P | 15,7 | N | 22,1 (misurato 22,3) |
| F+ | 16,3 | P+ | 19,0 | N+ | 26,2 (non passa) |
| FP- | 13,3 | PN- | 17,6 | S | 41,0 misurato (non passa) |
| FP | 14,4 | PN | 18,5 | | |
| FP+ | 17,6 | PN+ | 21,9 | | |

Latenze intere misurate a 640 (ms): P 28,0; PN 44,6; N- 53,2; N 56,1; N+ 70,5; S 142,7.

Osservazione: P intero fa già 28,0 ms, cioè sta esattamente nel budget a 640: è un candidato come baseline da
battere.

Si sceglie sempre il modello con la mAP migliore fra quelli che rispettano la latenza: la distanza di rilevamento non
entra nelle scelte, si considera solo alla fine. \<esempio>Previsione: **PI stretto** (a 640 il pico sta nel budget da 28 ms, il nano
no).\</esempio>

### 2e. Conferma con 3 seed

Le baseline sono già allenate con 3 seed (T2-O, in 2c). Resta da allenare il **modello da cui parte la potatura**,
solo sui dispositivi "stretti". Se è un punto della griglia (per esempio N, cioè T2-14), il seed 0 è
il suo training di mappatura e servono solo i seed 1 e 2.

| run | modello | seed |
|---|---|---|
| T2-C1…C3 | modello da potare su PI \<esempio>(previsto: N)\</esempio> | 0, 1, 2 |

Se due dispositivi condividono lo stesso modello, i suoi training si fanno una volta sola.

Questi training servono solo al blocco 2: si lanciano **quando la GPU è libera** durante il blocco 1, senza
farli diventare un collo di bottiglia. Se il blocco 2 non si fa, restano punti del fronte confermati con 3 seed.

### Cosa ci aspettiamo

\<esempio>

- Oltre una certa taglia la precisione smette di salire: con quattro classi potrebbe valere S ≈ 0.75.
- La mAP ai set di deploy cambia poco con la risoluzione (regole fisse); la differenza fra modelli la fanno le
  detection multiple sullo stesso oggetto, più frequenti nei modelli piccoli.
- Sotto il pico la precisione cala in fretta (canali interni minuscoli).
- **PI**: nel budget stanno solo i modelli piccoli (circa fino al pico) a 640; alle risoluzioni più alte dell'estensione, probabilmente nessuno.

\</esempio>

---

## 10. Fase 3 — Int8 (circa 1 settimana)

> **Cosa testiamo**: quanta precisione si perde passando da fp16 a int8, modello per modello. La velocità
> guadagnata è già misurata in fase 2a.
>
> **Come**: quantizzando dopo il training i modelli del fronte con la ricetta int8 del dispositivo (la stessa della
> latenza in 2a), e misurando la mAP sul val **con il runtime vero sul dispositivo**
> (OpenVINO/onnxruntime sul PI; TensorRT sull'AGX e DX-RT sulla DX, se abbiamo tempo); con il quantization-aware training solo dove la
> perdita supera 1 punto.
>
> **Conoscenza che produciamo**: per ogni modello e dispositivo, quanti punti di mAP costa l'int8 e quanti ms fa
> risparmiare; quale variante di quantizzazione funziona.
>
> **Come la usiamo**: si **ricalcola l'ottimo della fase 2c con le latenze int8** (il budget "compra" di più) e si
> ricontrolla la decisione 2d (un dispositivo "stretto" può diventare "libero"); si sceglie in quale precisione fare
> la potatura su ogni dispositivo e con quale variante quantizzare i potati.

**Domanda**: usando numeri a 8 bit invece che a 16, quanto si guadagna in velocità e quanto si perde in precisione?

Perché serve: sul PI l'int8 può spostare il fronte (sulla DX-M1 sarebbe **obbligatorio**, se abbiamo tempo), facendo entrare nel budget
modelli che in fp16 non ci stavano.

### Run

Modelli: gli ottimi T2-O (seed 0), i modelli da potare (2e) e i punti della griglia, per ricalcolare l'ottimo in int8.

| run | variante | cosa si misura |
|---|---|---|
| Q3-1 | PI: ricetta di X2 (onnxruntime, testa, ramo di classe e attention in float), 500 immagini di calibrazione dal train | mAP sul val a 640 e sui set di deploy, sul dispositivo (la latenza viene dalla 2a) |
| Q3-4 | quantization-aware training (30–50 epoche partendo dal modello già allenato) | solo per i modelli che con Q3-1 perdono più di 1 punto di mAP50-95 |

### Cosa ci aspettiamo

\<esempio>

- 0,5–2 punti di mAP persi con la quantizzazione dopo il training, recuperati dal quantization-aware training.
- **PI**: latenza simile all'fp16 e mAP più bassa (prova preliminare di X2), quindi int8 scartato; è la conferma sui
  modelli nuovi che la tesi riporta.

\</esempio>

### Cosa ne esce

Il fronte in int8. Si **ricalcola l'ottimo 2c con le latenze int8** e si ricontrolla la decisione 2d: un modello che
in fp16 non stava nel budget potrebbe starci in int8, e un dispositivo "stretto" potrebbe diventare "libero". Se
l'ottimo int8 cambia, lo si verifica con un training come in 2c.

---

## 11. Blocco 2 — Fase 4: potatura con lo stimatore di latenza (circa 4 settimane, se c'è tempo)

Si fa **solo dopo aver chiuso il blocco 1**, e **solo sui dispositivi "stretti"** della decisione 2d.
Sugli altri non c'è niente da ottimizzare. Ha due parti: prima lo stimatore (4a), poi la potatura (4b).

### 4a. Stimatore di latenza (chiusa il 09/11/2026)

> **Cosa testiamo**: se la latenza di una rete potata per canale si predice sommando le latenze dei suoi strati,
> misurate una volta per tutte in una tabella (LUT), più pochi parametri di composizione.
>
> **Come**: sul PI si misurano da sole tutte le conv potabili del nano, a passo 2 di canali (la LUT); poi si misurano
> intere circa 750 reti potate a caso, per stimare i parametri e verificare l'errore.
>
> **Conoscenza che produciamo**: lo **stimatore di latenza** del PI, con il suo errore; la LUT stessa, che mostra
> dove i canali costano (salti, cali, plateau); quanta parte della latenza non dipende dai canali (`C`).
>
> **Come la usiamo**: è il costo che guida la potatura in 4b, al posto del conteggio dei calcoli.

#### Perimetro

- **Un solo dispositivo come dimostrazione: il PI.** AGX e DX si fanno con lo stesso metodo solo se abbiamo tempo (sezione 16) e se
  la decisione 2d li dà stretti.
- **Una sola configurazione, fissata a priori**: OpenVINO fp16 (`infer_precision` f16), 4 core, governor
  `performance`, 2,4 GHz, ventola al massimo. Con un altro backend, un'altra precisione, altri core o un'altra
  frequenza i parametri vanno rimisurati. Il comando è quello del deploy (`conf/hardware/rpi5.yaml`,
  `conf/backend/openvino.yaml`, `conf/deploy/640.yaml`):

  ```
  benchmark_app -d CPU -b 1 -hint none -nstreams 1 -nthreads 4 -niter 200 -infer_precision f16 -report_type no_counters
  ```

  con il tuning della board di `src/measure/board.py` e la metrica `median_ms`.
- **Ingresso sempre 640×640, 4 classi.** Le risoluzioni di deploy diverse da 640 (480p, 720p, 1080p) restano fuori: 640×640 è la risoluzione di deploy principale.
- **Modello da potare: N** (0.25, D126). Per la verifica si usano anche le architetture più piccole (B4-R).

#### Cosa pota PIT (yolopit)

Verificato sul codice di yolopit (`~/Desktop/yolopit`, HEAD `fd2f3a3`):

- La potatura è **solo per canale**: toglie filtri di una conv. **Uno strato non viene mai eliminato**: sopravvive
  anche con un solo filtro, quindi la rete potata ha sempre gli strati del modello di partenza.
- **Esclusi** C2PSA, PSABlock e Detect (foglie fx, `LEAVES` in `yolopit/search.py:63`): il loro costo è fisso.
- Gli strati che alimentano una foglia hanno i **canali di uscita congelati** (`masks.py:256-320`); residui, add e
  concat hanno **maschere condivise** fra i rami (`masks.py:171`); i C3k2 sono potati all'interno (`C3k2Split`).
- Le depthwise di YOLO26 stanno tutte dentro Detect e non si potano (\<esempio>da verificare\</esempio>).
- Le strategie attuali (`pit_standard.yaml`, `pit_duccio.yaml`) potano **a blocchi di 16** (`pit.n: 16`). Lo
  stimatore toglie questo vincolo: è lui a dire quanto costa un numero di canali non allineato. Con la LUT
  `pit.n` diventa 2.

Ne seguono due correzioni al resto del piano. La sola leva della potatura è la larghezza: "togliere canali o togliere
blocchi" (2c, 4b) non si applica. Il costo degli strati (`α × numero di strati`) è uguale per tutte le reti potate
dallo stesso modello, quindi si fonde nella costante `C` e non guida la potatura.

#### Perché una LUT e non una formula

La formula prevista all'inizio era

```
latenza = c0 + (numero di strati × α) + Σ (calcoli di ogni strato, a canali arrotondati a g) × β del suo tipo
```

e, tolto `α` e l'arrotondamento forzato, una versione con un termine di allineamento:
`C + Σ_strati [β_tipo · MAC + δ · MAC_pad]`, con `MAC_pad` i calcoli in più arrotondando `c_in` e `c_out` a `g`.
Widget della formula: https://claude.ai/artifact/394Eb6UYTtSqU7UPAqa7G2 (valori di AGX e DX inventati).

Non basta, perché le curve canali → latenza del PI non sono né lineari né a gradini semplici:

- **probe `probe_edge`** (26 probe, conv 3×3 su 20×20, `c_in = c_out` da 1 a 96): salto del +30% a 33 canali, cali a
  48 e 64. Questi probe usano 30 iterazioni, con cui la mediana viene circa 2× il regime: sono poco affidabili, e da
  qui in poi si usano 200 iterazioni;
- **CPU della workstation**: andamento lineare con plateau (64, 68, 72 e 76 canali costano quasi uguale);
- **separabilità rifiutata** (05/11, [`pruning/separability`](experiments/pruning/separability), wandb `6lhj71tw`).
  L'ipotesi era `LUT_f(c_in, c_out) ≈ F_f(c_out) · G_f(c_in)`: se avesse retto, una LUT si sarebbe ricavata da due
  scansioni 1D (circa 2,5 ore). Esito su 5 928 misure, criterio "90% dei punti entro 2× il rumore":

  | | 1×1 s1 40×40 | 3×3 s1 20×20 |
  |---|---|---|
  | rumore (mediana della dispersione) | 3,4% | 2,2% |
  | predizione dalle scansioni 1D: errore mediano / massimo | 3,8% / 64% | 14% / 99% |
  | punti entro 2× il rumore | 71% | 34% |

  Dove fallisce: nella 3×3 un cambio di kernel a `c_in` = 64 (con `c_out` = 128: da 0,39 a 0,22 ms); gradini su
  `c_out` a 32, 64 e 96 con altezza che dipende da `c_in`; nella 1×1, con `c_out` = 128, alcuni `c_in` vanno 2× più
  veloci dei vicini, senza legame con i multipli di 8.

**Decisione: LUT 2D misurata direttamente, una per famiglia di strati, a passo 2.**

#### Strati potabili del nano e griglia

Tolti C2PSA, PSABlock e Detect, il nano a 640×640 ha **66 conv potabili, cioè 74 strati da misurare** in 13 famiglie (tipo, stride, mappa
d'uscita; per le s2 l'ingresso è il doppio). I 74 sono 66 più 8: nei C3k2 il `cv1` è diviso in due 1×1 (`cv1a`, `cv1b`), come in yolopit (`C3k2Split`). `c_in` e `c_out` massimi del nano sono i limiti della LUT.

| famiglia | strati | coppie (c_in, c_out) massime | MAC del nano | punti a passo 2 |
|---|---|---|---|---|
| 1×1 s1 20×20 | 11 | (128,64) (128,128) (256,128) (256,256) (384,256) (512,256) | 10% | 33 153 |
| 1×1 s1 40×40 | 18 | (64,32) (64,64) (128,128) (192,128) (384,128) | 13% | 12 545 |
| 1×1 s1 80×80 | 9 | (32,16) (32,32) (64,64) (96,64) (96,128) (256,64) | 12% | 5 825 |
| 1×1 s1 160×160 | 3 | (32,32) (48,64) | 5% | 825 |
| 3×3 s1 20×20 | 6 | (64,64) (64,128) (128,64) | 5% | 3 201 |
| 3×3 s1 40×40 | 12 | (32,32) | 8% | 289 |
| 3×3 s1 80×80 | 6 | (16,16) (16,32) (32,16) | 5% | 225 |
| 3×3 s1 160×160 | 2 | (8,16) (16,8) | 3% | 65 |
| 3×3 s2 20×20 | 2 | (128,128) (128,256) | 8% | 8 385 |
| 3×3 s2 40×40 | 2 | (64,64) (128,128) | 13% | 4 225 |
| 3×3 s2 80×80 | 1 | (64,64) | 11% | 1 089 |
| 3×3 s2 160×160 | 1 | (16,32) | 5% | 153 |
| 3×3 s2 320×320 (stem) | 1 | (3,16), `c_in` fisso a 3 | 2% | 9 |
| **totale** | **74** | | | **69 989** |

- I punti del nano sono 69 989. Con le forme che B4-R aggiunge (D101, D170 e w0.09375) la LUT arriva a **71 328**:
  +544 in `k3s1_40` (32×64 e 64×32), +512 in `k1s1_80` (128×128), +264 in `k1s1_160` (64×64), +19 in `k3s1_160`
  (3 canali in ingresso).
- Nessuna famiglia supera il 13% dei MAC: escluderne una per risparmiare misure vorrebbe dire non predire una parte
  reale della rete.
- La griglia di una famiglia è l'unione, sugli strati, dei punti con `c_in, c_out ∈ {1, 2, 4, …, massimo}`.
- **Il passo coincide con `pit.n`**: ogni configurazione che PIT può scegliere ha un valore misurato, senza
  interpolazione. A passo 1 i punti sarebbero 273 872 (circa 15 giorni di macchina senza ventola), a passo 2 69 989
  (stima iniziale circa 92 ore con 3 ripetizioni). Si è scelto il passo più piccolo che il tempo permette.

#### Protocollo della LUT

Dettagli, prove e decisioni in [`pruning/lut_pi_passo2`](experiments/pruning/lut_pi_passo2).

- **Rete di misura**: una sola conv con pesi casuali, bias e SiLU, come le Conv di YOLO26 con la BN fusa. La SiLU
  **non è fusa** nella conv, né qui né nel nano vero (102 `Convolution` e 101 `Subgraph`): la LUT misura lo strato
  come gira nella rete.
- **I/O fp16 in NHWC**, salvata come IR: con l'ONNX NCHW OpenVINO aggiungeva due `Reorder` che nel nano non ci sono
  (su 160×160 costavano più della conv).
- **Niente reti identità**: il costo fisso di ogni chiamata (circa 0,013 ms) entra nella somma sempre per gli stessi
  strati, qualunque sia la potatura, e lo assorbe `C` (`DECISIONE_IDENTITA.md`).
- **200 iterazioni dopo 60 ms di warmup** (`bench_dump.py`): il warmup dura un tempo fisso (circa 20 ms), non un
  numero di iterazioni; senza, le reti sotto 0,1 ms uscivano gonfiate fino al +76%.
- **3 ripetizioni**, ordine di tutti i punti mescolato con seme 0 e rimescolato a ogni ripetizione. I punti con
  dispersione oltre il 5% si rimisurano altre 6 volte (mediana su 9): ci sono reti con due regimi decisi all'avvio,
  come `k1s1_160` 48×64 (2,8 oppure 3,1 ms), causa ancora aperta.
- **Guardia termica**: sopra 75 °C o con throttling si aspetta e si ripete. Con la ventola al massimo il Pi sta fra
  35 e 51 °C.

**Stato**: finita il 09/11, 71 328 punti (le 13 famiglie, con le forme di B4-R).

#### Dalla LUT alla latenza della rete

La LUT dà il costo di uno strato da solo. Dentro la rete cambiano cache, layout e fusioni, e ci sono nodi che non
sono conv. Modello:

```
latenza_rete = C + Σ_f a_f · Σ_{strati in f} LUT_f(c_in, c_out)  [+ γ · elementi in uscita da concat, add, upsample, maxpool]
```

- `C`: tutto quello che la LUT non vede: C2PSA, PSABlock e Detect, i nodi non conv, il costo fisso di ogni chiamata
  contato una volta per strato. È **per architettura**, tarato sulla rete intera non potata.
- `a_f`: correzione della famiglia `f` dentro la rete (per esempio 0,8 = lì quegli strati costano il 20% in meno che
  da soli). Assorbe anche i pesi che nella rete non stanno in cache (circa 5 MB in fp16 contro 2 MB di L3), perché
  quel costo cresce come i calcoli, con `c_in·c_out·k²`.
- `γ` (facoltativo): concat, somme e upsample costano in proporzione ai canali, quindi cambiano con la potatura.

**Ipotesi sulla composizione**: sul PI OpenVINO esegue i nodi uno alla volta e i 4 core lavorano dentro ogni nodo,
quindi i rami paralleli (C3k2, FPN) non si sovrappongono e sommare gli strati è corretto.

**Stima e verifica** ([`pruning/b4r_composizione`](experiments/pruning/b4r_composizione)):

1. **Reti B4-R**: potature casuali di **tutte le architetture fino al nano** (larghezze 0.0625–0.25 × D101, D126,
   D170: 15 architetture), 50 per architettura più quella intera, 765 reti. Potatura con i vincoli di yolopit
   (`build_pit_model(n=2)`), rapporto indipendente per maschera, da circa il 10% alla rete intera; pesi casuali,
   export come nel deploy (IR fp16, one-to-one). Misura intera sul PI, 50 iterazioni, 3 ripetizioni.
   Le forme che la LUT del nano non copre (D101, D170 e w0.09375, 1 339 punti) si aggiungono alla LUT prima
   della misura. Larghezze 0.375, S e oltre solo in futuro: la LUT dovrebbe coprire fino a 4× i punti.
2. Per la rete `j` e la famiglia `f`: `x_jf = Σ LUT_f(c_in, c_out)` sui suoi strati.
3. **Ridge verso 1** con `a ≥ 0`: `min ‖y − C − X·a‖² + λ‖a − 1‖²`, `λ` con cross-validation a 5 fold di architetture intere (non di reti: con reti casuali il fit impara a memoria le architetture). Forma chiusa,
   niente gradient descent. Le famiglie che variano sempre insieme (correlazione > 0,98) condividono un solo `a`.
4. **Verifica** su reti tenute fuori (test diviso per architettura), confrontata con due stimatori più semplici:
   somma della LUT + `C` (`a_f` = 1) e `C + β·MAC`.

Se gli `a_f` non bastano, l'alternativa più fedele è misurare la latenza degli strati dentro la rete con i
performance counter di OpenVINO sulle reti B4-R. Prima della misura di B4-R si fa comunque il profiling per nodo di
un nano intero e di un potato, per vedere quanto pesano concat, add, upsample e maxpool sul PI.

#### Integrazione con PIT

Implementata in yolopit (`yolopit/latency.py`, `search.py`, `config.py`), con 13 test che passano.

- `pit.n` = 2, uguale al passo della LUT: ogni scelta discreta di PIT cade su un punto misurato.
- Il costo `latency` è `C_arch + Σ a_f · LUT_f` sulle maschere morbide, con interpolazione bilineare fra i punti della
  griglia. Il valore è sempre quello esatto della LUT; si cambia solo la pendenza per il gradiente.
- Il target DUCCIO si dà in ms: `target: {latency: X}`.
- Quattro pendenze, scelte con `pit.latency.estimator`: `lut` (derivata della LUT), `smooth` (superficie liscia per
  famiglia, **default**), `secant` (retta fino a D canali più in basso), `best` (retta più ripida verso qualsiasi punto
  più in basso). Servono perché la latenza va a gradini (ogni 32 canali di uscita) e la ricerca sente solo la pendenza
  nel punto in cui sta.

**Da provare** (dettagli in `reports/stimatore-latenza.md`, sezione 9):

1. Confronto delle quattro pendenze: stessa ricerca, stesso target in ms, cambia solo `estimator`. Si guardano mAP,
   latenza misurata sul PI e distanza dalla stima.
2. Aggiustamento dopo la search: i layer appena sopra un gradino si spostano al gradino sotto (per esempio da 34 a
   32 canali) e si misura quanto cambia la mAP dopo il finetune. È la vecchia idea "search con la sola parte liscia e
   poi ricerca locale sulla LUT", ridotta a un esperimento; non è implementata come strada alternativa.
3. Stimatore contro calcoli, alla stessa latenza misurata: è S4-01 contro S4-02 in 4b.

**Run**:

| run | cosa | stato |
|---|---|---|
| P4-1 | separabilità della LUT: 2 famiglie, scansioni 1D e griglia 2D | fatto il 05/11: rifiutata |
| P4-2 | LUT 2D a passo 2 delle 13 famiglie del nano, poi l'estensione alle forme di B4-R | fatto il 09/11 (71 328 punti) |
| B4-R | 765 reti potate a caso di 15 architetture, misurate intere | fatto il 09/11 |
| L4-1 | stima di `C`, `a_f` e verifica sulle reti tenute fuori | fatto il 09/11 |
| L4-2 | (solo se l'errore supera il 10%) un piccolo MLP che corregge il residuo | non serve |

**Esito.** Su 750 reti potate di 15 architetture, ognuna stimata senza averla vista nel fit, l'errore al p99 è
**4,1%** (obiettivo 5%), contro 12,9% della somma della LUT e 31,4% dei MAC. Mediana 0,9%, p95 3,0%, massimo 5,2%,
bias +0,2%: la mediana è pari al rumore fra due misure della stessa rete (1%). I MAC sbagliano sempre nella stessa
direzione per architettura: sottostimano del 20–25% le reti larghe 0,25. Gli `a_f` stanno fra 0,76 (1×1 a 160×160) e
1,51 (1×1 a 80×80), con `λ` = 1. `γ` non è servito. Dettagli in `reports/stimatore-latenza.md`.

**Aggiornamento del 13/11 (reti PIT nel fit).** Sulle reti prodotte da PIT (canali molto bassi, anche 2) il fit di sole B4-R
sottostimava di 0,75 ms (−2,8%, p99 4,0%). Il fit ora usa anche 108 reti PIT del nano e di tre architetture vicine, misurate sul Pi
(`experiments/pruning/fit_pit/`), contate 10 volte per pesare quanto B4-R. Su 32 reti PIT di test (mai nel fit): bias −2,0% → −0,3%,
p99 3,1% → 1,5%; sulle 15 reti del livello 0: bias −2,8% → −0,5%, p99 4,0% → 1,6%; su B4-R tenute fuori per architettura il p99
passa da 4,1% a 3,5% (bias +0,2% → 0,0%). Il vecchio fit è `artifacts/lut/rpi5/fit_b4r.json`; i numeri sopra (4,1%) sono quelli di B4-R da sola.
Secondo giro (13/11, sera): 15 reti PIT di train e 5 di test per ognuna delle 15 architetture. Il default `fit.json` tiene gli `a_f`
e rifà `C` come media dei residui sulle 15 reti PIT di ogni architettura. Sulle 75 reti di test: bias −1,0% → −0,1%, p99 2,7% → 2,2%;
B4-R tenute fuori invariato (3,5%); sul livello 0 del nano peggiora poco (bias −0,5% → −0,9%, p99 1,6% → 2,0%). Il fit di prima è
`fit_pit.json`; un fit rifatto da zero con tutte le PIT è scartato (p99 di B4-R 4,0%).

**Limiti da dichiarare**: strati misurati da soli, non dentro la rete (lo correggono `C` e `a_f`); temperatura letta
dopo ogni misura, non durante; le prove preliminari del 05–06/11 sono senza ventola, la LUT con la ventola.

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
  seed in 2e. Deve passare il test del pavimento della 2d (pavimento sotto l'85% di `B`). \<esempio>Su PI è previsto il **nano**: per portarlo alla latenza del pico va tolto circa il 70% dei calcoli
  (stima dalla retta 16,6 ms + 15,2 ms/GMAC).\</esempio>
- **La potatura lavora sulla leva che rende di più (tasso di cambio, 2c)**, alla risoluzione di deploy a cui il
  dispositivo è stretto.
- Il costo che guida la potatura (PIT) è o il **numero di calcoli** (il metodo classico) o lo **stimatore** del
  dispositivo (4a); sul PI con `pit.n` = 2, il passo della LUT.
- Dopo la potatura il modello si riallena per recuperare precisione (finetune).

**Run: potatura e finetune (seed 0)**. Ogni search `S4-xx` è seguita dal suo finetune `FT4-xx`. `B` è il budget del
modello del dispositivo; `B'` è un budget più stretto (per esempio 20 fps), per vedere se la tendenza regge.

| run | dispositivo | si parte da | costo che guida la potatura | budget | nota |
|---|---|---|---|---|---|
| S4-01 | PI | N | calcoli | `B` | metodo classico |
| S4-02 | PI | N | stimatore di PI | `B` | |
| S4-03 | PI | N | stimatore di PI | `B'` | budget più stretto |

**Run: i confronti (3 seed)**, sulla variante di potatura migliore del PI (scelta sul val con il seed 0):

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

\</esempio>
- Se la potatura perde contro l'ottimo scalato, è comunque un risultato: "su questo dispositivo conviene scegliere
  bene il modello piccolo".
- La potatura in profondità (togliere blocchi interi) resta fuori dal piano: si valuta solo se lo stimatore mostra
  che il costo per strato domina (AGX, se abbiamo tempo).

---

## 12. Fase 5 — Esame finale (circa mezza settimana)

> **Cosa testiamo**: se precisione e latenza dei modelli scelti reggono su dati mai visti e a bassa potenza.
>
> **Come**: aprendo il test una volta sola sui
> modelli scelti con il val, a 640 e alle risoluzioni di deploy; rimisurando latenza ed energia nelle condizioni
> finali; rimisurando alla frequenza sostenibile sul Pi (a 15 W sull'AGX: se abbiamo tempo).
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
| B5-3 | robustezza: i modelli scelti rimisurati sul PI alla frequenza sostenibile senza throttling | 3 passate |

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

\<esempio>Il test qualche punto sotto il val (lo split per gruppi di sessioni è più severo), ma con lo stesso ordinamento fra i metodi.\</esempio>

---

## 13. Figure di sintesi

Tre figure riassumono il blocco 1 e l'ingresso al blocco 2. Sono tutte a **piccoli multipli**: un pannello per ogni
combinazione di precisione (righe: fp16, int8) e profondità (colonne: D101, D126, D170), e dentro ogni
pannello la griglia discreta larghezza × risoluzione di deploy (480p, 720p, 1080p; sezione 4). Se ne fa **una per dispositivo**; affiancate, mostrano
come il dispositivo cambia la risposta.

| figura | cosa mostra | quando si fa | a quale scelta serve |
|---|---|---|---|
| **A. Chi sta nel budget** | ogni configurazione come casella colorata: entro il budget, entro ±5% dal budget, oltre il budget | dopo la griglia di latenza (2a) | dove passano i confini del budget; quanto l'int8 e la profondità spostano il confine |
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
| aggiunte al dataset (uccelli e aerei grandi) non trovate o troppo lente da etichettare | fase 0 | **successo in parte**: aerei ed elicotteri trovati (DVIDS), uccelli in volo pochi (Open Images); uccelli e droni piccoli di val e test dichiarati come limite |
| SAM 2 non rende aderenti i box degli oggetti minuscoli (sagome vuote o sbagliate) | fase 0 | **successo**: con il box come prompt SAM 2 restituiva il box stesso; box rifatti con contrasto locale e SAM 2 con un punto, poi revisione in CVAT di tutti i frame |
| sul Pi non sta nessun modello a 720p e 1080p | 2a | sul Pi nessun modello oltre i 640: è un risultato (estensione; a 480p la misura è storica) |

---

## 15. Scelte da confermare: stato

| # | scelta | stato |
|---|---|---|
| 1 | **ricetta di training** | **decisa**: una ricetta unica per tutti i training, niente versione ridotta per la mappatura; 300 epoche con early stopping a patience 200 (tempo di GPU) |
| 2 | **software di inferenza per dispositivo** | **decisa** con X2: sul PI OpenVINO per l'fp16 e onnxruntime per l'int8, perché OpenVINO int8 su ARM è più lento (41,1 contro 29,4 ms sul pico); AGX TensorRT (int8 con la ricetta standard di Ultralytics) e DX DX-COM/DX-RT, se abbiamo tempo |
| 3 | **ottimo predetto al posto del riempimento** (2c) | **accettata** |
| 4 | **secondo budget più stretto** (`B'`, es. 20 fps) in fase 4 | aperta, si decide all'inizio del blocco 2 |
| 5 | **latenza misurata con pesi casuali**, subito (fasi 1 e 2) | **accettata**, insieme alla 3: la latenza delle candidate dell'ottimo si misura invece di predirla |
| 6 | **alta risoluzione su S solo se sul nano aiuta** | **superata**: la risoluzione non si allena più, è una condizione di deploy |
| 7 | **aggiunte al dataset come ritagli 640×640 a scala nativa** (fase 0) | **superata**: tutte le immagini sono 640×512, le fonti più grandi ritagliate attorno agli oggetti senza ridimensionare |
| 8 | **quali ottimi confermare con 3 seed**: tutti quelli distinti per dispositivo e risoluzione di deploy, o uno per dispositivo | **aperta**, si decide in 2c quando si sa quanti sono |
| 9 | **ottimizzatore, learning rate e augmentation** della ricetta unica | **decisi** MuSGD, `lr0` 0.01, `lrf` 0.01, batch secondo la GPU; augmentation di `yolo26n.pt` con cutmix e degrado della camera dal 28/10/2026 (regole fisse, training e seed) |
| 10 | **soglia di confidenza di deploy** | **decisa**: quella che massimizza l'F1 sul val, una per tutte le classi, per modello e risoluzione di deploy; applicata al test in fase 5 senza ricalcolarla (regole fisse, metriche) |
| 11 | **minimo di istanze e di video per cella** della matrice classe × fascia (fase 0) | **superata**: niente minimi per cella; una distribuzione obiettivo delle grandezze uguale per tutte le classi (s^−0,3 fra 8 e 320 px; val sulla curva, test uniforme in log s), con almeno 2.000 istanze per classe in val e 1.000 in test, mai oltre il 20% del materiale in dominio; le classi sotto l'obiettivo sono limiti dichiarati (`vedetta/README.md`, Regole e Limiti dichiarati) |

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
- **Lunghezza**: 300 epoche con patience 200, per il tempo di GPU. Su AOD4 il pico arrivava al 99% del suo massimo
  verso l'epoca 620 e il nano verso la 400–430; su VEDETTA (28/10/2026) pico, nano e small toccano il massimo
  fra l'epoca 187 e la 233.

### Da zero o da COCO

Tutto **da zero**. I pesi COCO esistono solo per n, s, m: le nostre varianti (F, P, profondità D101 o D170, gli ottimi
con larghezze intermedie) non li hanno. Mescolare modelli preaddestrati e non mescolerebbe l'effetto dell'architettura
con quello del preaddestramento, e i trend della fase 2 non sarebbero più leggibili. Preaddestrare tutto su COCO
costerebbe giorni di GPU per modello. \<esempio>Con un dataset grande e un training lungo, partire da zero si avvicina al
preaddestrato.\</esempio> Il guadagno del preaddestramento si può citare come limite, o misurare con un solo run (lo stesso
modello da zero e da COCO) se resta tempo.

---

## 16. Se abbiamo tempo: AGX e DX

Si lavora solo sul PI. Qui sta quello che riguarda gli altri due dispositivi, spostato dalle sezioni indicate. Si fa dopo
aver chiuso il PI, solo se resta tempo, e non cambia le conclusioni sul PI.

### Dispositivi (sezione 3)

| sigla | dispositivo | dove calcola | software di inferenza | precisioni | configurazione |
|---|---|---|---|---|---|
| **AGX** | Jetson AGX Orin | GPU Ampere | TensorRT | fp16, int8 | profilo MAXN, clock fissati |
| **DX** | Raspberry Pi 5 + DX-M1 | NPU, collegato via PCIe | DX-COM (compilatore) e DX-RT (runtime) | solo int8 | host come PI |

- **AGX**: \<esempio>per modelli piccoli la GPU è quasi vuota e la latenza dipende soprattutto dal **numero di
  strati** (ogni strato ha un costo minimo di qualche µs), e i canali "tondi" (multipli di 8 o 16) contano\</esempio>.
  È quello che succede sulla RTX 6000 della workstation, misurato: il probe dà N=16 e un gradino netto a 64 canali, e
  il pico potato canale per canale è più lento del pico intero. Sulla AGX va verificato.
- **DX**: un NPU elabora i canali a blocchi fissi; \<esempio>ci aspettiamo una latenza a gradini, con la dimensione dei blocchi
  da scoprire. Il trasferimento dell'immagine sul PCIe potrebbe pesare quanto l'inferenza per i modelli piccoli.\</esempio>

### Incognita X1 (sezione 6)

La prova resta nella sezione 6, marcata. Se la DX non gira, resta un risultato descrittivo; se la compilazione DX-COM è troppo lenta per i probe completi, si riducono i probe e le reti potate a caso sulla DX.

### Fase 1 (sezione 7)

B1-0, B1-1, B1-2, P1-1 e B1-3 si ripetono su **AGX** (fp16) e **DX** (int8). Il probe ridotto sono 288 reti per dispositivo:
\<esempio>4–6 ore di build sull'AGX\</esempio>, da verificare sulla DX (dipende dal compilatore, si scopre con X1).
Il controllo di H2D e D2H sull'AGX è in sezione 5.

Cosa ci aspettiamo:

\<esempio>

- **AGX**: gradini (attorno ai 64 canali), costo per strato di qualche µs che pesa molto sui modelli piccoli; N = 8–16
  (sulla RTX 6000 viene 16).
- **DX**: gradini alla granularità dell'NPU; trasferimento PCIe importante per i modelli piccoli.

\</esempio>

### Fase 2 (sezione 9)

**Griglia di latenza (2a).** Costo indicativo: \<esempio>sull'AGX la build di ogni engine richiede minuti, quindi circa un giorno di
macchina senza presidio\</esempio>; sulla DX dipende dal tempo di compilazione (X1). Ricetta int8: sull'AGX quella standard di Ultralytics
per TensorRT (`format=engine`, `int8=True`); sulla DX la conversione di DX-COM.

| run | modelli | dove | precisioni |
|---|---|---|---|
| B2-2 | griglia discreta di latenza | AGX | fp16, int8 Ultralytics (fp32 sui modelli della mappatura) |
| B2-3 | griglia discreta di latenza | DX | int8 DX-COM |

**Ottimo (2c) e modello da potare (2e).**

| run | cosa | seed |
|---|---|---|
| T2-O2 | ottimo predetto per AGX | 0, 1, 2 |
| T2-O3 | ottimo predetto per DX | 0, 1, 2 |
| T2-C4…C6 | modello da potare su AGX, se serve | 0, 1, 2 |
| T2-C7…C9 | modello da potare su DX, se serve | 0, 1, 2 |

Cosa ci aspettiamo: \<esempio>su AGX il tasso di cambio è diverso da quello del PI, perché lì la larghezza costa poco finché la GPU resta mezza
vuota; ottimi più larghi (su AGX anche a 1080p) e, su DX, canali allineati ai blocchi dell'NPU; AGX e DX probabilmente «liberi» nella
decisione 2d, con S nel budget almeno a 720p.\</esempio>

### Fase 3 (sezione 10)

| run | variante | cosa si misura |
|---|---|---|
| Q3-2 | AGX: ricetta standard di Ultralytics per TensorRT (`format=engine`, `int8=True`), calibrazione dal train | mAP sul val a 640 e sui set di deploy, sul dispositivo |
| Q3-3 | DX: conversione di DX-COM, con la sua calibrazione | come sopra |

\<esempio>AGX: guadagno modesto, perché per i modelli piccoli pesa il costo minimo per strato. DX: l'int8 è l'unico modo di farli girare.\</esempio>

### Fase 4 (sezione 11)

Come per il PI, solo se il dispositivo risulta «stretto» nella decisione 2d. Se lo stimatore sbaglia più del 10% sull'AGX: MLP sul residuo (L4-2); altrimenti sull'AGX si pota solo con i calcoli.

| run | dispositivo | si parte da | costo che guida la potatura | budget | nota |
|---|---|---|---|---|---|
| S4-04 | AGX | dalla decisione 2d | calcoli | `B` | solo se AGX è stretto |
| S4-05 | AGX | dalla decisione 2d | stimatore di AGX | come sopra | |
| S4-06 | DX | dalla decisione 2d | calcoli | `B` | solo se DX è stretto |
| S4-07 | DX | dalla decisione 2d | stimatore di DX | `B` | |

Cosa ci aspettiamo: \<esempio>AGX: togliere canali serve poco; lo stimatore evita i canali non allineati. DX: lo stimatore allinea i canali ai blocchi dell'NPU, con un guadagno netto rispetto ai calcoli.\</esempio>

- **Nota, se c'è tempo: allineamento dei canali su TensorRT.** Lo stimatore (4a) tiene già conto dei blocchi di
  canali (`g`). Come verifica diretta, sull'AGX si confronta un potato con i suoi canali così come escono dalla
  potatura e lo stesso potato con i canali arrotondati a multipli di `g`. Sul Pi l'arrotondamento non dava speedup;
  su GPU HALP (Shen et al., NeurIPS 2022) mostra latenza a gradini e guadagno con canali allineati: è il contrappunto
  da citare.

### Fase 5 (sezione 12)

| run | cosa | ripetizioni |
|---|---|---|
| B5-2 | energia per inferenza sull'AGX (sensori di bordo) | 3 passate |
| B5-3 (AGX) | i modelli scelti rimisurati sull'AGX a 15 W | 3 passate |

### Rischi (sezione 14)

| rischio | quando lo si scopre | piano B |
|---|---|---|
| operatori di YOLO26 non supportati dalla DX-M1 | X1 | sostituirli; altrimenti DX diventa descrittivo |
| AGX non disponibile | fase 1 | si lavora su PI e DX |
| lo stimatore sbaglia più del 10% sull'AGX | 4a (L4-1) | MLP sul residuo (L4-2); altrimenti sull'AGX si pota solo con i calcoli |
| compilazione DX-COM troppo lenta per i probe completi | X1 | probe e reti potate a caso ridotti sulla DX |
