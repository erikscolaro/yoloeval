# Ricerca dataset: oggetti in volo ad alta risoluzione

Ricerca web del 12/10/2026, fatta da un agente. I numeri vengono dalle fonti citate e **non sono stati ricontrollati**.

**Obiettivo**: dataset pubblici per allenare da zero YOLO26 su 4 classi (`airplane`, `bird`, `drone`, `helicopter`),
anche combinando dataset con una sola classe.

**Requisiti**: lato corto ≥ 960 px (quadrati, o ritagliabili a ≥ 960×960); dataset autorevoli; ID di video o scena
per uno split senza leakage; bounding box; RGB; licenza adatta a una tesi.

AOD4 (split casuale per immagine, leakage) e Svanström (640×512) sono esclusi in partenza.

## In sintesi

Nessun dataset pubblico RGB con le 4 classi insieme ha il lato corto ≥ 960 px e gli ID dei video. L'unico che copre
tutte le classi è **AOT** (Amazon), ma è in scala di grigi e i droni stanno nella classe "Other" con altri oggetti.

## 1. Dataset sopra la soglia

Citazioni approssimative: S2 = Semantic Scholar, OA = OpenAlex.

| Nome | Paper, sede, citazioni | Classi | Risoluzione e ritaglio ≥ 960 | Immagini / video / istanze | ID di video o scena | Label | Modalità e punto di vista | Licenza e accesso | Download | Problemi noti |
|---|---|---|---|---|---|---|---|---|---|---|
| **AOT** ([AWS](https://registry.opendata.aws/airborne-object-tracking/), [AIcrowd](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge)) | Nessun paper sul dataset, solo la challenge AIcrowd; citazioni n/d | Aereo, elicottero, uccello, Other (mongolfiere, ultraleggeri, droni) | 2448×2048 fissa; ritaglio 2048×2048 | 5,9M frame, 4.943 sequenze da 120 s; label: aereo 0,92M, elicottero 1,39M, uccello 0,39M, Other 0,69M; 3,3M frame vuoti | Sì: flight_id; split per giorno | JSON/CSV, box [top, left, w, h], con distanza | **Scala di grigi**, aria-aria | CDLA-Permissive 1.0, accesso libero da S3 | 13,4 TB (3–5 GB per volo) | Oggetti in media pari allo 0,01% dell'immagine; i droni non hanno una classe propria (nome nel groundtruth non trovato); per il registry AWS "tutti gli oggetti in volo sono etichettati" |
| **TartanAviation** ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11926361/), [GitHub](https://github.com/castacks/TartanAviation)) | Scientific Data 2025; ~8 (OA) | Aerei (monomotore 56%, plurimotore), rotorcraft | 2448×2048; ritaglio 2048×2048 | 3,1M frame, 550 sequenze, oltre 700k label; quota rotorcraft non indicata | Sì: sequenza = camera_id + timestamp | Un .label per frame con box, track ID, distanza, modello dell'aereo | RGB (dalle figure), da terra verso il cielo, 4 camere fisse in un solo aeroporto (KAGC) | Licenza del dataset non indicata (articolo CC BY 4.0), download libero | 3,3 TB compressi | Uccelli e droni non menzionati, quindi non etichettati; un solo sito; timestamp sovraimpresso; box generati da AirTrack e ADS-B, poi verificati a mano frame per frame |
| **Drone-vs-Bird** ([review OJSP](https://publica.fraunhofer.de/entities/publication/4fc46818-3c7c-4b90-a9b1-124b1c5e457e)) | Sensors 2021 ~141 (S2); OJSP 2024 ~30 (OA); challenge WOSDETC (AVSS, ICASSP, IJCNN) | Solo drone | Mista, da 720×576 a 3840×2160; quota sopra soglia **non trovata** | 77 video, circa 1.384 frame per video ([arXiv 2504.19347](https://arxiv.org/html/2504.19347)) | Sì: nome del video | Un file per video: frame + box | RGB, da terra verso il cielo, camere fisse e mobili | Su richiesta, solo uso non commerciale | Non trovato | **Uccelli presenti ma non etichettati**; quasi tutti i droni sotto 32×32 px; desincronizzazioni nelle label di 2 video |
| **Anti-UAV300**, parte RGB ([GitHub](https://github.com/ZhaoJ9014/Anti-UAV)) | IEEE TMM 2021 ~180 (OA); challenge a CVPR/ICCV | Solo drone | 1920×1080; ritaglio 1080×1080 | 318 coppie RGB-IR; split 160/67/91 video (fonte secondaria) | Sì | Box per frame + flag di presenza | RGB (più IR 640×512), da terra verso il cielo | Repository MIT; Google Drive o Baidu | Non trovato | Single-object tracking: viene etichettato solo il target; mediana dei box 116×61 px, quindi droni grandi; è la stessa fonte usata da AOD4 |
| **Det-Fly** ([GitHub](https://github.com/Jake-WU/Det-Fly)) | IEEE RA-L 2021, ~202 (S2) | Solo drone (un solo modello, DJI Mavic) | 3840×2160 | 13.271 immagini, un drone per immagine | **No** (in parte da video a 5 FPS, ID non documentati) | Formato non indicato | RGB, aria-aria; molti sfondi a terra (vista dall'alto 32,5%) | MIT sul repository; OneDrive/Baidu | Non trovato | Rischio di leakage; circa metà dei target sotto il 5% dell'immagine |
| **ARD-MAV** ([arXiv](https://arxiv.org/html/2312.11008), [GLAD](https://github.com/WestlakeIntelligentRobotics/GLAD)) | IEEE T-ITS 2024, ~29 (OA) | Solo drone | 1920×1080; ritaglio 1080 | 60 video, 106.665 frame, un MAV per video | Sì: split 45/15 video | labelImg, formato non indicato | RGB, aria-aria | Licenza non indicata | Non trovato | Box da 6×3 a 136×75 px, media pari allo 0,02% dell'area |
| **SMOT4SB** ([arXiv](https://arxiv.org/html/2507.12832)) | MVA 2025 challenge, ~4–7; predecessore SOD4SB (MVA 2023) ~20 | Solo uccello | 1920×1080 e 3840×2160; quota di ciascuna non trovata | 211 video, 108.192 frame, 371.690 istanze | Sì: split 128/38/45 video | Box + ID di traccia | RGB, ripreso da drone (sfondi cielo, alberi, campi) | Licenza non indicata; Google Drive | Non trovato | Quasi tutte le istanze sotto 32×32 px; le label del test pubblico non sono rilasciate |
| **AirBirds** ([sito](https://airbirdsdata.github.io/), [arXiv](https://ar5iv.labs.arxiv.org/html/2304.11662)) | ACCV 2022, ~16 (OA) | Solo uccello | 1920×1080; ritaglio 1080 | 118.312 frame, 409.967 box | Parziale: sequenze in ordine temporale, split per sequenza a chunk (circa 98k train / 20k test) | Stile YOLO | RGB, camere a terra lungo la pista di un aeroporto | CC BY-NC-SA 4.0, si scarica dopo un modulo | Circa 165 GB | **88% delle istanze sotto 10 px**; aerei visibili ma tolti dalle label; label del test non rilasciate |
| **LRDDv3** ([arXiv](https://arxiv.org/html/2605.25942v1)) | arXiv, maggio 2026 | Drone 93.652, uccello 6.031, aereo 722 | 4K (IR 640×512) | 102.532 immagini, 128 clip, 17 giorni | Clip sì, ma formato degli ID non indicato | Box (prelabel YOLO + SAHI, poi correzione manuale) | RGB + IR, aria-aria | CDLA v2; **accesso sospeso** ([pagina](https://research.coe.drexel.edu/ece/imaple/lrddv3/)) | Non trovato | Non ancora scaricabile; gli aerei sono pochissimi |

## 2. Esclusi

- **FBD-SV-2024**: 1280×720 ([arXiv](https://arxiv.org/html/2409.00317v1)).
- **DUT Anti-UAV**: la parte di detection ha risoluzioni da 160×240 a 5616×3744 (test al massimo 1920×1080) e nessun
  ID di scena; la parte di tracking ha solo 20 sequenze, a 1920×1080 o 1280×720.
- **MAV-VID**: risoluzione non documentata, fonti miste, la pagina Kaggle non funziona.
- **NPS-Drones**: video a 1920×1080 mescolati a 1280×760 (quota non trovata); droni da 10×8 a 65×21 px.
- **FL-Drones**: 640×480 e 752×480 in scala di grigi, accesso su permesso.
- **Anti-UAV410 e Anti-UAV600**: solo IR.
- **SOD4SB**: è 4K, ma i frame sono mescolati senza ID di video. Usare SMOT4SB al suo posto.
- **LRDDv1 e LRDDv2**: 1920×1080 secondo la tabella di LRDDv3; accesso sospeso; ID dei video non trovati.
- **UAVDB**: solo arXiv, box generati automaticamente (PIC), un solo drone.
- **BirDrone/YOLOBirDrone e AODTA**: arXiv 2026, risoluzione e ID non indicati, assemblati da altre fonti.
- **COCO e LVIS** (al massimo 640 px) e **Open Images** (lato lungo al massimo 1024 nella distribuzione FiftyOne;
  bucket CVDF non verificato): sotto soglia e quasi sempre oggetti non in volo. **Objects365**: risoluzione non
  verificata.

## 3. Le 3 combinazioni migliori

**A. Tutto RGB, da terra verso il cielo**: TartanAviation (aerei, elicotteri) + Drone-vs-Bird e Anti-UAV RGB
(droni) + AirBirds (uccelli).

- Pro: punto di vista omogeneo, RGB, sopra soglia, split per video o sequenza in tutti e quattro, sedi solide.
- Contro: in TartanAviation e in Drone-vs-Bird ci sono uccelli non etichettati, in AirBirds aerei non etichettati.
  Servono pseudo-label incrociate (un modello allenato sugli altri dataset etichetta le classi mancanti) oppure il
  filtro dei frame. Gli uccelli di AirBirds sono minuscoli.
- Confusione classe-dataset: **alta** per uccelli e droni, perché ognuna di queste classi viene da un solo dataset.
  Lo sfondo d'aeroporto è comune a TartanAviation e AirBirds, ma ognuno contiene una sola delle due classi, quindi il
  rischio resta.

**B. Tutto in scala di grigi, aria-aria**: AOT (aerei, elicotteri, uccelli, più i droni dentro "Other") + Det-Fly e
ARD-MAV convertiti in scala di grigi (droni).

- Pro: tre classi su quattro nello stesso dataset, con le stesse camere; secondo il registry AWS sono etichettati
  tutti gli oggetti; licenza permissiva; enorme; split per giorno.
- Contro: niente colore. I droni di AOT vanno separati da "Other" (da verificare nel groundtruth.json). Il download
  completo pesa 13 TB, ma si può scaricare per volo. Det-Fly non ha ID di sequenza.
- Confusione classe-dataset: **media**. Solo i droni vengono da altre camere, e a distanze diverse (10–100 m in
  Det-Fly, chilometri in AOT).

**C. Varietà RGB con classi miste**: LRDDv3 (droni, uccelli, aerei, appena sarà disponibile) + SMOT4SB (uccelli) +
TartanAviation (elicotteri e aerei).

- Pro: in LRDDv3 tre classi convivono nello stesso dataset; tutto è RGB ad alta risoluzione; gli split sono per
  video.
- Contro: LRDDv3 oggi non si scarica e ha solo 722 aerei. Il punto di vista è misto: aria-aria per LRDDv3 e
  SMOT4SB, da terra verso il cielo per TartanAviation.
- Confusione classe-dataset: **alta per gli elicotteri** (vengono solo da TartanAviation), **media** per le altre
  classi.

## 4. Raccomandazione

Se basta l'RGB, la combinazione **A**, con il nucleo TartanAviation + Drone-vs-Bird + Anti-UAV RGB e pseudo-label
incrociate per le classi non etichettate. Per misurare quanto il modello impara lo stile del dataset: validazione per
dataset e su video tenuti fuori.

**AOT** è l'unico dataset con le 4 classi in un solo set: conviene almeno come test esterno. Se la scala di grigi è
accettabile, la combinazione **B** è quella con meno confusione fra classe e dataset; sul dispositivo basta
convertire il fotogramma in grigio prima del modello.

LRDDv3: da tenere d'occhio, l'accesso è sospeso.

## Verifiche da fare prima di scegliere

1. **AOT, droni**: come sono nominati dentro "Other" nel groundtruth, e quanti sono.
2. **AOT, oggetti piccoli**: in media lo 0,01% dell'immagine, circa 20×20 px a 2448×2048, quindi circa 8 px a 960.
   Contare quanti restano sopra 16 px dopo lo scaling.
3. **Drone-vs-Bird**: quanti video hanno il lato corto ≥ 960 px.

## Decisione (13/10/2026)

Scenario: **torretta anti-drone a terra**, camera rivolta verso il cielo, **a colori** (la scala di grigi non basta).
Quindi si parte dalla combinazione **A** (TartanAviation + Drone-vs-Bird + Anti-UAV RGB + AirBirds). AOT è escluso
anche come test esterno: è in scala di grigi e ripreso aria-aria.

## Verifica di TartanAviation sul paper (13/10/2026)

Dal paper (Scientific Data 2025), sezione sui dati visivi:

- **Un solo punto di ripresa**: un unico supporto con 4 camere fisse all'aeroporto KAGC (Pittsburgh). Tutti i
  3,1M frame hanno quindi gli stessi 4 sfondi.
- **Varietà solo di condizioni**: da dicembre 2021 a febbraio 2023, dall'alba al tramonto; 56% cielo sereno, 12%
  coperto, 6% precipitazioni.
- **Oggetti lontani**: la registrazione parte quando un aereo con ADS-B entra in un raggio di 8 km. Il dataset è
  pensato per la rilevazione a lunga distanza, quindi i box sono piccoli.
- **Tipi di velivolo**: monomotori 56%; plurimotori e rotorcraft insieme 44%, quota degli elicotteri non data.
- **Label**: AirTrack (input in scala di grigi) + proiezione ADS-B, poi verifica manuale frame per frame.
- **Colore**: il paper non lo dice (camera Sony IMX264, che esiste sia a colori sia monocromatica).

Conclusione: varietà di scena molto ridotta. Usabile al massimo come una fonte fra le altre, con un tetto di frame
per sequenza, non come unica fonte di aerei ed elicotteri.

## Droni e uccelli etichettati insieme (13/10/2026)

Requisito aggiunto: gli uccelli devono essere etichettati. Ricerca di dataset con entrambe le classi annotate:

- **Drone-vs-Bird**: accesso via email a wosdetc@googlegroups.com, previa firma di un accordo d'uso, solo ricerca
  ([GitHub](https://github.com/wosdetc/challenge)). Label in txt, una riga per frame (`framenum n x y w h classe ...`).
  Gli uccelli compaiono ma **non sono etichettati** (review ICASSP 2023). Le label dei droni corrette (frame
  desincronizzati) sono su raysonlaroca.github.io.
- **LAT-BirdDrone** (Data in Brief 2025, [IEEE DataPort](https://ieee-dataport.org/documents/lat-birddrone)):
  1920×1080 da torretta a terra, ma i **droni sono solo in infrarosso** (15 video); uccelli 30 video visibili + 30
  IR. Label generate con YOLOv11n + ByteTrack. Accesso con abbonamento DataPort. Scartato.
- **VIP Cup 2025** (IEEE SPS): droni e uccelli etichettati, RGB + IR, ma **320×256**. Scartato.
- **YOLOBirDrone** (arXiv 2026): uccelli etichettati solo in 4 video (3.067 frame); risoluzione non indicata.

Conclusione: nessun dataset pubblico RGB, da terra, ≥ 960 px, con droni e uccelli entrambi etichettati. La strada più
solida: **Drone-vs-Bird con gli uccelli etichettati da noi** (pre-label automatico + revisione manuale). Gli uccelli
sono già nelle stesse scene dei droni, quindi le due classi condividono camere e sfondi e il rischio di confondere
classe e dataset sparisce.

## Uccelli ripresi da terra, scaricabili senza richieste (13/10/2026)

| dataset | risoluzione | punto di vista | accesso | problemi |
|---|---|---|---|---|
| AirBirds (ACCV 2022) | 1920×1080 | camere a terra in un aeroporto, 4 stagioni | modulo sul sito, CC BY-NC-SA 4.0 | 88% degli uccelli sotto 10 px |
| Skagen e Klim ([researchdata.se](https://researchdata.se/en/catalogue/dataset/2021-316-1), J. Imaging 2021) | non dichiarata | camere fisse vicino a parchi eolici, Danimarca | download diretto, CC BY 4.0, 3,7 GB | **camere monocromatiche**: scartato |
| Yoshihashi et al. (Wind Energy 2017; arXiv 2105.08253) | 3840×2160 (5616×3744 nella versione 2017, non verificato) | una camera fissa in un parco eolico | non trovato (video supplementari su bird.nae-lab.org/tfod4k) | un solo luogo; uccelli di circa 25 px |
| FBD-SV-2024 (Scientific Data 2025) | 1280×720 | videosorveglianza | — | sotto soglia |

## Seconda ricerca: aerei ed elicotteri da terra (sonnet-1, 13/10/2026)

Nessun dataset nuovo supera tutte le soglie.

- **LAA3D** (arXiv 2511.19057, nov. 2025): parte reale 1920×1280, 185 sequenze, 15.000 frame (più 600k sintetici);
  MAV, eVTOL, elicotteri (circa 6%); niente aerei di linea o da turismo; oggetti entro 100 m; licenza e download non
  trovati.
- Esclusi: QUT Aircraft Collision Course (1280×720, un solo aereo, niente box), sekilab FlyingObjectDataset (4K ma un
  solo punto di ripresa, circa 1.630 immagini, nessun ID), KI4Flight (ripreso da drone), MBAOD, Anti2, AVD, AODTA
  (assemblati senza ID), General Aviation Aircraft Utah (aerei a terra), Deep4Air e AVOIDDS (sintetici), Rozantsev
  2014, FGVC-Aircraft, Military Aircraft Detection, MADAI.
- Da verificare: GOT-10k e VastTrack (molti video e scene, ma single-object tracking: un solo oggetto etichettato per
  video; classi e risoluzione non confermate).

Raccomandazione di sonnet-1: per gli elicotteri nessun dataset pubblico rispetta le soglie; TartanAviation resta
l'unico dataset reale di aerei da terra.

## Le fonti di AOD4 (13/10/2026)

Paper: "Introducing AOD 4" (Data in Brief 2024). Fonti dichiarate: YouTube-8M, Anti-UAV, un dataset Roboflow. Dai
nomi dei file locali (`data/aod4`, 23.028 immagini) si risale al **video d'origine**
(script: `aod4_fonti.py` nello scratchpad della sessione):

| fonte (prefisso del file) | video | immagini | box per classe |
|---|---|---|---|
| YouTube-8M (`yt-<id YouTube>-<n>`) | 142 | 10.182 | aereo 6.926, elicottero 5.526, uccello 2.171, drone 867 |
| Anti-UAV RGB (`20190925_101846_1_1_...`) | 23 sequenze | 5.810 | drone 5.808 |
| Svanström (`V_AIRPLANE_046_...`) | 185 | 4.942 | elicottero 2.374, drone 1.735, aereo 974 |
| foto stock Pexels/Unsplash | ~400 foto singole | 2.094 | uccello 5.729 |

- **Il leakage si può correggere**: 289 video su 350 hanno frame in più split, ma l'ID del video c'è, quindi si può
  rifare lo split per video.
- **Risoluzione**: tutte le immagini sono 1024×1024, ma **stirate** da Roboflow (il 16:9 diventa quadrato) e la parte
  Svanström è **ingrandita** da 640×512 (nessun dettaglio reale sopra 640).
- **Confusione classe-fonte**: gli uccelli vengono soprattutto da foto stock (stile diverso dai video), i droni
  soprattutto da Anti-UAV e Svanström, aerei ed elicotteri soprattutto da YouTube.
- Se Svanström fa da test esterno, i suoi 185 video vanno tolti da AOD4.

## Decisione aggiornata (13/10/2026)

Sostituisce la decisione sulla combinazione A. Base **Svanström** (tutte e 4 le classi etichettate negli stessi video,
stessa camera, da terra), più aggiunte per i buchi: **Anti-UAV** a colori per i droni grandi, un piccolo set di
**uccelli e aerei grandi** da video a licenza libera (Pexels, Pixabay). AOD4 non si usa. Training sempre a 640×640;
la risoluzione diventa una condizione di deploy (480p, 720p, 1080p). Dettagli in NEW_MASTERPLAN.md (regole fisse e
fase 0).
