# Dataset

| dataset | uso | dove | come si ottiene |
|---|---|---|---|
| AOD4 | training da zero, search e finetune dei potati | `data/aod4/` | `scripts/provision/x86_local.sh` (Mendeley) |
| Svanström | adapt (ultimo passo prima dell'export) e tutto il resto (valutazione, calibrazione) | `data/svanstrom/<subset>/` | `python tools/svanstrom.py` (anche dal provisioning) |

Entrambi hanno le stesse 4 classi nello stesso ordine: `0 airplane`, `1 bird`,
`2 drone`, `3 helicopter`. La testa del modello resta compatibile.

## AOD4

[AOD 4](https://data.mendeley.com/datasets/cd5z895tr2/1) (doi
10.17632/cd5z895tr2.1): 22.516 immagini, split `train/valid/test` degli autori.
Lo split è casuale **per immagine**: frame consecutivi dello stesso video
finiscono in split diversi, quindi le mAP su AOD4 sono ottimistiche. Per
questo serve solo al training da zero e non per misurare.

`calib` sono 512 immagini di train a seed fisso: disgiunte da val e test.

## Svanström

[Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset)
(Svanström et al., CC0): 650 video con label MATLAB, ~10 s a 30 fps. Usiamo
solo i **285 video visibili**, 640×512; gli IR (320×256, grigio) sono esclusi.
L'xlsx del dataset dà per ogni video classe e *distance bin* (close, medium,
distant): lo usiamo solo per l'elenco dei file. Il distance bin non separa
bene le dimensioni a schermo (un elicottero lontano può essere più grande di
un drone vicino), quindi le fasce si calcolano dai box.

### Pipeline (`tools/svanstrom.py`)

1. **Download** da GitHub di xlsx, video e label, con 10 tentativi a 10 s
   (proxy instabile). Già scaricato = saltato.
2. **Label.** I `.mat` sono oggetti MCOS: né scipy né `mcos-decoder` li
   leggono bene (`mcos-decoder` legge come vuoti i frame con 2 oggetti e perde
   quelli con 3-5). Un parser MAT5 minimo estrae 4 celle per frame, una per
   classe. Verifica per ogni clip: numero di label = numero di frame del video,
   altrimenti la clip è scartata (V_AIRPLANE_002).
3. **Selezione dei frame** con un hash semantico, per non tenere 300 frame
   quasi identici per video. Un frame è nuovo se il dHash del ritaglio attorno
   al bersaglio cambia (soglia `--hash-thr`, 8 bit su 64) **oppure** il box si
   sposta o cambia dimensione (`--geo-thr`). L'hash sul frame intero non
   funziona: il cielo domina e tutti i frame sembrano uguali. Dimensione
   originale preservata; frame scritti in `data/svanstrom_src/frames/<video>/`.
   Ogni immagine ha una **fascia** secondo le soglie COCO sulla √area del
   box più grande: small < 32 px, medium 32-96, large > 96. Per immagine e non
   per box: togliere un box dalle label insegnerebbe che lì non c'è niente.
   A `imgsz=640` il frame 640×512 non viene scalato, quindi le soglie valgono
   anche in training.
4. **Background.** I video hanno solo 82 frame senza oggetti, in 4 video. Per
   ogni video si aggiunge il 10% (`--bg-frac`) di frame non selezionati in cui
   il bersaglio è su cielo uniforme (`--bg-std`), con il bersaglio cancellato
   per inpainting (`cv2.inpaint`, Telea). Label vuota, suffisso `_bg`.
5. **Split 70/20/10 per video**, mai per frame (era il leakage di AOD4).
   Per ogni classe i video, in ordine casuale a seed fisso, vanno allo split
   più indietro rispetto al target sul loro mix di frame small/medium/large:
   così si bilanciano insieme classi e fasce. L'ordine casuale evita che i
   video lunghi finiscano tutti in train. Il background segue il suo video,
   quindi si bilancia da sé.
6. **Costruzione** in `data/svanstrom/<subset>/images|labels/{train,val,test,calib}`
   come hardlink ai frame: nessuna copia. `calib` = 512 immagini di train.
7. **Report** in `data/svanstrom/<subset>/report/`: immagini per split,
   classi per split (con il background), box per split come il `labels.jpg`
   di Ultralytics (campione di box, centri, larghezza × altezza), quota di ogni
   cella classe × fascia per split (solo in `all`). `data/svanstrom/manifest.json` registra
   parametri, split e frame scelti per ogni video.

### Sottoinsiemi

| `dataset.subset` | contenuto |
|---|---|
| `all` | tutto, background compreso (default) |
| `small`, `medium`, `large` | immagini con oggetti della fascia (box più grande) |
| `background` | solo frame `_bg`, da aggiungere a mano (data yaml con liste di cartelle) |

```bash
python run.py ... dataset=svanstrom dataset.subset=close
```

Gli split sono gli stessi in ogni sottoinsieme: un'immagine di test in `all`
è in test anche nella sua fascia.

### Numeri (seed 42)

284 video su 285 (V_AIRPLANE_002 scartato), 88.932 frame: 25.991 selezionati
(29%) più 2.462 di background.

| subset | train | val | test | calib |
|---|---|---|---|---|
| `all` | 20.234 | 5.523 | 2.696 | 512 |
| `small` | 9.695 | 2.569 | 1.260 | 512 |
| `medium` | 7.137 | 1.953 | 992 | 512 |
| `large` | 1.670 | 497 | 206 | 512 |
| `background` | 1.721 | 503 | 238 | 512 |

Video per split: 199 / 56 / 29. Frame per fascia divisi train/val/test:
small 71,7 / 19,0 / 9,3%, medium 70,8 / 19,4 / 9,8%, large 70,4 / 20,9 / 8,7%.
Il background è l'8,5% di `all`. Le immagini large sono quasi tutte di
elicotteri e aerei: drone large sono 37 frame di un solo video, bird large
nessuno.

### Limiti

- Gli oggetti non annotati restano non annotati (es. stormi in alcuni video
  di uccelli): anche nei frame `_bg`.
- Le immagini large sono poche (~10%) e concentrate in pochi video (i
  passaggi ravvicinati): la loro quota per split oscilla più delle altre.
- I video della stessa sessione di volo potrebbero finire in split diversi:
  l'xlsx non dice la sessione.
