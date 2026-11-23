# Miglior modello dello SCAN per camera e ingresso

**Scopo**: per ogni coppia (risoluzione della camera, ingresso del modello) il miglior modello dello SCAN che sta nel
budget a 30 fps nel caso peggiore, con la sua mAP. Schema di partenza: uno schizzo del 23/11/2026.
**Stato**: misurato il 23/11/2026.

## Setup

- **Camera** (16:9): 640 (640×360), HD (1280×720), FHD (1920×1080). **Ingresso del modello**: 640×384, HD (1280×736),
  FHD (1920×1088). Il fotogramma si riduce senza cambiare le proporzioni e si completa con il bordo grigio. Niente
  ingrandimenti: 6 celle valide (ingresso ≤ camera). La diagonale è "senza scalatura".
- **`misura.py`**: costo fuori dal modello per cella sul Pi 5 (resize, bordo, pre), un core, 3 passate, AREA e LINEAR.
- **Budget** = 33,3 ms − p99 del fuori modello − post (0,009 ms) − Kalman con un track (0,21 ms).
- **Modello nel budget**: il massimo delle sue passate (stima prudente del p99) è sotto il budget. Fra i 27 modelli della
  griglia (reti `baseline`, OpenVINO fp16, Pi 5) si prende la mAP massima. Se più modelli sono entro 0,01 di mAP si
  prende il più veloce.
- **mAP**: mAP50-95 del val 640×640, uguale per ogni cella. Per HD e FHD non esiste una mAP misurata a quella risoluzione.

```bash
experiments/latenza/scan_modelli_per_camera_e_ingresso/rifai.sh   # circa 3 minuti; tabella in results/celle.csv
```

## Risultati

| camera | ingresso | interp. | fuori modello | budget | modello | latenza (mediana / max) | mAP50-95 |
|---|---|---|---|---|---|---|---|
| 640 | 640×384 | AREA | 2,0 ms | 31,1 ms | PN- | 24,3 / 27,5 ms | 0,587 |
| 640 | 640×384 | LINEAR | 2,0 ms | 31,1 ms | PN- | 24,3 / 27,5 ms | 0,587 |
| HD | 640×384 | AREA | 2,7 ms | 30,4 ms | P+ | 21,6 / 24,4 ms | 0,577 |
| HD | 640×384 | LINEAR | 3,1 ms | 30,1 ms | P+ | 21,6 / 24,4 ms | 0,577 |
| HD | HD | AREA | 7,8 ms | 25,3 ms | nessuno | — | — |
| HD | HD | LINEAR | 7,8 ms | 25,3 ms | nessuno | — | — |
| FHD | 640×384 | AREA | 9,2 ms | 23,9 ms | P- | 15,6 / 23,9 ms | 0,570 |
| FHD | 640×384 | LINEAR | 2,2 ms | 31,0 ms | PN- | 24,3 / 27,5 ms | 0,587 |
| FHD | HD | AREA | 21,4 ms | 11,7 ms | nessuno | — | — |
| FHD | HD | LINEAR | 9,7 ms | 23,4 ms | nessuno | — | — |
| FHD | FHD | AREA | 17,3 ms | 15,8 ms | nessuno | — | — |
| FHD | FHD | LINEAR | 17,3 ms | 15,8 ms | nessuno | — | — |

## Commento

- **Con ingresso HD o FHD nessun modello della griglia sta nel budget.** Il più piccolo (F-, larghezza 0,0625) costa
  41 ms a 1280×736 e 152 ms a 1920×1088, contro budget di 25 e 16 ms.
- **Ingresso 640×384**: lo SCAN regge con camera 640, HD e FHD. La mAP cala da 0,587 a 0,577 (HD, P+) fino a 0,570 (FHD con
  AREA, P-).
- **AREA contro LINEAR dipende dalla camera**: con HD la differenza è 0,3 ms. Con FHD AREA costa 9,2 ms contro 2,2 ms (fattore 3
  intero, ma senza SIMD) e costringe a un modello più piccolo: −0,017 di mAP rispetto a LINEAR.
- **FHD → HD con AREA costa 21 ms** (fattore 1,5, caso lento di OpenCV).
- **Scelta dei pari merito**: con camera HD e ingresso 640×384, PN- (0,587) ha mAP massima nel budget. P+ (0,577) è entro
  0,01 (0,0098) e più veloce di 2,7 ms: la regola sceglie P+. Per lo SCAN a 720p avevamo scelto PN-: con questa regola
  la scelta è borderline. Da decidere.

## Limiti

- mAP del val 640 per tutte le celle, non misurata alla risoluzione della cella.
- Le latenze di 720p e 1080p sono quelle del 4/11/2026 (testa a k=300), da rimisurare a k=64.
- La camera 640 è 640×360; il bordo grigio porta l'ingresso a 640×384.
- Un solo track (Kalman 0,21 ms); il massimo delle passate è una stima del p99.
