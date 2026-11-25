# Confronto: P-, S-, Kalman con P-, Kalman con crop

**Scopo**: capire se il tracker con crop (P- in SCAN, S- su un crop 256 in TRACK) vale la pena rispetto ai detector da
soli, e in particolare rispetto a S- sul fotogramma intero, che sul Pi non sta nel budget a 30 fps.

**Comando**: `bash experiments/tracking/confronto_p_s_crop/rifai.sh [split=val]`. Servono i risultati di
`../taratura_kalman_crop` (e quindi di `../taratura_kalman`); le detection di S- a fotogramma intero si fanno se mancano
(`../taratura_kalman/detection.py --modello S-`, GPU, circa 10 min).

## Metodo

- Stessa metrica della taratura: F2 sui droni per fotogramma (IoU >= 0,5), recall, precisione, quota di aerei ed
  elicotteri veri con una box drone sopra.
- P- e S- da soli: fotogramma ridotto a 640 (INTER_AREA), ognuno alla sua soglia con F2 massimo sul totale (P- 0,02,
  S- 0,01).
- Kalman con P-: parametri dello studio F2 `tutte`. Kalman con crop: prova migliore di `../taratura_kalman_crop`.
- Gruppi: `tutti`; `1ogg` e `2+ogg` (fotogrammi con 0-1 o 2+ oggetti veri); `video_1ogg`, solo i video che non hanno
  mai 2+ oggetti, il caso del single tracking.

## Risultati (val, 24/11/2026; `results/confronto_val.txt`)

F2 sui droni (recall / precisione):

| Gruppo | P- | S- | Kalman P- | Kalman crop |
|---|---|---|---|---|
| Svanström, tutti | 0,428 (0,38 / 0,80) | **0,534** (0,49 / 0,84) | 0,421 | 0,461 (0,42 / 0,82) |
| Svanström, video con 1 oggetto | 0,362 | **0,463** | 0,350 | 0,402 |
| Svanström, fotogrammi 2+ oggetti | 0,649 | **0,780** | 0,659 | 0,654 |
| Anti-UAV | 0,775 (0,78 / 0,74) | 0,750 (0,73 / 0,87) | 0,830 | **0,837** (0,84 / 0,82) |
| Totale, tutti | 0,666 | 0,679 | 0,702 | **0,718** |
| Totale, video con 1 oggetto | 0,666 | 0,671 | 0,704 | **0,722** |

Aerei ed elicotteri chiamati drone (totale): P- 3,2%, S- 0,44%, Kalman P- 2,6%, Kalman crop 0,13%.

- Sul totale il tracker con crop supera S- da solo (0,718 contro 0,679), ma per merito di Anti-UAV, dove S- a 640 è
  sotto P-: i droni Anti-UAV sono grandi e il Kalman tiene il track.
- Su Svanström (oggetti piccoli) S- da solo resta sopra: il crop recupera circa metà del distacco fra P- e S-
  (0,428 → 0,461, S- 0,534). Il limite è lo SCAN di P-: S- lavora solo dove P- ha già trovato l'oggetto.
- Il crop ha meno aerei chiamati drone di tutti, anche di S- (0,13% contro 0,44%).
- Togliere i video con più oggetti cambia poco sul totale (0,718 → 0,722). Sui fotogrammi con 2+ oggetti il crop è
  pari a P-: con un solo crop vede un oggetto.

## Crop 128 e 64 con i parametri del 256 (24/11/2026; `results/confronto_val_c{128,64}.txt`)

`confronto.py --crop 128` e `--crop 64`: stesso tracker e stessi parametri di Optuna (trovati sul 256), cambia solo il lato del crop
di S-. F2 sui droni:

| Gruppo | P- | S- | Kalman P- | Kalman crop 256 | Kalman crop 128 | Kalman crop 64 |
|---|---|---|---|---|---|---|
| Svanström, tutti | 0,428 | 0,534 | 0,421 | 0,461 | 0,466 | 0,476 |
| Svanström, video con 1 oggetto | 0,362 | 0,463 | 0,350 | 0,402 | 0,409 | 0,424 |
| Anti-UAV | 0,775 | 0,750 | 0,830 | 0,837 | 0,825 | 0,786 |
| Totale, tutti | 0,666 | 0,679 | 0,702 | 0,718 | 0,711 | 0,688 |
| Aerei/elicotteri chiamati drone | 3,2% | 0,44% | 2,6% | 0,13% | 0,41% | 1,7% |

- A 128 il tracker perde poco sul totale (−0,007), meno di quanto diceva la mAP dei crop (`../crop_zoom_map`): l'F2
  conta il drone trovato (IoU >= 0,5), non la precisione della box.
- Su Svanström il 128 è pari o un poco sopra; su Anti-UAV (droni grandi) perde 0,012. TRACK nel 64% dei fotogrammi
  contro 68%.
- I parametri non sono tarati sul 128: una ricerca dedicata potrebbe recuperare qualcosa.
- A 64 Svanström sale ancora (0,476), Anti-UAV scende (0,786, sotto il Kalman con P-), aerei chiamati drone 1,7%. Il
  lato del crop migliore dipende dalla grandezza dei droni: il 256 vince sul totale perché il val ha più fotogrammi
  Anti-UAV.
- Da verificare: sui fotogrammi con 2+ oggetti il crop dà lo stesso F2 (0,654) a 256, 128 e 64.

## Prova scartata: con piu' track, crop solo del piu' vecchio (25/11/2026)

Con piu' track confermati di `max_crops` il tracker torna allo SCAN. Si e' provato invece a fare il crop solo del track
con piu' hit (a parita', piu' confidenza; prima ancora per sola confidenza, con risultati quasi uguali), con gli
stessi parametri di Optuna. F2 sui droni, crop 256 / 128: Svanstrom 0,441 / 0,445 (contro 0,461 / 0,466), Anti-UAV
0,831 / 0,811 (contro 0,837 / 0,825), totale 0,707 / 0,695 (contro 0,718 / 0,711). La precisione sale di 2-4 punti, il
recall scende di piu': lo SCAN forzato dal secondo track aiuta P- a ritrovare il drone. Opzione tolta dal codice.
