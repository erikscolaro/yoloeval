# Quale crop rende di più in mAP

**Scopo**: scegliere lo zoom del crop adattativo (`src/crop_tracker.py`, `docs/tracker.md`) in base alla precisione:
quanti pixel deve occupare l'oggetto nel crop perché P- lo trovi e lo classifichi meglio, per classe e per fascia di
grandezza.

**Stato**: chiuso il 19/11/2026. La regola del crop di oggi (oggetto al massimo al 60% del crop, zoom al massimo 2×,
crop 256) cade già vicino all'ottimo per mAP50 da 16 px in su; nessuna modifica.

## Setup

- **Modello**: P- (`yolo26w0.125_D101`, `e7d00a5f`), one-to-one, conf 0,001, max 64 detection.
- **Dati**: tutti i 7.389 oggetti del validation set (immagini 640×512 native).
- **Crop**: per ogni oggetto un quadrato centrato su di lui, con un errore di centratura uniforme fino al 10% del lato
  (simula il Kalman), portato a 256×256 (INTER_AREA se riduce, lineare se ingrandisce, grigio 114 fuori
  dall'immagine). Lo zoom fa occupare all'oggetto T px nel crop, con T in {16, 24, 32, 48, 64, 96, 128, 160, 192}.
- **Metrica**: AP COCO a 101 punti, mAP50 e mAP50-95 per classe e per fascia della tesi (lato √(w·h) nell'immagine
  originale: <8, 8–16, 16–32, 32–64, 64–128, 128–256, >256 px). In un crop conta solo il suo oggetto: le detection
  sugli altri oggetti del crop sono ignorate, le altre sono falsi positivi.
- **Riferimento**: l'immagine intera a 640, per fascia come COCO (GT fuori fascia ignorati, detection fuori fascia
  senza GT non contate).

```bash
experiments/tracking/crop_zoom_map/rifai.sh          # circa 20 minuti sulla GPU; log in runs/, json in results/
```

## Risultati

Media sulle classi presenti (`results/crop_map_*_c256.json` ha i valori per classe).

| fascia | oggetti | 640 mAP50 | crop: miglior mAP50 | 640 mAP50-95 | crop: miglior mAP50-95 |
|---|---|---|---|---|---|
| <8 | 19 | 0,254 | 0,671 (T16) | 0,068 | 0,372 (T16) |
| 8–16 | 943 | **0,492** | 0,468 (T16) | **0,196** | 0,189 (T16) |
| 16–32 | 1.122 | 0,729 | **0,789** (T32) | 0,395 | **0,408** (T32) |
| 32–64 | 2.016 | 0,757 | **0,785** (T128) | **0,520** | 0,496 (T96) |
| 64–128 | 1.762 | 0,902 | **0,915** (T160) | **0,653** | 0,638 (T128) |
| 128–256 | 1.372 | 0,942 | **0,958** (T160) | **0,758** | 0,738 (T128) |
| >256 | 155 | 0,914 | **0,938** (T128) | **0,775** | 0,768 (T128) |

mAP50 media sulle classi per T (le altre colonne nel json):

| fascia | T16 | T24 | T32 | T48 | T64 | T96 | T128 | T160 | T192 |
|---|---|---|---|---|---|---|---|---|---|
| 8–16 | **0,468** | 0,464 | 0,381 | 0,334 | 0,335 | 0,376 | 0,342 | 0,263 | 0,173 |
| 16–32 | 0,533 | 0,718 | **0,789** | 0,787 | 0,783 | 0,668 | 0,582 | 0,545 | 0,455 |
| 32–64 | 0,495 | 0,593 | 0,629 | 0,690 | 0,732 | 0,781 | **0,785** | 0,778 | 0,738 |
| 64–128 | 0,601 | 0,699 | 0,729 | 0,820 | 0,852 | 0,885 | 0,899 | **0,915** | 0,910 |
| 128–256 | 0,452 | 0,614 | 0,700 | 0,805 | 0,871 | 0,917 | 0,940 | **0,958** | 0,940 |

## Commento

- **Lo zoom migliore cresce con l'oggetto, ma meno che in proporzione**: circa 32 px nel crop per gli oggetti di
  16–32 px (zoom 1–2×), 96–128 per quelli di 32–64 (2–3×), 128–160 da 64 px in su (fino a ridurre). Sotto 16 px
  ingrandire non serve: il meglio è a zoom circa 1, pari al 640.
- **Il crop aiuta la mAP50, cioè trovare e classificare**: +1 a +6 punti da 16 px in su, il massimo nella fascia
  16–32 (0,789 contro 0,729).
- **La mAP50-95 no**: da 32 px in su la box del crop è un poco meno precisa (−0,7 a −2,4 punti). Solo in 16–32 il
  crop vince anche lì.
- **La regola di oggi ci sta vicina**: lato del crop = max(incertezza del Kalman, lato dell'oggetto / 0,6, 128 px),
  portato a 256. Per un oggetto di 30 px dà T ≈ 60 (0,783 contro il meglio 0,789), di 50 px T ≈ 100, di 100 px
  T ≈ 150, di 200 px T ≈ 150: tutti entro un punto dall'ottimo di mAP50.

## Limiti

- Nei crop i falsi positivi possibili sono solo quelli dentro il crop; nell'immagine intera sono su tutta l'immagine.
  Il confronto con il 640 è quindi un po' favorevole ai crop; fra i diversi T è alla pari.
- Un seed, un errore di centratura (10%), un solo modello; la fascia <8 ha 19 oggetti, i suoi numeri sono rumore.
- Le immagini del val sono già ritagli 640×512 attorno agli oggetti: non c'è il caso di un oggetto piccolo in un
  fotogramma 1080p, dove il crop rende di più rispetto a un fit che riduce.

## S- a crop 128 contro 256 (24/11/2026)

S- (`yolo26w0.5_D101`, `21d2b4bd`), stessi crop e stesso seme; a 128 T fino a 96 (`rifai.sh`, ultima riga). Classe
drone, mAP50 / mAP50-95 al T migliore per la mAP50-95 (`results/crop_map_*21d2b4bd_c{128,256}.json`):

| fascia | 640 | crop 256 | crop 128 |
|---|---|---|---|
| <8 | 0,257 / 0,180 | 1,000 / 0,701 (T24) | 0,752 / 0,407 (T24) |
| 8–16 | 0,725 / 0,405 | 0,805 / 0,452 (T16) | 0,750 / 0,402 (T16) |
| 16–32 | 0,850 / 0,520 | 0,765 / 0,489 (T24) | 0,837 / 0,445 (T24) |
| 32–64 | 0,710 / 0,465 | 0,778 / 0,490 (T128) | 0,730 / 0,468 (T48) |
| 64–128 | 0,929 / 0,600 | 0,968 / 0,650 (T160) | 0,926 / 0,578 (T96) |
| 128–256 | 0,960 / 0,764 | 0,956 / 0,755 (T160) | 0,899 / 0,626 (T96) |
| >256 | 0,918 / 0,739 | 0,954 / 0,729 (T128) | 0,883 / 0,645 (T64) |

- Il crop 128 perde in mAP50-95 in tutte le fasce rispetto al 256, anche sui droni piccoli (8–16: 0,402 contro 0,452),
  dove lo zoom è lo stesso: a 128 resta meno contesto intorno all'oggetto.
- Da 64 px in su il crop 128 è sotto anche il fotogramma intero a 640 (mAP50-95 0,626 contro 0,764 in 128–256).
- La fascia <8 ha pochi droni (19 oggetti in tutto): il numero ballerà.
