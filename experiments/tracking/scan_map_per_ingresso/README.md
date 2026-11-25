# mAP dello SCAN con ingresso 640×512 o 640×384

**Scopo**: con un solo export dello SCAN a 640×384, le immagini Svanström (640×512) vanno deformate o ridotte. Quanto si
perde rispetto all'ingresso nativo 640×512?
**Stato**: misurato il 23/11/2026 con P- e PN- (i due candidati dello SCAN).

## Setup

- **Modelli**: P- (`yolo26w0.125_D101`, `e7d00a5f`) e PN- (`yolo26w0.1875_D101`, `a033e56f`), one-to-one, conf 0,001,
  max 64 detection.
- **Dati**: tutto il val (6.659 immagini, 640×512, tutte le fonti).
- **Ingressi**:
  - `nativo`: 640×512, nessun resize;
  - `schiacciato`: resize a 640×384 con INTER_AREA, proporzioni cambiate;
  - `letterbox`: resize a 480×384 con INTER_AREA, bordo grigio fino a 640×384, proporzioni mantenute (oggetti ×0,75).
- **Metrica**: AP COCO a 101 punti (codice di `../crop_zoom_map/run.py`), media sulle classi, per fascia di grandezza
  (lato √(w·h) nell'immagine originale). I numeri assoluti sono più bassi del val di Ultralytics (0,570); conta il
  confronto fra le righe.

```bash
experiments/tracking/scan_map_per_ingresso/rifai.sh     # GPU, circa 5 minuti
```

## Risultati

**PN-**, mAP50 / mAP50-95 (tutto il val): nativo 0,826 / 0,560; schiacciato 0,769 / 0,496 (−0,057 / −0,064); letterbox
0,786 / 0,531 (−0,040 / −0,029). Le perdite hanno lo stesso ordine di P-.

**P-**, mAP50 / mAP50-95:

| fascia | nativo 640×512 | schiacciato 640×384 | letterbox 640×384 |
|---|---|---|---|
| **tutto** | **0,794 / 0,540** | 0,727 / 0,466 | 0,750 / 0,511 |
| <8 px | 0,254 / 0,068 | 0,072 / 0,011 | 0,003 / 0,000 |
| 8–16 | 0,493 / 0,196 | 0,344 / 0,126 | 0,389 / 0,152 |
| 16–32 | 0,729 / 0,396 | 0,560 / 0,282 | 0,587 / 0,316 |
| 32–64 | 0,757 / 0,520 | 0,714 / 0,460 | 0,718 / 0,487 |
| 64–128 | 0,902 / 0,652 | 0,870 / 0,594 | 0,888 / 0,640 |
| 128–256 | 0,942 / 0,758 | 0,919 / 0,685 | 0,947 / 0,755 |
| >256 | 0,914 / 0,775 | 0,895 / 0,731 | 0,920 / 0,784 |

Per classe (tutto): drone 0,770 / 0,506 nativo, 0,671 / 0,429 schiacciato, 0,707 / 0,473 letterbox; aereo, elicottero e
uccello perdono meno (fino a −0,11 di mAP50-95 per l'elicottero schiacciato).

## Commento

- **Schiacciare rompe**: −0,067 di mAP50 e −0,074 di mAP50-95. Gli oggetti da 8 a 32 px perdono da 0,15 a 0,17 di mAP50, e
  anche quelli grandi perdono precisione delle box (−0,04/−0,07 di mAP50-95 sopra 128 px).
- **Il letterbox perde meno ma perde**: −0,044 di mAP50 e −0,029 di mAP50-95. La perdita è tutta sugli oggetti piccoli
  (riduzione ×0,75): da 8 a 32 px −0,10/−0,14 di mAP50. Da 64 px in su è pari al nativo.
- **Quindi Svanström va valutato a 640×512.** Per il deploy non servono due export: una camera ha una sola risoluzione,
  quindi lo stesso IR si porta alla forma della camera all'avvio (`model.reshape` di OpenVINO) prima di `compile_model`.
  Da verificare: la latenza di P- a 640×512 sul Pi (stima: mediana circa 21 ms, massimo circa 30 ms, budget circa 31 ms).

## Limiti

- Un solo seme per modello; per PN- solo il totale nel testo (le fasce sono nel json).
- Il val contiene tutte le fonti (non solo Svanström), tutte a 640×512.
