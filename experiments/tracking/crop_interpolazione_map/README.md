# INTER_AREA o INTER_LINEAR nel crop tracker

**Scopo**: scegliere l'interpolazione dei due resize del crop tracker (`src/crop_tracker.py`) guardando la precisione.
Sul Pi INTER_LINEAR costa meno (`experiments/latenza/track_costo_ritaglio`): lo SCAN da 1080p passa da 11,5 a 2,9 ms, un
crop 720 → 256 da 3,7 a 0,84 ms. Domanda: quanta mAP si perde?

**Stato**: misurato il 20/11/2026. Scelta del 23/11/2026 (utente): **INTER_AREA ovunque** (SCAN e crop), conservativa per preservare la qualità dell'immagine. La proposta originale era AREA nello SCAN e LINEAR nel crop.

## Setup

- **Modello**: P- (`yolo26w0.125_D101`, `e7d00a5f`), one-to-one, conf 0,001, max 64 detection.
- **SCAN**: i 17 video Anti-UAV di val (`data/tracking/val`, 1920×1080, solo drone), un fotogramma ogni 5 (3.243
  fotogrammi). Il fotogramma si riduce a 640 di lato con AREA o LINEAR, poi il modello a 640 (Ultralytics aggiunge
  solo il padding). mAP del drone per fascia della tesi, lato √(w·h) nel fotogramma originale.
- **Crop**: lo stesso esperimento di `../crop_zoom_map` (val, 7.389 oggetti, crop 256, zoom T, stesso seme e stessi
  crop), con LINEAR anche quando il crop si riduce. Il riferimento AREA è `../crop_zoom_map/results`. `run.py` usa il
  codice di `../crop_zoom_map/run.py` senza modificarlo.

```bash
experiments/tracking/crop_interpolazione_map/rifai.sh     # GPU, circa 15 minuti; log in runs/, json in results/
```

## Risultati

**SCAN**, drone, mAP50 / mAP50-95:

| fascia | oggetti | AREA | LINEAR |
|---|---|---|---|
| 16–32 | 409 | **0,396** / **0,103** | 0,312 / 0,081 |
| 32–64 | 695 | **0,718** / **0,312** | 0,665 / 0,283 |
| 64–128 | 1.620 | 0,984 / 0,545 | 0,983 / 0,532 |
| 128–256 | 448 | 1,000 / 0,625 | 1,000 / 0,623 |

Con LINEAR i droni piccoli perdono da 5 a 8 punti di mAP50: la riduzione 1080p → 640 (3×) toglie dettagli. Da 64 px
in su non cambia niente.

**Crop**, media sulle classi, allo zoom che la regola del crop usa per ogni fascia (`../crop_zoom_map`):

| fascia | zoom T | AREA mAP50 | LINEAR mAP50 | AREA mAP50-95 | LINEAR mAP50-95 |
|---|---|---|---|---|---|
| 8–16 | 16 | 0,468 | 0,468 | 0,189 | 0,189 |
| 16–32 | 32 | 0,789 | 0,789 | 0,408 | 0,408 |
| 32–64 | 128 | 0,785 | 0,785 | 0,489 | 0,489 |
| 64–128 | 160 | 0,915 | 0,915 | 0,624 | 0,624 |
| 128–256 | 160 | 0,958 | 0,958 | 0,724 | 0,722 |

Agli zoom usati la differenza è sotto 0,003. LINEAR perde (fino a 9 punti) solo con riduzioni forti, cioè oggetti
grandi con T = 16–32, che la regola del crop non usa. Il json con tutti gli zoom è in `results/crop_linear/`.

## Proposta originale (superata dalla scelta del 23/11)

- **SCAN: INTER_AREA.** Costa 8,6 ms in più a 1080p, ma è dove si trovano gli oggetti piccoli e nuovi.
- **Crop: INTER_LINEAR.** Stessa mAP, e un crop grande costa fino a 2,9 ms in meno.

## Limiti

- SCAN misurato solo sul drone di Anti-UAV (2 sessioni di ripresa); Svanström è a 640×512 e non si riduce.
- Le fasce 8–16 e >256 non hanno oggetti nei video Anti-UAV di val.
- Solo P-: un modello più grande può reagire in modo diverso alla perdita di dettaglio.
