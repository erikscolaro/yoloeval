# Video completi di val e test per il tracker

**Scopo**: tarare (val) e provare (test) il tracker di Kalman (`src/tracking.py`, `src/crop_tracker.py`) su video
con una box in ogni fotogramma, senza annotare niente. Le sequenze sono quelle dei gruppi di val e test di VEDETTA
(manifest del dataset), quindi non c'è leakage con il training.

- **Svanström**: mp4 originali, label rifatte e riviste in CVAT su tutti i frame delle clip
  (`vedetta/data/input/svanstrom/4_box/manuale.tar.gz`), classi convertite all'ordine di yoloeval.
- **Anti-UAV300 RGB**: frame dal mirror Hugging Face (MIT), label originali per frame (un drone o nessuno).

```bash
.venv/bin/python experiments/tracking/video_val_test/scarica.py   # circa 1 ora, 3,4 GB in data/tracking (non versionata)
```

Uscita: `data/tracking/<split>/<sequenza>/{images,labels}/<frame>.jpg|txt`, con `<frame>` l'indice nel video
originale, e `seq.json` (fonte, classe, split, gruppo, fps, frame).

## Contenuto (19/11/2026)

| split | fonte | classe | sequenze | frame | frame senza oggetto | frame con più oggetti |
|---|---|---|---|---|---|---|
| val | Anti-UAV | drone | 17 | 16.212 | 363 | 0 |
| val | Svanström | drone | 25 | 7.372 | 0 | 820 |
| val | Svanström | aereo | 10 | 1.700 | 0 | 0 |
| val | Svanström | elicottero | 7 | 1.617 | 0 | 316 |
| val | Svanström | uccello | 9 | 1.525 | 0 | 379 |
| test | Anti-UAV | drone | 10 | 9.348 | 496 | 0 |
| test | Svanström | drone | 9 | 2.386 | 0 | 2 |
| test | Svanström | aereo | 7 | 1.686 | 0 | 0 |
| test | Svanström | elicottero | 6 | 1.508 | 0 | 0 |
| test | Svanström | uccello | 11 | 1.206 | 0 | 33 |

## Limiti

- **Buchi**: 7 sequenze Svanström hanno frame mancanti (frame tolti in revisione), per esempio `V_DRONE_050_s0` ha
  97 frame su 310. Nei buchi lo stato del frame non è noto: la valutazione deve usare l'indice del frame per il dt e
  dividere la sequenza dove c'è un buco, oppure togliere queste sequenze.
- **ID**: le label YOLO non hanno l'ID degli oggetti. Anti-UAV ha sempre un oggetto; Svanström ha più oggetti in
  circa il 10% dei frame di val. Le metriche di detection e di classe non hanno bisogno dell'ID; HOTA e IDF1 sì.
- **Varietà**: i video Anti-UAV vengono da 2 sessioni in val e da 1 in test.
- **Notte**: non controllato se i video Anti-UAV hanno frame notturni (VEDETTA li ha tolti dal dataset).
