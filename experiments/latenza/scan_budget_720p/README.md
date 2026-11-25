# Budget del modello di SCAN a 720p sul Raspberry Pi 5

**Scopo**: quanto tempo resta al modello di SCAN a 30 fps con una camera 720p, riducendo a 640 sul lato lungo senza
cambiare le proporzioni. `budget = 33,3 ms − costo fuori dal modello`.

Il costo fuori dal modello misurato in `misura.py` (un core, 1000 fotogrammi dopo 100 di riscaldamento):
resize 1280×720 → 640×360, bordo grigio fino a 640×384 (ingresso `conf/deploy/640x384.yaml`), pre
(uint8 → float32 NCHW / 255), post (64×6 → soglia). INTER_AREA e INTER_LINEAR.

La misura vecchia (6,7 ms, `../track_costo_ritaglio`) stirava il 720p a 640×384 e cambiava le proporzioni.

```bash
experiments/latenza/scan_budget_720p/rifai.sh   # circa 1 minuto, aspetta il lock del Pi
```

## Risultati (20/11/2026, 3 passate)

| interpolazione | costo fuori dal modello | p99 | budget del modello |
|---|---|---|---|
| INTER_AREA | 2,63 ms | 2,69 ms | **30,7 ms** |
| INTER_LINEAR | 2,83 ms | 2,88 ms | 30,5 ms |

- Le passate differiscono di meno di 0,03 ms.
- Con 720p → 640 la riduzione è esattamente 2×: OpenCV fa INTER_AREA come media di blocchi 2×2, che costa come LINEAR
  o meno. Quindi a 720p si usa INTER_AREA senza costo in più (a 1080p, riduzione 3×, non vale).
- Il vecchio 6,7 ms veniva soprattutto dallo stiramento a 384 (riduzione non intera).

## Limiti

- Il passo del Kalman nello SCAN (solo predict, senza track) non c'è: è sotto 0,1 ms (`../track_budget_n_crop`).
