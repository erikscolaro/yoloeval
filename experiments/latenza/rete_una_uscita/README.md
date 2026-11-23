# Neck e testa a una sola scala (reti lineari) sul Raspberry Pi 5

**Scopo**: vedere quanto costa il neck. Tre reti "lineari" prendono una sola uscita del backbone del nano
(N, larghezza 0,25, D126) e la mandano direttamente al Detect, senza neck PAN. Nessun training (pesi casuali): solo
parametri, MAC e latenza.

| rete | cosa contiene |
|---|---|
| nano | riferimento: backbone + neck + Detect(P3, P4, P5) |
| L3 | backbone fino al layer 4 (P3, stride 8) → Detect |
| L4 | backbone fino al layer 6 (P4, stride 16) → Detect |
| L5 | backbone fino al layer 10 (P5, stride 32, con SPPF e C2PSA) → Detect |

- `run.py`: `build` (yaml da `src.models.arch(0.25, "D126")` tagliato, .pt, ONNX fp32, IR fp16 come l'`export` della
  pipeline, a 640×640, 4 classi, testa one-to-one con 64 box), `measure`, `table`.
- Parametri e MAC: dall'ONNX fp32 (`src.measure.complexity.onnx_complexity`), come il resto del repo: contano solo la
  testa one-to-one e le conv/MatMul.
- Latenza: Pi 5, OpenVINO fp16, 4 core, governor performance, 300 iterazioni, 3 passate in ordine casuale, mediana
  (`src/lut/pi.py`, le stesse condizioni della LUT e della griglia).

```bash
experiments/latenza/rete_una_uscita/rifai.sh      # circa 10 minuti, con il lock del Pi
```

## Risultati (20/11/2026)

| rete | parametri | GMAC | latenza (ms) | rispetto al nano |
|---|---|---|---|---|
| nano | 2.409.768 | 2.66 | 56.5 | 100% |
| L3 | 187.964 | 1.25 | 32.2 | 57% |
| L4 | 397.438 | 1.23 | 29.4 | 52% |
| L5 | 1.680.124 | 1.68 | 34.9 | 62% |

## Limiti

- Una sola larghezza (nano) e una sola profondità (D126), a 640×640.
- Nessuna mAP: le reti sono casuali e non addestrate. Non sappiamo se una rete lineare detecta bene i droni piccoli.
- L3 e L4 non hanno l'attenzione (C2PSA) né SPPF, che stanno dopo il layer 6.
- La testa (Detect) è sempre quella one-to-one a una scala sola: canali e stride cambiano.
