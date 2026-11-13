# Pavimento della S sul Pi 5

Scopo: misurare la latenza del **pavimento** della S (YOLO26 w0.5 D126, 640, 4 classi) con tutte le maschere
potabili di yolopit al minimo, per il test del pavimento di NEW_MASTERPLAN §4 (soglia 70% di B = 19,6 ms; la S con la
sola LUT+fit era stimata 29-30 ms). Controlli: lo stesso pavimento per il nano (w0.25 D126, stimato 19,5 ms) e le due
reti intere (nano intero misurato 56,1 ms).

```
python experiments/pruning/pavimento_s/run.py build     # reti in runs/nets, results/nets.csv, layers.csv
python experiments/pruning/pavimento_s/run.py measure   # Pi -> results/raw.csv (+ env.json, runs/latencies_raw.txt)
```

Pavimento: `build_pit_model(n=2)` di yolopit, ogni maschera potabile con il solo blocco sempre tenuto (2 canali),
export `export_pruned` e IR OpenVINO fp16 come lo stage `export` (src.lut.b4r). Misura: src.lut.pi (come B4-R):
benchmark_app, 50 iterazioni, 3 ripetizioni in ordine mischiato, guardia termica 75/65 C, 2,4 GHz governor
performance, cpu_4. Il Pi era a 33 C e scarico; temperatura massima in misura 57,9 C, nessun throttling.

## Risultato (13/11/2026, mediana delle 3 ripetizioni; ms)

| rete | rep 0 | rep 1 | rep 2 | mediana |
|---|---|---|---|---|
| S pavimento | 40,96 | 41,03 | 41,05 | **41,0** |
| nano pavimento | 22,56 | 22,34 | 22,25 | **22,3** (stima LUT+fit 19,5) |
| nano intero | 56,67 | 56,05 | 56,36 | 56,4 (56,1 in B4-R) |
| S intera | 142,71 | 142,70 | 142,99 | 142,7 |

La S potata al massimo sta a 41,0 ms: sopra 28 ms (B) e sopra 19,6 ms. Non passa il test del pavimento (non può
arrivare a 28 ms, nemmeno con la potatura massima).

Strati che non scendono a 2 canali (7 su 74, in entrambe le reti; vincoli di yolopit: uscite che alimentano
C2PSA/PSABlock/Detect congelate): `layers.8.cv2`, `9.cv2`, `16.cv2`, `19.cv2`, `22.cv1b`, `22.m.0.0.cv2`, `22.cv2`
(S: c_out 512, 512, 128, 256, 256, 256, 512). Il pavimento misurato li contiene a larghezza piena, come "resto
invariato"; i punti con c_in 514/258 stanno fuori dalla griglia della LUT. Il nano pavimento misurato (22,3) supera
la stima 19,5 di 2,8 ms: il controllo non torna e la stima del nano è ottimistica (o la LUT somma meno dei 7 strati
pieni); il pavimento di S da stima (29-30 ms) è invece sottostimato rispetto alla misura (41,0).
