# SCAN a 720p senza cambiare le proporzioni sul Raspberry Pi 5

**Scopo**: latenza del modello di SCAN quando la camera è 720p e il fotogramma si riduce a 640 sul lato lungo
mantenendo le proporzioni: 640×360, più un bordo fino a 384 (multiplo di 32). Ingresso `conf/deploy/640x384.yaml`.
Oggi la griglia ha solo 640×640, che per un 720p spreca il 44% dei pixel nel bordo.

In più rimisura N (larghezza 0,25) a 640×640: N a D126 (27,9 ms) risultava più veloce di N a D101 (30,5 ms).
Prima della rimisura i json vecchi sono stati copiati in `results/vecchi_640/`.

- Larghezze F…N × D101/D126/D170, pesi casuali, OpenVINO fp16, 4 core, governor performance, 3 passate.
- Budget del modello di SCAN a 720p: 30,7 ms (resize, bordo, pre e post misurati in `../scan_budget_720p`).

```bash
experiments/latenza/scan_modelli_640x384/rifai.sh   # train (pesi casuali), export, benchmark con il lock del Pi
```

## Risultati (20/11/2026, mediana di 3 passate)

Latenza a 640×384 (ms) e mAP50-95 (val a 640×640, training e300, media fra i semi):

| larghezza | D101 | D126 | D170 | mAP D101 / D126 / D170 |
|---|---|---|---|---|
| F | 10,4 | 11,5 | 14,3 | 0,531 / 0,532 / 0,535 |
| FP | 13,2 | 14,5 | 18,3 | 0,560 / 0,557 / 0,558 |
| P | 15,6 | 17,0 | 21,6 | 0,570 / 0,569 / 0,577 |
| PN | 24,3 | 26,2 | 32,8 | 0,587 / 0,588 / 0,597 |
| N | 31,4 | 33,5 | 42,3 | 0,596 / 0,596 / 0,601 |

- 640×384 costa circa il 60% di 640×640 (il 60% dei pixel).
- Budget 30,7 ms (+5% = 32,2 ms). Entro il budget il migliore è **PN-** (24,3 ms, 0,587). N- (31,4 ms, 0,596) sta
  nella fascia del +5% e guadagna 0,009 di mAP: sotto il punto di pari merito (0,01), quindi si sceglie PN-.
- **N a 640×640 rimisurato** (mediana di 3 passate, 20/11/2026): N- 53,3 ms, N 56,6 ms, N+ 70,5 ms. Le misure del 13/11 davano 53,4, 56,5 e 70,9 ms. Il 27,9 ms (e il 30,5 ms di
  N-) veniva da un errore di `tools/figure_sintesi.lat_misurate`: la mediana metteva insieme la griglia e le reti
  potate (`strategy=pit_latency`) con la stessa larghezza nominale. Corretto il 20/11/2026 (solo `baseline`). Le
  figure del batch (`../track_modelli_batch_crop256`) non cambiano.
