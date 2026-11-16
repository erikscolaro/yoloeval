# Tiling dei primi strati sul Raspberry Pi 5

**Scopo**: capire se conviene dividere l'immagine in tile e far processare ogni tile da un core, con gli stessi pesi
(parallelismo sull'ingresso), invece di lasciare che OpenVINO divida ogni nodo fra i 4 core. Si guardano solo i
primi strati: la coda della rete non interessa. Domanda collegata: OpenVINO divide davvero ogni conv sui 4 core?

**Stato**: chiuso il 16/11/2026.

## Setup

- **Modelli**: P `59d8534a` e N `9dca8d15` (D126), BN fusa. **Testa** = strati 0..k di YOLO26, con k = 0..3:
  Conv s2, Conv s2, C3k2, Conv s2. Sono in sequenza; il primo ramo laterale verso la testa di detection parte
  dallo strato 4.
- **Board**: Raspberry Pi 5, 2,4 GHz, governor performance, OpenVINO 2026.4.1, IR fp16 (`-infer_precision f16`)
  con I/O NHWC come le reti della LUT. Misura con la sessione di `src/lut/pi.py`: board tuned come il deploy,
  `chrt -f 80`, 200 iterazioni con riscaldamento a tempo, guardia termica.
- **Non tiled**: la testa sull'immagine 640×640, 1 stream da 4 thread (come il deploy) e da 1 thread.
- **Tiled**: T tile su griglia 2×2 (T = 4), 2×4 (8) e 4×4 (16), con halo; 4 stream da 1 thread e 4 richieste
  in volo, cioè ogni core una tile con gli stessi pesi. Si misurano le tile interne, le più grandi (con l'halo
  su tutti i lati), quindi la stima è per eccesso.
  - **Tempo dell'immagine** = ⌈T/4⌉ × mediana di una tile. Controllo: T / throughput, sempre entro +5%.
  - **Halo**: il più piccolo multiplo del passo per cui le tile ricucite danno *esattamente* l'uscita intera
    (`allclose` in fp32 sull'host, tutte le griglie).
  - **Non misurati**: il taglio e la ricucitura delle tile (copie di memoria). Il tempo tiled è quindi un limite
    inferiore.

```bash
python experiments/deploy/tiling_rpi5/run.py build      # teste, halo verificato, IR (host)
flock ~/Desktop/yoloeval/runs/.pi.lock python experiments/deploy/tiling_rpi5/run.py measure
python experiments/deploy/tiling_rpi5/run.py table      # results/tabella.csv
```

## Risultati

Tempo della testa per un'immagine, mediana in ms; speedup = non tiled a 4 thread / variante.

| modello | testa fino a | passo | halo | 4 thread (deploy) | 1 thread | T = 4 | T = 8 | T = 16 |
|---|---|---|---|---|---|---|---|---|
| P | strato 0 (Conv) | 2 | 2 px | 3,53 | 11,23 (3,2×) | 3,25 (**1,09×**) | 3,10 (**1,14×**) | 3,10 (**1,14×**) |
| P | strato 1 (Conv) | 4 | 4 px | 4,69 | 15,76 (3,4×) | 4,53 (1,04×) | 4,46 (1,05×) | 4,52 (1,04×) |
| P | strato 2 (C3k2) | 4 | 12 px | 8,39 | 29,78 (3,5×) | 9,18 (0,91×) | 9,66 (0,87×) | 10,20 (0,82×) |
| P | strato 3 (Conv) | 8 | 16 px | 9,03 | 32,51 (3,6×) | 10,52 (0,86×) | 11,28 (0,80×) | 12,08 (0,75×) |
| N | strato 0 (Conv) | 2 | 2 px | 4,58 | 15,39 (3,4×) | 4,39 (1,04×) | 4,34 (1,06×) | 4,20 (**1,09×**) |
| N | strato 1 (Conv) | 4 | 4 px | 6,63 | 23,43 (3,5×) | 6,68 (0,99×) | 6,70 (0,99×) | 6,68 (0,99×) |
| N | strato 2 (C3k2) | 4 | 12 px | 14,27 | 49,29 (3,5×) | 15,73 (0,91×) | 16,18 (0,88×) | 16,84 (0,85×) |
| N | strato 3 (Conv) | 8 | 16 px | 16,54 | 57,94 (3,5×) | 19,02 (0,87×) | 20,60 (0,80×) | 21,60 (0,77×) |

Nella colonna 1 thread, fra parentesi, quanto è più lento di 4 thread. Dati: `results/misure.csv` (con temperatura,
frequenza e governor di ogni misura) e `results/tabella.csv`.

## Commento

- **Sì, OpenVINO divide ogni nodo fra i 4 core**, e lo fa bene: con 4 thread le teste vanno 3,2–3,6 volte più
  veloci che con 1 thread, cioè 80–90% di efficienza. Il tiling può recuperare solo quel 10–20% mancante, meno
  il costo dell'halo.
- **Conviene solo sul primo strato, e poco**: lo strato 0 guadagna fino al 14% su P (−0,44 ms) e al 9% su N
  (−0,38 ms). Lì l'halo è di 2 px e la conv, con 3 canali in ingresso, è la meno efficiente da dividere fra i
  core. Su P intero (27,7 ms) è l'1,6%, prima di pagare taglio e ricucitura, che qui non sono misurati.
- **Dallo strato 1 in poi non conviene**: a k = 1 è pari, da k = 2 perde fra il 9% e il 25%. Il C3k2 porta l'halo
  a 12 px e lo strato 3 a 16: con tile da 160 px si ricalcola il 32% di pixel in più a k = 2 (184² contro 160²,
  T = 16) e il 44% a k = 3 (192²), e il guadagno di parallelismo non lo compensa. Più tile ci sono, peggio va.
- **Conclusione**: il parallelismo sull'ingresso non porta niente di utile sul Pi con OpenVINO. Il massimo
  teorico è basso (efficienza già all'80–90%) e l'halo cresce in fretta con la profondità.

## Limiti

- Taglio e ricucitura delle tile non sono misurati (il guadagno reale è più basso di quello in tabella).
- Le tile di bordo, più piccole, sono misurate come quelle interne (stima per eccesso del tempo tiled).
- Lo stream OpenVINO a 1 thread non fissa il core: l'affinità la decide il plugin.
