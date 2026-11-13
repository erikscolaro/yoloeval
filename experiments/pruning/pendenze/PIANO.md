# Validazione delle pendenze dello stimatore di latenza

Modalità a confronto: `lut`, `smooth`, `secant`, `best` (vedi `reports/stimatore-latenza.md`, "Come la ricerca
usa la pendenza dello stimatore").

Prova: nano D126 → 28 ms a 640. Pavimento N = 22,3 ms (80 % di 28 ms): N supera il test dell'85 %.
Ogni livello ha un `rifai.sh` che lo ripete.

## Livello 0: solo costo di latenza, offline (minuti, PC)

- Ottimizzare solo le maschere con il costo di latenza, senza loss di detection, una volta per modalità.
- Controllare: arriva a 28 ms? oscilla? quanti layer restano appena sopra un gradino?
- Misurare sul Pi le 4 reti ottenute: latenza stimata contro misurata.
- Scopo: filtro economico per trovare bug e modalità che si bloccano.

## Livello 1: ricerca corta (poche epoche, seed 0)

- 4 ricerche con loss di detection completa; cambia solo `estimator`.
- Controllare convergenza al target, stabilità della spinta e mAP prima del fine-tuning.
- Scartare le modalità che oscillano o non arrivano al target.

## Livello 2: esperimento 1 del report (completo)

- Per le modalità rimaste: ricerca completa, fine-tuning e misura sul Pi.
- Metriche: mAP finale, latenza misurata, errore tra stimata e misurata, layer appena sopra un gradino.
- 1 seed. Ripetere con 3 seed solo se due modalità sono vicine.

## Regola di scelta

1. mAP più alta con latenza misurata ≤ 28 ms.
2. A parità: errore tra stimata e misurata più piccolo.
3. A parità: `smooth` (default).

## Dopo

- La modalità vincente va nell'esperimento 2 (stimatore contro MAC).
- Esperimento 3 (spostare i layer al gradino sotto) e 4 (miscela `smooth` → `secant`) solo se servono.
- Durata del livello 2: da stimare dopo il livello 1.
