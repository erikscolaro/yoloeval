#!/bin/bash
# Rifà B4-R con la procedura di src/lut e i parametri di allora (src/lut/pi.py: 15 architetture fino al nano ×
# (50 potate + l'intera) = 765 reti, n = 2, 3 ripetizioni da 50 iterazioni) sulla LUT di ../lut_pi_passo2:
# generazione (saltata se c'è <dir>/results/nets.csv), misura sul Pi (riprende da results/raw.csv), fit -> fit.json.
#   ./rifai.sh [--dry-run] [dir]     dir con results/ e runs/; default questa cartella
# Con i risultati già qui non misura niente e rifà solo il fit (fit.json e predictions.csv identici).
set -euo pipefail
X=$(cd "$(dirname "$0")" && pwd)
DRY=()
[ "${1:-}" = --dry-run ] && { DRY=(--dry-run); shift; }
LUT_DIR=$X/../lut_pi_passo2 B4R_DIR=$(realpath -m "${1:-$X}") STEPS=b4r exec "$X/../../../scripts/build_lut.sh" "${DRY[@]}"
