#!/bin/bash
# Rifà la LUT di questo esperimento con la procedura di src/lut e i parametri di allora (src/lut/pi.py: 13 famiglie,
# passo 2, 71 328 punti × 3 ripetizioni + 6 sui punti instabili, 200 iterazioni, warmup di 60 ms, guardia 75/65 °C):
# misura sul Pi (~40 h con la ventola, riprende da <dir>/results/raw.csv) e analisi -> <dir>/results/lut_<famiglia>.csv.
#   ./rifai.sh [--dry-run] [dir]     dir con results/ e runs/; default questa cartella
#   systemd-run --user --unit=lut-pi-passo2 --working-directory=$PWD bash -c './rifai.sh >> runs/run.log 2>&1'
# Con i risultati già qui non misura niente e rifà solo l'analisi (lut_<famiglia>.csv e summary.json identici).
set -euo pipefail
X=$(cd "$(dirname "$0")" && pwd)
DRY=()
[ "${1:-}" = --dry-run ] && { DRY=(--dry-run); shift; }
LUT_DIR=$(realpath -m "${1:-$X}") STEPS=lut exec "$X/../../../scripts/build_lut.sh" "${DRY[@]}"
