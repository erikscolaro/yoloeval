#!/bin/bash
# Separabilità della latenza sul Pi 5: misura completa (circa 2,7 h, riprende da results/raw.csv), analisi e figure.
#   bash experiments/pruning/separability/rifai.sh [run|analyze]      senza argomenti: tutti e due
# Il pilota (ITERS=200 python run.py pilot 20) e il log su wandb (log_wandb.py) restano a mano, vedi README.
set -euo pipefail
X=$(cd "$(dirname "$0")" && pwd); R=$X/../../..
PY=$R/.venv/bin/python
cd "$X"
for s in ${1:-run analyze}; do
  case $s in
    run) ITERS=${ITERS:-200} flock "$R/runs/.pi.lock" "$PY" run.py run ;;
    analyze) "$PY" analyze.py && "$PY" sizes.py ;;
  esac
done
