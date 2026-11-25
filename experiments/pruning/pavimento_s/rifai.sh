#!/bin/bash
# Pavimento della S (e del nano) sul Pi 5: costruisce le reti (CPU) e le misura sul Pi, con il lock del Pi.
#   bash experiments/pruning/pavimento_s/rifai.sh [build|measure]      senza argomenti: tutti e due
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd); cd "$R"
PY=$R/.venv/bin/python
for s in ${1:-build measure}; do
  case $s in
    measure) flock runs/.pi.lock "$PY" experiments/pruning/pavimento_s/run.py measure ;;
    *) "$PY" experiments/pruning/pavimento_s/run.py "$s" ;;
  esac
done
