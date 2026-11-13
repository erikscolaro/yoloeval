#!/bin/bash
# Reti PIT dentro il fit dello stimatore del Pi 5: genera (CPU), misura sul Pi (riprende da results/raw.csv), fit e confronto.
#   ./rifai.sh [gen|measure|fit]      senza argomenti: tutto. Idempotente: gen salta le traiettorie gia' fatte.
#   systemd-run --user --unit=fit-pit --working-directory=$PWD ./rifai.sh      # staccato; log in runs/
# Variabili: STEPS (default 1500 passi di ottimizzazione per traiettoria), SHARDS (default 3 processi di gen in parallelo).
# Le reti costruite con pesi finti (yaml) del primo tentativo stanno in results_yaml/ e runs/nets_yaml/ (non rifatte).
set -euo pipefail
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)   # gen apre molti file: senza, "Too many open files"
SHARDS=${SHARDS:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}   # gen in parallelo senza saturare la CPU (la ricerca di pendenze/livello1 gira insieme)
X=$(cd "$(dirname "$0")" && pwd)
PY=$X/../../../.venv/bin/python
cd "$X"
mkdir -p runs
for step in ${1:-gen measure l0 fit}; do
  case $step in
    measure|l0) for i in $(seq 1 20); do "$PY" run.py $step && break; echo "$(date -Is) errore, riprovo fra 60 s ($i/20)"; sleep 60; done ;;
    gen) for k in $(seq 0 $((SHARDS-1))); do "$PY" run.py gen $k $SHARDS & done; wait ;;   # shard in parallelo
    *) "$PY" run.py "$step" ;;
  esac 2>&1 | tee -a "runs/$step.log"
done
