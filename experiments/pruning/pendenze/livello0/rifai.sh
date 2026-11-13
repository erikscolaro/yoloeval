#!/bin/bash
# Livello 0 delle pendenze: ottimizzazione offline delle maschere (PC), poi misura sul Pi e confronto.
#   ./rifai.sh              tutto: search (minuti), measure (Pi), report
#   ./rifai.sh offline      solo search
#   ./rifai.sh pi           aspetta che coda-13nov (e altre misure sul Pi) finisca, poi measure e report;
#                           staccato: systemd-run --user --unit=pendenze-l0 --working-directory=$PWD ./rifai.sh pi
# Variabili: STEPS (default 3000), SGD_LR (default 10). La misura riprende da results/raw.csv.
set -euo pipefail
X=$(cd "$(dirname "$0")" && pwd)
PY=$X/../../../../.venv/bin/python
cd "$X"
mkdir -p runs
case "${1:-tutto}" in
  offline|tutto) "$PY" run.py search 2>&1 | tee runs/search.log ;;&
  pi|tutto)
    while systemctl --user is-active --quiet coda-13nov; do echo "$(date -Is) coda-13nov attiva, attendo"; sleep 60; done
    "$PY" run.py measure && "$PY" run.py report ;;
esac
