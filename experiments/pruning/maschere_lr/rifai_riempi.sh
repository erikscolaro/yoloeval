#!/usr/bin/env bash
# Strada 3: la rete di rifai.sh (ultima epoca) con il budget riempito da riempi.py fino a 30,3 ms stimati, poi da zero
# 300 epoche (tutti i checkpoint) e misura sul Pi. La "ricerca" riempita ha la chiave della config con
# +strategy.search.riempi=30.3 (runs/X_riempi.txt), cosi' finetune, export e benchmark la trovano da soli.
#   systemd-run --user --unit=maschere-riempi --working-directory=$PWD experiments/pruning/maschere_lr/rifai_riempi.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=${M:-$HOME/Desktop/yoloeval}; PY=$M/.venv/bin/python; E=experiments/pruning/maschere_lr; L=$E/runs/diario.log
export PYTHONPATH=${YOLOPIT:-$HOME/Desktop/yolopit}
S=$M/artifacts/search/yolo26w0.1875_D101_air_detection_all_pit_latency
X=$(cat $E/runs/X_riempi.txt)
passo() { echo "[$(date '+%F %T')] $*" | tee -a $L; }
[ -f ${S}_8114da10/pruned.pt ] || $PY $E/riempi.py ${S}_c0d7cb3f epoch29.pt 30.3 ${S}_8114da10 8114da10 > $E/runs/riempi.log 2>&1 \
  || { passo "riempi FALLITO"; exit 1; }
passo "riempita: da zero 300 epoche"
$PY run.py stage=finetune $X finetune=scratch +finetune.save_period=1 > $E/runs/scratch_riempi.log 2>&1 || { passo "da zero FALLITO"; exit 1; }
passo "riempita: Pi"
flock $M/runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $X finetune=scratch +finetune.save_period=1 results_dir=$PWD/$E/results || break; done" > $E/runs/pi_riempi.log 2>&1
passo "riempita: FINE (exit $?)"
