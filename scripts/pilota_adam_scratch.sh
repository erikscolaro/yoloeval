#!/usr/bin/env bash
# Da zero 300 epoche + Pi delle due reti dei piloti Adam (ricerca da 30 e da 100 epoche), in parallelo:
# dice se la ricerca lunga da' un'architettura migliore.
#   systemd-run --user --unit=pilota-scratch --working-directory=$PWD scripts/pilota_adam_scratch.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=runs/catena_notte.log
B="model=yolo26 model.width=0.25 model.depth=D101 strategy=pit_latency target=rpi5_640 logging=local strategy.search.nas.optimizer=Adam strategy.search.nas.lr0=0.0005 strategy.search.regularizer.strength=2 strategy.search.pit.latency.estimator=best strategy.search.workers=8"
uno() { local n=$1 X=$2
  $PY run.py stage=finetune $X finetune=scratch > runs/${n}_scratch.log 2>&1 &&
  flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $X finetune=scratch || break; done" > runs/${n}_pi_scratch.log 2>&1
  echo "[$(date '+%F %T')] $n da zero: FINE (exit $?)" >> $L; }
echo "[$(date '+%F %T')] piloti Adam 30 e 100: da zero 300 epoche" >> $L
uno pilota_adam "$B" & uno pilota_adam100 "$B strategy.search.epochs=100" & wait
