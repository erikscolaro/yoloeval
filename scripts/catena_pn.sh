#!/usr/bin/env bash
# Dopo catena_notte.sh: stessa ricerca finale (30,3 ms, pendenza vincente) ma da PN- (w0.1875 D101) invece che da N-.
# Motivo: il pavimento. N- ha 21,6 ms non potabili su 30,3; PN- ne ha 18,0 e lascia piu' latenza agli strati potabili.
# Poi finetuning 100 epoche e da zero 300 epoche in parallelo, Pi. Diario in runs/catena_notte.log.
#   systemd-run --user --unit=catena-pn --working-directory=$PWD scripts/catena_pn.sh
set -uo pipefail
cd "$(dirname "$0")/.."
D=experiments/pruning/pendenze/livello1; PY=.venv/bin/python; L=runs/catena_notte.log
passo() { echo "[$(date '+%F %T')] $*" >> $L; }
pi() { flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $* || break; done"; }
while systemctl --user is-active -q catena-notte4; do sleep 120; done      # GPU e Pi liberi
W=$(TAG=scratch $PY $D/scegli.py vincitore)
X="model=yolo26 model.width=0.1875 model.depth=D101 strategy=pit_latency target=rpi5_640 logging=local
   strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=0.01 strategy.search.regularizer.strength=2
   strategy.search.pit.latency.estimator=$W"
X=$(echo $X)
passo "7. ricerca da PN-, pendenza $W"
if $PY run.py stage=search $X > runs/pn_search.log 2>&1; then
  passo "8. finetuning 100 epoche e da zero 300 epoche della rete da PN-"
  ( $PY run.py stage=finetune $X > runs/pn_ft100.log 2>&1 && pi "$X" > runs/pn_pi_ft100.log 2>&1 ) &
  ( $PY run.py stage=finetune $X finetune=scratch > runs/pn_scratch.log 2>&1 && pi "$X finetune=scratch" > runs/pn_pi_scratch.log 2>&1 ) &
  wait
else
  passo "ricerca da PN- FALLITA: vedi runs/pn_search.log"
fi
passo "FINE PN-"
