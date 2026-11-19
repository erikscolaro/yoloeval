#!/usr/bin/env bash
# Ricerca finale (30,3 ms, ricetta SGD dei piloti: maschere SGD lr 0,01, forza 2, stimatore best) da tre partenze
# meno potate di N-: P+ (w0.125 D170, ~13% da togliere), PN (w0.1875 D126, ~32%), PN+ (w0.1875 D170, ~45%).
# Ognuna: ricerca, da zero 300 epoche, Pi. Le tre in parallelo; il Pi aspetta il suo lock. Diario in runs/catena_notte.log.
#   systemd-run --user --unit=catena-p-pn --working-directory=$PWD scripts/catena_p_pn.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=runs/catena_notte.log
passo() { echo "[$(date '+%F %T')] $*" >> $L; }
pi() { flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $* || break; done"; }
R="strategy=pit_latency target=rpi5_640 logging=local strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=0.01
   strategy.search.regularizer.strength=2 strategy.search.pit.latency.estimator=best"
uno() { local n=$1 X; X=$(echo model=yolo26 $2 $R)   # senza virgolette: toglie l a capo di R, che spezzerebbe bash -c in pi()
  passo "$n: ricerca"
  if $PY run.py stage=search $X > runs/${n}_search.log 2>&1; then
    passo "$n: da zero 300 epoche"
    $PY run.py stage=finetune $X finetune=scratch > runs/${n}_scratch.log 2>&1 &&
      pi "$X finetune=scratch" > runs/${n}_pi_scratch.log 2>&1
    passo "$n: FINE (exit $?)"
  else
    passo "$n: ricerca FALLITA, vedi runs/${n}_search.log"
  fi; }
passo "catena P+, PN, PN+ (30,3 ms, ricetta SGD)"
uno p+ "model.width=0.125 model.depth=D170" &
uno pn "model.width=0.1875 model.depth=D126" &
uno pn+ "model.width=0.1875 model.depth=D170" &
wait
passo "FINE catena P+, PN, PN+"
