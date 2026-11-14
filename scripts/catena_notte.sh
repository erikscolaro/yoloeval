#!/usr/bin/env bash
# Catena del 14/11, senza intervento, dal passo 3 (i passi 1-2, finetuning corto e intero di smooth e best, sono fatti;
# lut e' esclusa: si ferma a 31,9 ms, target mancato). Diario in runs/catena_notte.log. Ogni passo e' in cache se gia' fatto.
#   3. secondo seed (43) di smooth e best: ricerca
#   4. da zero 300 epoche e Pi delle 4 reti (2 pendenze x 2 seed), a coppie: e' il protocollo della scelta, misura
#      l'architettura senza i pesi usciti dalla ricerca
#   5. scelta (scegli.py, TAG=scratch), select, ricerca finale a 30,3 ms con la pendenza vincente
#   6. rete finale: finetuning 100 epoche e da zero 300 epoche in parallelo, Pi
#   systemd-run --user --unit=catena-notte --working-directory=$PWD scripts/catena_notte.sh
set -uo pipefail
cd "$(dirname "$0")/.."
D=experiments/pruning/pendenze/livello1; PY=.venv/bin/python; L=runs/catena_notte.log
passo() { echo "[$(date '+%F %T')] $*" >> $L; }
pi() { flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $* || break; done"; }
insieme() { local t=$1 f=$2; shift 2; for e in "$@"; do TAG=$t FT="$f" $D/dopo.sh $e & done; wait; }   # una rete per processo

TAG=ft100 $PY $D/scegli.py >> $L 2>&1
passo "3. secondo seed (43): ricerca di smooth e best"
OPTS=sgd LR=0.01 FORZA=2 SEED=43 STIME="smooth best" bash $D/rifai.sh >> $L 2>&1
passo "4. da zero 300 epoche e Pi, seed 42: smooth e best"; insieme scratch "finetune=scratch" smooth best
TAG=scratch $PY $D/scegli.py >> $L 2>&1
passo "4. da zero 300 epoche e Pi, seed 43: smooth e best"; export SEED=43; insieme scratch "finetune=scratch" smooth best; unset SEED
TAG=scratch $PY $D/scegli.py >> $L 2>&1
W=$(TAG=scratch $PY $D/scegli.py vincitore); passo "5. pendenza vincente: $W"

$PY run.py stage=select target=rpi5_640 logging=local > runs/notte_select.log 2>&1 || passo "select FALLITA"
MS=$($PY -m src.selection results/select/rpi5_640.json); passo "modello di partenza: $MS"
X="$MS strategy=pit_latency target=rpi5_640 logging=local strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=0.01
   strategy.search.regularizer.strength=2 strategy.search.pit.latency.estimator=$W"
X=$(echo $X)
passo "5. ricerca finale"
if $PY run.py stage=search $X > runs/notte_search.log 2>&1; then
  passo "6. finetuning 100 epoche e da zero 300 epoche della rete finale"
  ( $PY run.py stage=finetune $X > runs/notte_ft100.log 2>&1 && pi "$X" > runs/notte_pi_ft100.log 2>&1 ) &
  ( $PY run.py stage=finetune $X finetune=scratch > runs/notte_scratch.log 2>&1 && pi "$X finetune=scratch" > runs/notte_pi_scratch.log 2>&1 ) &
  wait
else
  passo "ricerca finale FALLITA: vedi runs/notte_search.log"
fi
passo FINE
