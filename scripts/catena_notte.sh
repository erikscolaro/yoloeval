#!/usr/bin/env bash
# Catena del 14/11, senza intervento: chiude il livello 1 delle pendenze, poi la ricerca finale. Diario in
# runs/catena_notte.log. Ogni passo e' in cache se gia' fatto: si puo' rilanciare.
#   1. finetuning corto (10 epoche) e Pi di best e lut: prima occhiata
#   2. finetuning intero (100 epoche) e Pi delle 3 reti, in parallelo: a 10 epoche la mAP sale ancora, non basta a scegliere
#   3. secondo seed (43) delle due migliori: ricerca, finetuning intero, Pi
#   4. scelta (scegli.py, TAG=ft100), select, ricerca finale a 30,3 ms con la pendenza vincente
#   5. rete finale: finetuning 100 epoche e da zero 300 epoche in parallelo, Pi
#   6. da zero 300 epoche delle 3 reti del livello 1, in parallelo, Pi
#   systemd-run --user --unit=catena-notte --working-directory=$PWD scripts/catena_notte.sh
set -uo pipefail
cd "$(dirname "$0")/.."
D=experiments/pruning/pendenze/livello1; PY=.venv/bin/python; L=runs/catena_notte.log
passo() { echo "[$(date '+%F %T')] $*" >> $L; }
pi() { flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $* || break; done"; }
insieme() { local t=$1 f=$2; shift 2; for e in "$@"; do TAG=$t FT="$f" $D/dopo.sh $e & done; wait; }   # una rete per processo

passo "1. finetuning corto e Pi di best"; $D/dopo.sh best
while systemctl --user is-active -q pendenze-l1; do sleep 30; done
passo "1. finetuning corto e Pi di lut"; $D/dopo.sh lut
$PY $D/scegli.py >> $L 2>&1
passo "2. finetuning 100 epoche e Pi di smooth, best, lut"; insieme ft100 "" smooth best lut
TAG=ft100 $PY $D/scegli.py >> $L 2>&1
T=$(TAG=ft100 $PY $D/scegli.py top2); passo "3. secondo seed (43) per: $T"
OPTS=sgd LR=0.01 FORZA=2 SEED=43 STIME="$T" bash $D/rifai.sh >> $L 2>&1
export SEED=43; insieme ft100 "" $T; unset SEED
TAG=ft100 $PY $D/scegli.py >> $L 2>&1
W=$(TAG=ft100 $PY $D/scegli.py vincitore); passo "4. pendenza vincente: $W"

$PY run.py stage=select target=rpi5_640 logging=local > runs/notte_select.log 2>&1 || passo "select FALLITA"
MS=$($PY -m src.selection results/select/rpi5_640.json); passo "modello di partenza: $MS"
X="$MS strategy=pit_latency target=rpi5_640 logging=local strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=0.01
   strategy.search.regularizer.strength=2 strategy.search.pit.latency.estimator=$W"
X=$(echo $X)
passo "4. ricerca finale"
if $PY run.py stage=search $X > runs/notte_search.log 2>&1; then
  passo "5. finetuning 100 epoche e da zero 300 epoche della rete finale"
  ( $PY run.py stage=finetune $X > runs/notte_ft100.log 2>&1 && pi "$X" > runs/notte_pi_ft100.log 2>&1 ) &
  ( $PY run.py stage=finetune $X finetune=scratch > runs/notte_scratch.log 2>&1 && pi "$X finetune=scratch" > runs/notte_pi_scratch.log 2>&1 ) &
  wait
else
  passo "ricerca finale FALLITA: vedi runs/notte_search.log"
fi
passo "6. da zero 300 epoche delle 3 reti del livello 1"; insieme scratch "finetune=scratch" smooth best lut
passo FINE
