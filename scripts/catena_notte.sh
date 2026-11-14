#!/usr/bin/env bash
# Catena del 14/11, senza intervento: chiude il livello 1 delle pendenze (finetuning corto e Pi di best e lut, secondo
# seed delle due migliori, scelta), poi select, ricerca finale con la pendenza vincente, finetuning 100 epoche e da zero
# 300 epoche in parallelo, misure sul Pi; infine finetuning e da zero delle 3 reti del livello 1. Diario in
# runs/catena_notte.log. Ogni passo e' in cache se gia' fatto: si puo' rilanciare.
#   systemd-run --user --unit=catena-notte --working-directory=$PWD scripts/catena_notte.sh
set -uo pipefail
cd "$(dirname "$0")/.."
D=experiments/pruning/pendenze/livello1; PY=.venv/bin/python; L=runs/catena_notte.log
passo() { echo "[$(date '+%F %T')] $*" >> $L; }
pi() { flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $* || break; done"; }

passo "finetuning corto e Pi di best"; $D/dopo.sh best
while systemctl --user is-active -q pendenze-l1; do sleep 30; done
passo "finetuning corto e Pi di lut"; $D/dopo.sh lut
$PY $D/scegli.py >> $L 2>&1
T=$($PY $D/scegli.py top2); passo "secondo seed (43) per: $T"
OPTS=sgd LR=0.01 FORZA=2 SEED=43 STIME="$T" bash $D/rifai.sh >> $L 2>&1
SEED=43 $D/dopo.sh $T
$PY $D/scegli.py >> $L 2>&1
W=$($PY $D/scegli.py vincitore); passo "pendenza vincente: $W"

$PY run.py stage=select target=rpi5_640 logging=local > runs/notte_select.log 2>&1 || passo "select FALLITA"
MS=$($PY -m src.selection results/select/rpi5_640.json); passo "modello di partenza: $MS"
X="$MS strategy=pit_latency target=rpi5_640 logging=local strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=0.01
   strategy.search.regularizer.strength=2 strategy.search.pit.latency.estimator=$W"
X=$(echo $X)
passo "ricerca finale"
if $PY run.py stage=search $X > runs/notte_search.log 2>&1; then
  passo "finetuning 100 epoche e da zero 300 epoche della rete finale"
  ( $PY run.py stage=finetune $X > runs/notte_ft100.log 2>&1 && pi "$X" > runs/notte_pi_ft100.log 2>&1 ) &
  ( $PY run.py stage=finetune $X finetune=scratch > runs/notte_scratch.log 2>&1 && pi "$X finetune=scratch" > runs/notte_pi_scratch.log 2>&1 ) &
  wait
else
  passo "ricerca finale FALLITA: vedi runs/notte_search.log"
fi
for e in smooth best lut; do
  passo "livello 1, $e: finetuning 100 epoche e da zero 300 epoche"
  TAG=ft100 FT="" $D/dopo.sh $e & TAG=scratch FT="finetune=scratch" $D/dopo.sh $e & wait
done
passo FINE
