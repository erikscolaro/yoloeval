#!/usr/bin/env bash
# Livello 1 della validazione delle pendenze (PIANO.md): ricerca corta con la loss di detection sul nano D126, target
# 28 ms (budget del modello), 10 epoche, per le 4 pendenze e 2 ottimizzatori delle maschere:
#   adamw: quello di pit_latency (AdamW lr 0.01): normalizza la spinta, le pendenze contano poco (livello 0)
#   sgd:   SGD lr 1.0 con il momento 0.9 di yolopit (passo effettivo ~10, quello del livello 0): conta la forza
# Due ondate da 4 ricerche in parallelo sulla GPU (workers 6 ciascuna). Idempotente: le ricerche in cache si saltano.
#   experiments/pruning/pendenze/livello1/rifai.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
D=experiments/pruning/pendenze/livello1
C="stage=search model=yolo26n strategy=pit_latency target=rpi5_640 target.model_budget_ms=28 logging=local
   strategy.search.epochs=10 strategy.search.workers=6"
for opt in adamw sgd; do
  O=""; [[ $opt == sgd ]] && O="strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=1.0"
  for e in lut smooth secant best; do
    .venv/bin/python run.py $C $O strategy.search.pit.latency.estimator=$e > $D/runs/${e}_$opt.log 2>&1 &
    sleep 20   # sfasa l'avvio: setup e cache del dataset
  done
  wait
done
grep -h "ricerca\|latenza\|Error" $D/runs/*.log | tail -20
