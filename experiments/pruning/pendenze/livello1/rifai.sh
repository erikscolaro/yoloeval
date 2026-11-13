#!/usr/bin/env bash
# Livello 1 della validazione delle pendenze (PIANO.md): ricerca corta con la loss di detection sul nano D126, target
# 28 ms (budget del modello), 30 epoche, per le pendenze e 2 ottimizzatori delle maschere:
#   adamw: quello di pit_latency (AdamW lr 0.01): normalizza la spinta, le pendenze contano poco (livello 0)
#   sgd:   SGD lr 0.05 con il momento 0.9 di yolopit: conta la forza. NON lr 1.0: un canale e' spento solo se
#          |maschera| < 0.5, e con passi di qualche unita' la maschera salta la finestra e resta accesa (13/11:
#          maschere a -26, +10; la latenza si fermava a 0,92 con forza 1 e con forza 10)
# STIME="lut smooth best" le pendenze, LR il passo di SGD, FORZA il moltiplicatore di DUCCIO (vuoto = 1).
# Una ricerca alla volta (batch 64, conf/strategy; 4 in parallelo dividevano la GPU: piu' di 17 min per epoca ciascuna). Idempotente: le ricerche
# in cache si saltano; una interrotta riparte da capo nella stessa cartella.
#   experiments/pruning/pendenze/livello1/rifai.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
D=experiments/pruning/pendenze/livello1
C="stage=search model=yolo26n strategy=pit_latency target=rpi5_640 target.model_budget_ms=28 logging=local
   strategy.search.epochs=30"
for opt in ${OPTS:-adamw}; do   # OPTS="adamw sgd" per anche SGD
  O=""; [[ $opt == sgd ]] && O="strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=${LR:-0.05}"
  [[ -n ${FORZA:-} ]] && O="$O strategy.search.regularizer.strength=$FORZA"
  for e in ${STIME:-lut smooth best}; do
    .venv/bin/python run.py $C $O strategy.search.pit.latency.estimator=$e > $D/runs/${e}_$opt.log 2>&1
  done
done
grep -h "ricerca\|latenza\|Error" $D/runs/*.log | tail -20
