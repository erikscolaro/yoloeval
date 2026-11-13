#!/usr/bin/env bash
# 13/11: con AdamW le pendenze contano poco (AdamW normalizza il gradiente di ogni maschera, livello 0) e la ricerca
# lut_adamw oscilla (mAP 0,01-0,42) e scende piano. Si tiene lut_adamw come riferimento, poi le 4 pendenze con SGD
# una alla volta (due in parallelo: stessa resa, GPU a 46/49 GB).
set -uo pipefail
cd "$(dirname "$0")/.."
L=experiments/pruning/pendenze/livello1/runs
until [[ -f $L/smooth_adamw.log ]] || ! systemctl --user is-active -q pendenze-l1; do sleep 30; done
systemctl --user stop pendenze-l1
OPTS=sgd bash experiments/pruning/pendenze/livello1/rifai.sh
