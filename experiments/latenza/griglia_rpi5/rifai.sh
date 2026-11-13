#!/usr/bin/env bash
# Griglia di latenza del Raspberry Pi 5 (NEW_MASTERPLAN fase 2a, tabella A_budget_dettaglio): larghezze F..M,
# profondità D101/D126/D170, pesi casuali (train=random), OpenVINO fp16, 4 core, freq max, 3 passate.
# Idempotente: train, export e benchmark saltano quello che c'è già in cache; rilanciarlo misura solo il mancante.
#   experiments/latenza/griglia_rpi5/rifai.sh               # DEPLOY di default: 640
#   DEPLOY=640,480p experiments/latenza/griglia_rpi5/rifai.sh
# Le righe 480p, 720p, 1080p, 512², 256², 128² sono le misure del 4/11/2026 (stessi comandi, testa a k=300): con
# DEPLOY che le include l'export le rifà a k=64 (controllo sulla forma dell'uscita), ma il benchmark della cella resta
# quello vecchio. Per rimisurarle a k=64 serve force=true sul benchmark.
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=experiments/latenza/griglia_rpi5
W=${W:-0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1.0}
DEPLOY=${DEPLOY:-640}
I=$PWD/data/air-detection/all/images/calib   # train=random: il dataset serve solo alla forma, basta la calib
A="train=random model=yolo26 model.width=$W model.depth=D101,D126,D170 logging=local dataset.local_path=$I"
M="deploy=$DEPLOY quantization=fp16 backend=openvino hardware=rpi5 freq_target=max compute_target=cpu_4"
mkdir -p $D/runs
.venv/bin/python run.py -m stage=train $A 2>&1 | tee $D/runs/train.log
.venv/bin/python run.py -m stage=export $A $M 2>&1 | tee $D/runs/export.log
.venv/bin/python run.py -m stage=benchmark rep=0,1,2 $A $M 2>&1 | tee $D/runs/benchmark.log
.venv/bin/python tools/figure_sintesi.py reports/budget_latenza_rpi5 --risultati results --board rpi5 --precisioni fp16 --tesi
