#!/usr/bin/env bash
# SCAN a 720p senza cambiare le proporzioni: ingresso 640x384 (720p -> 640x360 + bordo a 384) sul Raspberry Pi 5,
# larghezze F..N x D101/D126/D170, pesi casuali, OpenVINO fp16, 4 core, governor performance, 3 passate.
# Poi rimisura N a 640x640 (27,9 ms sospetto): i json vecchi vanno prima in results/vecchi_640/.
#   experiments/latenza/scan_modelli_640x384/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=experiments/latenza/scan_modelli_640x384
W=${W:-0.0625,0.09375,0.125,0.1875,0.25}
I=$PWD/data/air-detection/all/images/calib   # train=random: il dataset serve solo alla forma
A="train=random model=yolo26 model.depth=D101,D126,D170 logging=local dataset.local_path=$I"
M="quantization=fp16 backend=openvino hardware=rpi5 freq_target=max compute_target=cpu_4"
mkdir -p $D/runs
.venv/bin/python run.py -m stage=train model.width=$W $A 2>&1 | tee $D/runs/train.log
.venv/bin/python run.py -m stage=export model.width=$W deploy=640x384 $A $M 2>&1 | tee $D/runs/export.log
flock runs/.pi.lock .venv/bin/python run.py -m stage=benchmark rep=0,1,2 model.width=$W deploy=640x384 $A $M \
  2>&1 | tee $D/runs/benchmark.log
flock runs/.pi.lock .venv/bin/python run.py -m stage=benchmark rep=0,1,2 force=true model.width=0.25 deploy="'640'" \
  $A $M 2>&1 | tee $D/runs/benchmark_n640.log
