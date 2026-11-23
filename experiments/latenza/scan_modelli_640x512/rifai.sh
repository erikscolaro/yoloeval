#!/usr/bin/env bash
# SCAN con camera Svanstrom nativa: ingresso 640x512 sul Raspberry Pi 5, P- e PN- (larghezze 0.125 e 0.1875, D101),
# pesi casuali, OpenVINO fp16, 4 core, governor performance, 3 passate.
#   experiments/latenza/scan_modelli_640x512/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=experiments/latenza/scan_modelli_640x512
W=${W:-0.125,0.1875}
I=$PWD/data/air-detection/all/images/calib   # train=random: il dataset serve solo alla forma
A="train=random model=yolo26 model.depth=D101 logging=local dataset.local_path=$I"
M="quantization=fp16 backend=openvino hardware=rpi5 freq_target=max compute_target=cpu_4"
mkdir -p $D/runs
.venv/bin/python run.py -m stage=train model.width=$W $A 2>&1 | tee $D/runs/train.log
.venv/bin/python run.py -m stage=export model.width=$W deploy=640x512 $A $M 2>&1 | tee $D/runs/export.log
flock runs/.pi.lock .venv/bin/python run.py -m stage=benchmark rep=0,1,2 model.width=$W deploy=640x512 $A $M \
  2>&1 | tee $D/runs/benchmark.log
