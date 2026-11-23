#!/usr/bin/env bash
# Latenza a batch 1, 2, 4, 8 a 256x256 sul Raspberry Pi 5 per i 27 modelli della griglia (larghezze F..M x D101/D126/
# D170), pesi casuali, OpenVINO fp16, 4 core, freq_target=max (governor performance), 3 passate. Il batch e' quello
# del crop tracker (src/crop_tracker.py): N crop in una sola inferenza. Stessa pipeline di ../modelli_latenza_per_ingresso/rifai.sh,
# deploy crop256_b1..b8 (conf/deploy). Idempotente; il benchmark tiene il lock del Pi. Poi le figure in report/.
#   experiments/latenza/track_modelli_batch_crop256/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=experiments/latenza/track_modelli_batch_crop256
W=${W:-0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1.0}
DEPLOY=${DEPLOY:-crop256_b1,crop256_b2,crop256_b4,crop256_b8}
I=$PWD/data/air-detection/all/images/calib   # train=random: il dataset serve solo alla forma, basta la calib
A="train=random model=yolo26 model.width=$W model.depth=D101,D126,D170 logging=local dataset.local_path=$I"
M="deploy=$DEPLOY quantization=fp16 backend=openvino hardware=rpi5 freq_target=max compute_target=cpu_4"
mkdir -p $D/runs
.venv/bin/python run.py -m stage=train $A 2>&1 | tee $D/runs/train.log
.venv/bin/python run.py -m stage=export $A $M 2>&1 | tee $D/runs/export.log
flock runs/.pi.lock .venv/bin/python run.py -m stage=benchmark rep=0,1,2 $A $M 2>&1 | tee $D/runs/benchmark.log
.venv/bin/python $D/figure.py
