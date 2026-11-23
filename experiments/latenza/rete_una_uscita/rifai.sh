#!/usr/bin/env bash
# Neck e testa a una sola scala: 3 reti lineari (backbone -> Detect su P3, P4 o P5) e il nano (N, D126), pesi
# casuali, 640x640, 4 classi. Parametri e MAC (ONNX fp32) e latenza sul Pi 5 (OpenVINO fp16, 4 core, governor
# performance, 3 passate), con il lock del Pi.
#   experiments/latenza/rete_una_uscita/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=experiments/latenza/rete_una_uscita
mkdir -p $D/results $D/runs
.venv/bin/python $D/run.py build 2>&1 | tee $D/runs/build.log
flock runs/.pi.lock .venv/bin/python $D/run.py measure 2>&1 | tee $D/runs/measure.log
.venv/bin/python $D/run.py table
