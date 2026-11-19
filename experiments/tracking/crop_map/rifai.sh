#!/usr/bin/env bash
# Crop che rende di piu' in mAP per classe e fascia (README): P- a crop 256, poi il riassunto.
#   systemd-run --user --unit=crop-map --working-directory=$PWD experiments/tracking/crop_map/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/tracking/crop_map; mkdir -p $E/runs
.venv/bin/python $E/run.py artifacts/weights/yolo26w0.125_D101_air_detection_all_e300_e7d00a5f/best.pt --crop 256 > $E/runs/p-_c256.log 2>&1
