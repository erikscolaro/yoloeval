#!/usr/bin/env bash
# Crop che rende di piu' in mAP per classe e fascia (README): P- a crop 256, poi il riassunto.
#   systemd-run --user --unit=crop-map --working-directory=$PWD experiments/tracking/crop_zoom_map/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/tracking/crop_zoom_map; mkdir -p $E/runs
.venv/bin/python $E/run.py artifacts/weights/yolo26w0.125_D101_air_detection_all_e300_e7d00a5f/best.pt --crop 256 > $E/runs/p-_c256.log 2>&1
# S- a crop 128 (24/11/2026): T fino a 96, un oggetto piu' grande non sta nel crop
.venv/bin/python $E/run.py artifacts/weights/yolo26w0.5_D101_air_detection_all_e300_21d2b4bd/best.pt --crop 128 \
  --target 16 24 32 48 64 96 > $E/runs/s-_c128.log 2>&1
