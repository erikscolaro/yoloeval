#!/usr/bin/env bash
# INTER_AREA contro INTER_LINEAR con P- (README): scan sui video Anti-UAV 1080p di val, poi i crop come ../crop_zoom_map
# con INTER_LINEAR. GPU, circa 30-40 minuti; log in runs/, json in results/.
#   systemd-run --user --unit=interp-map --working-directory=$PWD experiments/tracking/crop_interpolazione_map/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/tracking/crop_interpolazione_map; mkdir -p $E/runs
M=artifacts/weights/yolo26w0.125_D101_air_detection_all_e300_e7d00a5f/best.pt
.venv/bin/python $E/run.py $M scan > $E/runs/scan.log 2>&1
.venv/bin/python $E/run.py $M crop > $E/runs/crop.log 2>&1
