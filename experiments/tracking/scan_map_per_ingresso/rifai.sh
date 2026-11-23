#!/usr/bin/env bash
# mAP dello SCAN sul val con ingresso nativo 640x512, schiacciato a 640x384 e letterbox a 640x384 (run.py). GPU.
# Modelli: P- e PN- (candidati dello SCAN, 23/11/2026). Risultati in results/map_<pesi>.json.
#   experiments/tracking/scan_map_per_ingresso/rifai.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p experiments/tracking/scan_map_per_ingresso/runs
for m in p-:yolo26w0.125_D101_air_detection_all_e300_e7d00a5f pn-:yolo26w0.1875_D101_air_detection_all_e300_a033e56f; do
  [[ -n ${SOLO:-} && ${m%%:*} != $SOLO ]] && continue
  .venv/bin/python experiments/tracking/scan_map_per_ingresso/run.py artifacts/weights/${m#*:}/best.pt \
    2>&1 | tee experiments/tracking/scan_map_per_ingresso/runs/${m%%:*}.log
done
