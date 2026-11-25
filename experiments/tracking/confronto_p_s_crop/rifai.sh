#!/usr/bin/env bash
# Confronto P-, S-, Kalman con P- e Kalman con crop sul val (vedi README). Uso: rifai.sh [split=val]
set -euo pipefail
cd "$(dirname "$0")/../../.."
SPLIT=${1:-val}; PY=.venv/bin/python; K=experiments/tracking/taratura_kalman
[ -f $K/runs/detection_$SPLIT.pkl ] && [ -f experiments/tracking/taratura_kalman_crop/results/kalman_crop_f2_$SPLIT.json ] \
  || { echo "prima: taratura_kalman_crop/rifai.sh $SPLIT"; exit 1; }
[ -f $K/runs/detection_s_$SPLIT.pkl ] || $PY $K/detection.py --split $SPLIT --modello S-
$PY experiments/tracking/confronto_p_s_crop/confronto.py --split $SPLIT 2>&1 | grep -v Warning
