#!/usr/bin/env bash
# mAP sul Pi (eval.map_on_target) per le reti della griglia nel budget a 640, con i pesi allenati 300 epoche:
# serve per confrontare griglia e potate alla pari nel report (P- e N- ce l'hanno già). OpenVINO fp16.
#   systemd-run --user --unit=map-pi-griglia --working-directory=$PWD scripts/map_pi_griglia.sh
#   tail -f runs/map_pi_griglia.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=runs/map_pi_griglia.log
RETI=("F- model=yolo26 model.width=0.0625 model.depth=D101" "FP- model=yolo26 model.width=0.09375 model.depth=D101"
      "F model=yolo26 model.width=0.0625 model.depth=D126" "FP model=yolo26 model.width=0.09375 model.depth=D126"
      "P model=yolo26p"
      "F+ model=yolo26 model.width=0.0625 model.depth=D170" "FP+ model=yolo26 model.width=0.09375 model.depth=D170")
for r in "${RETI[@]}"; do read -r n m <<< "$r"
  X="$m target=rpi5_640 logging=local eval.map_on_target=true"
  echo "[$(date '+%F %T')] $n" >> $L
  flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $X || exit 1; done" >> runs/map_pi_griglia_$n.log 2>&1
  echo "[$(date '+%F %T')] $n: FINE (exit $?)" >> $L
done
