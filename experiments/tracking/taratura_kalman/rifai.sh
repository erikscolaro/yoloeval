#!/usr/bin/env bash
# Taratura del Kalman, dall'inizio alla fine, sul dataset in data/tracking/<split> (vedi README.md).
#   bash experiments/tracking/taratura_kalman/rifai.sh [split=val] [prove=1000]
# Ogni fase scrive in results/ (piccolo, in git) o runs/ (grande, fuori da git). Una fase gia' fatta si salta:
# per rifarla, cancellare il suo file. Optuna riprende dalle prove gia' nel database.
set -euo pipefail
cd "$(dirname "$0")/../../.."
SPLIT=${1:-val}; PROVE=${2:-1000}
D=experiments/tracking/taratura_kalman; PY=.venv/bin/python

echo "== 1/4 detection di P- su ogni fotogramma (GPU, una volta sola)"
[ -f $D/runs/detection_$SPLIT.pkl ] || $PY $D/detection.py --split $SPLIT

echo "== 2/4 R e Q dai dati"
[ -f $D/results/rq_$SPLIT.json ] || $PY $D/rq.py --split $SPLIT

echo "== 3/4 Optuna sul ciclo di vita del track ($PROVE prove per studio, tutti i core)"
$PY $D/optuna_kalman.py --split $SPLIT --trials $PROVE | tee $D/results/kalman_f1_$SPLIT.txt

echo "== 3b/4 Optuna con l'obiettivo della torretta: F2 sui droni + vincolo su aerei ed elicotteri"
$PY $D/optuna_kalman_f2.py --split $SPLIT --trials $PROVE | tee $D/results/kalman_f2_$SPLIT.txt

echo "== 4/4 valutazione sui video con i parametri trovati: detector (alla sua conf migliore), Kalman, Kalman con un"
echo "   crop (GPU, studi in parallelo)"
for S in svanstrom anti_uav tutte; do
  J=$D/results/kalman_f1_${S}_$SPLIT.json
  P=$($PY -c "import json; p=json.load(open('$J'))['parametri']; p.pop('fps'); print(' '.join(f'{k}={v}' for k, v in p.items()))")
  C=$($PY -c "import json; print(json.load(open('$J'))['detector_miglior_conf']['conf'])")   # detector alla sua soglia migliore
  $PY experiments/tracking/valuta_tracker_video/run.py --split $SPLIT --tag tarato_$S --p $P --det-conf $C \
    --ct max_crops=1 > $D/runs/valuta_${S}_$SPLIT.log 2>&1 &
done
wait
for S in svanstrom anti_uav tutte; do echo "-- parametri dello studio $S"; sed -n "/^fonte/,\$p" $D/runs/valuta_${S}_$SPLIT.log; done \
  | tee $D/results/valuta_$SPLIT.txt
