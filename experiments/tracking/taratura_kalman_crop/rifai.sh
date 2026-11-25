#!/usr/bin/env bash
# Taratura del tracker con crop (SCAN con P-, un crop 256 con S-), dall'inizio alla fine (vedi README.md).
#   bash experiments/tracking/taratura_kalman_crop/rifai.sh [split=val] [prove=1000] [processi=12]
# Usa le detection di P- e R e Q di ../taratura_kalman: se mancano, le fa. Una fase gia' fatta si salta: per rifarla,
# cancellare il suo file. Optuna riprende dalle prove gia' nel database.
set -euo pipefail
cd "$(dirname "$0")/../../.."
SPLIT=${1:-val}; PROVE=${2:-1000}; PROC=${3:-12}
D=experiments/tracking/taratura_kalman_crop; K=experiments/tracking/taratura_kalman; PY=.venv/bin/python

echo "== 0/3 dati di ../taratura_kalman: detection di P- e R, Q (e i parametri del Kalman da solo, prima prova)"
[ -f $K/runs/detection_$SPLIT.pkl ] || $PY $K/detection.py --split $SPLIT
[ -f $K/results/rq_$SPLIT.json ] || $PY $K/rq.py --split $SPLIT
[ -f $K/results/kalman_f2_tutte_$SPLIT.json ] || $PY $K/optuna_kalman_f2.py --split $SPLIT --studi tutte --trials 1000

echo "== 1/3 R e confidenza di S- sui ritagli"
[ -f $D/results/rs_$SPLIT.json ] || $PY $D/rs.py --split $SPLIT

echo "== 2/3 Optuna sul tracker con crop ($PROVE prove, $PROC processi sulla GPU)"
$PY $D/optuna_crop.py --split $SPLIT --trials $PROVE --proc $PROC | tee $D/results/kalman_crop_f2_$SPLIT.txt

echo "== 3/3 valutazione sui video con i parametri trovati: detector, Kalman, Kalman con un crop"
$PY - "$D/results/kalman_crop_f2_$SPLIT.json" > $D/runs/argomenti_$SPLIT.txt <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); b, f = r["migliore"]["parametri"], r["fissi"]
kal = {k: b[k] for k in ("hi", "lo", "min_hits", "max_age", "show_age", "gate", "v0", "iou_dup", "vote")}
kal |= {k: f["rq"][k] for k in ("q_pos", "q_size", "r_pos", "r_size")}
tr = {"hi": b["hi_track"], "lo": b["lo_track"], "r_pos": f["rs"]["r_pos"], "r_size": f["rs"]["r_size"]}
ct = {k: b[k] for k in ("k", "ctx", "fill", "rescan", "max_side")} | {"max_crops": 1}
print("--p " + " ".join(f"{k}={v}" for k, v in kal.items()) + " --ct " + " ".join(f"{k}={v}" for k, v in ct.items())
      + " --track-p " + " ".join(f"{k}={v}" for k, v in tr.items()) + f" --det-conf {r['detector']['conf']}")
PY
$PY experiments/tracking/valuta_tracker_video/run.py --split $SPLIT --tag tarato_crop $(cat $D/runs/argomenti_$SPLIT.txt) \
  2>&1 | sed -n "/^fonte/,\$p" | tee $D/results/valuta_$SPLIT.txt
