#!/usr/bin/env bash
# int8 contro fp16 sul Pi, sui modelli della griglia nuova: P- e N-, 4 varianti ciascuno (export + misura + mAP sul Pi).
# Stampa una riga di avanzamento per passo (barra, percentuale, tempo rimanente) e alla fine la tabella.
#   systemd-run --user --unit=int8-minimo --working-directory=$PWD scripts/int8_minimo.sh
#   tail -f runs/int8_minimo/avanzamento.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=runs/int8_minimo; mkdir -p $D; A=$D/avanzamento.log; T=$D/tabella.tsv
MODELLI=("P- model.width=0.125" "N- model.width=0.25")
VARIANTI=("fp16_openvino quantization=fp16 backend=openvino" "int8_ort quantization=int8 backend=onnxruntime"
          "int8full_ort quantization=int8_full backend=onnxruntime" "int8_openvino quantization=int8 backend=openvino")
TOT=$(( ${#MODELLI[@]} * ${#VARIANTI[@]} * 2 )); n=0; t0=$(date +%s)
: > $A; printf "modello\tvariante\tlatenza_ms\tmAP50-95\n" > $T

riga() {  # barra, percentuale, tempo rimanente stimato dalla media dei passi fatti
  local fatti=$1 cosa=$2 w=30 p=$(( $1 * 100 / TOT )) el=$(( $(date +%s) - t0 )) eta="--:--"
  (( fatti > 0 )) && { local r=$(( el * (TOT - fatti) / fatti )); eta=$(printf "%02d:%02d" $((r/60)) $((r%60))); }
  local pieni=$(( fatti * w / TOT )); local b; b=$(printf "%${pieni}s" | tr ' ' '#')$(printf "%$((w-pieni))s" | tr ' ' '.')
  printf "[%s] %3d%%  %2d/%d  rimanente %s  | %s\n" "$b" $p $fatti $TOT "$eta" "$cosa" >> $A
}

riga 0 "inizio"
for m in "${MODELLI[@]}"; do read -r sm om <<< "$m"
  for v in "${VARIANTI[@]}"; do read -r sv ov <<< "$v"
    X="model=yolo26 $om model.depth=D101 target=rpi5_640 $ov logging=local retry_failed=true"; L=$D/${sm}_$sv.log; : > $L; esito=ok
    for s in export benchmark; do
      flock runs/.pi.lock $PY run.py stage=$s $X >> $L 2>&1 || esito=FALLITO
      n=$((n+1)); riga $n "$sm $sv $s $esito"
    done
    # record riuscito piu' recente di questa cella (modello, precisione, backend), anche se gia' in cache
    $PY - "$om" "$ov" "$sm" "$sv" >> $T <<'PYEOF'
import json, sys, glob, os
om, ov, sm, sv = sys.argv[1:5]
w = float(om.split("=")[1]); q, b = (x.split("=")[1] for x in ov.split())
best = None
for p in glob.glob("results/*.json"):
    try: d = json.load(open(p))
    except Exception: continue
    c = d.get("config") or {}
    if (d.get("latency") and (c.get("model") or {}).get("width") == w and c["model"].get("depth") == "D101"
            and c["quantization"]["name"] == q and c["backend"]["name"] == b and c["strategy"].get("name") == "baseline"
            and "rpi5" in json.dumps(c.get("hardware")) and (best is None or os.path.getmtime(p) > best[0])):
        best = (os.path.getmtime(p), d)
lat = acc = "-"
if best:
    lat = best[1]["latency"].get("median_ms", "-"); acc = (best[1].get("accuracy") or {}).get("map50_95", "-")
    acc = f"{acc:.3f}" if isinstance(acc, float) else acc
print(f"{sm}\t{sv}\t{lat}\t{acc}")
PYEOF
  done
done
{ echo; echo "FINITO in $(( ($(date +%s) - t0) / 60 )) min"; column -t -s $'\t' $T; } >> $A
