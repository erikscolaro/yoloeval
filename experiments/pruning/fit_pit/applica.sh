#!/bin/bash
# Rende fit_pit2.json il default: artifacts/lut/rpi5/fit.json (il vecchio fit.json resta come fit_pit.json) e la copia in yolopit.
# Rifiuta se c'e' un'unita' pendenze-l1* o una ricerca pit_latency in corso: il fit fa parte della chiave di cache delle ricerche.
set -euo pipefail
X=$(cd "$(dirname "$0")" && pwd)
LUT=$X/../../../artifacts/lut/rpi5
YP=${YOLOPIT:-$HOME/Desktop/yolopit}/yolopit/data/rpi5
if systemctl --user is-active --quiet 'pendenze-l1*' || pgrep -f "strategy=pit_latency" >/dev/null; then
  echo "ricerche con il fit in corso: aspetta" >&2; exit 1
fi
cmp -s "$X/results/fit_pit2.json" "$LUT/fit.json" && { echo "gia' applicato"; exit 0; }
cp "$LUT/fit.json" "$LUT/fit_pit.json"          # il default di prima (PIT del solo nano)
cp "$X/results/fit_pit2.json" "$LUT/fit_pit2.json"
cp "$X/results/fit_pit2.json" "$LUT/fit.json"
cp "$X/results/fit_pit2.json" "$YP/fit.json"
echo "fit.json aggiornato (artifacts/lut/rpi5 e yolopit); fit_b4r.json resta il vecchio di B4-R"
