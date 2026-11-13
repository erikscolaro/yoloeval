#!/bin/bash
# Confronta la LUT di out_dir (scripts/build_lut.sh) con quella del repo di yolopit (yolopit/data/rpi5, il default
# di pit.latency) e, con --write, la copia lì. Esce con 1 se qualche file è diverso o manca da una delle due parti.
#   scripts/sync_yolopit_lut.sh [--write] [out_dir]      out_dir di default artifacts/lut/rpi5
# Il repo di yolopit è YOLOPIT (default ../yolopit accanto a questo repo), non la copia installata nel venv.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=0
[ "${1:-}" = --write ] && { W=1; shift; }
OUT=${1:-$ROOT/artifacts/lut/rpi5}
DST=${YOLOPIT:-$ROOT/../yolopit}/yolopit/data/rpi5
[ -d "$DST" ] || { echo "$DST non esiste: indicare YOLOPIT=<repo di yolopit>"; exit 2; }
diff=0
for f in "$OUT"/lut_*.csv "$OUT/fit.json"; do
  if cmp -s "$f" "$DST/$(basename "$f")"; then echo "identico  $(basename "$f")"; else echo "DIVERSO   $(basename "$f")"; diff=1; fi
done
for f in "$DST"/*; do
  [ -e "$OUT/$(basename "$f")" ] || { echo "solo in yolopit: $(basename "$f")"; diff=1; }
done
if [ $W = 1 ]; then
  cp "$OUT"/lut_*.csv "$OUT/fit.json" "$DST/"
  echo "copiati in $DST"
fi
exit $diff
