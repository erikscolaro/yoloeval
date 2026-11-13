#!/bin/bash
# LUT di latenza del Raspberry Pi 5 per il nano e stimatore di yolopit, da capo (src/lut, parametri in src/lut/pi.py):
#   lut  misura della LUT (13 famiglie, passo 2, ~40 h) e analisi -> lut_<famiglia>.csv
#   b4r  reti B4-R (765, generate una volta sola), misura intera sul Pi, fit -> fit.json
#   out  copia lut_<famiglia>.csv e fit.json in out_dir, i file di pit.latency (lut_dir, fit)
#
#   scripts/build_lut.sh [--dry-run] [out_dir]          out_dir di default artifacts/lut/rpi5
#   systemd-run --user --unit=build-lut --working-directory=$PWD scripts/build_lut.sh   # staccato dal terminale
#
# Ripresa: basta rilanciarlo. Le misure saltano quello che c'è già nei raw.csv, le reti B4-R non si rigenerano
# se c'è results/nets.csv. Un errore della misura (ssh, Pi riavviato) la rilancia, fino a 20 volte, 60 s dopo;
# Ctrl-C o systemctl stop la fermano (la board torna com'era) senza rilanciarla.
# --dry-run stampa passi e conteggi senza toccare il Pi.
# LUT_DIR e B4R_DIR (default out_dir/lut, out_dir/b4r, ognuna con results/ e runs/) e STEPS (default
# "lut b4r out") servono ai rifai.sh degli esperimenti (experiments/pruning/lut_pi_passo2, b4r_composizione).
# Dopo: scripts/sync_yolopit_lut.sh confronta out_dir con i dati dentro yolopit e, con --write, li sostituisce.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DRY=0
[ "${1:-}" = --dry-run ] && { DRY=1; shift; }
OUT=$(realpath -m "${1:-$ROOT/artifacts/lut/rpi5}")
LUT_DIR=$(realpath -m "${LUT_DIR:-$OUT/lut}")
B4R_DIR=$(realpath -m "${B4R_DIR:-$OUT/b4r}")
STEPS=${STEPS:-lut b4r out}
PY=$ROOT/.venv/bin/python
cd "$ROOT"
trap 'echo "$(date -Is) STOP richiesto"; exit 130' INT TERM   # niente rilancio dopo uno stop

run() {   # run <comando...>: in dry-run lo stampa soltanto
  echo "   $*"
  [ $DRY = 1 ] || "$@"
}
retry() {
  echo "   $* (fino a 20 tentativi)"
  [ $DRY = 1 ] && return 0
  for i in $(seq 1 20); do
    "$@" && return 0
    echo "$(date -Is) errore, riprovo fra 60 s ($i/20)"
    sleep 60
  done
  return 1
}
rows() { [ -f "$1" ] && echo $(($(wc -l < "$1") - 1)) || echo 0; }

echo "out_dir $OUT, LUT_DIR $LUT_DIR, B4R_DIR $B4R_DIR, passi: $STEPS"
if [ $DRY = 1 ]; then
  "$PY" -c "from src.lut.pi import ARCHS, N_NETS, REPEATS, EXTRA_REPEATS, points
n = len(points())
print(f'LUT: {n} punti x {REPEATS} = {n * REPEATS} misure, più {EXTRA_REPEATS} ripetizioni sui punti instabili')
print(f'B4-R: {len(ARCHS) * (N_NETS + 1)} reti x {REPEATS} = {len(ARCHS) * (N_NETS + 1) * REPEATS} misure')"
  echo "già fatte: LUT $(rows "$LUT_DIR/results/raw.csv") misure, B4-R $(rows "$B4R_DIR/results/raw.csv") misure"
fi

for s in $STEPS; do
  echo "== $s"
  case $s in
    lut)
      retry "$PY" -m src.lut.pi lut "$LUT_DIR"
      run "$PY" -m src.lut.analyze "$LUT_DIR" ;;
    b4r)
      if [ -f "$B4R_DIR/results/nets.csv" ]; then echo "   reti B4-R già generate ($B4R_DIR/results/nets.csv)"
      else run "$PY" -m src.lut.b4r "$B4R_DIR"; fi
      retry "$PY" -m src.lut.pi b4r "$B4R_DIR"
      run "$PY" -m src.lut.fit "$B4R_DIR" "$LUT_DIR/results" ;;
    out)
      run mkdir -p "$OUT"
      run cp "$LUT_DIR"/results/lut_*.csv "$B4R_DIR/results/fit.json" "$OUT/" ;;
    *) echo "passo sconosciuto: $s"; exit 2 ;;
  esac
done
