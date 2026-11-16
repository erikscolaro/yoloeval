#!/usr/bin/env bash
# Compila i tool di ncnn che il wheel pip non contiene e li installa nel venv:
# benchncnn (latenza sulla board), ncnn2table e ncnn2int8 (quantizzazione int8 sulla workstation).
#
#   NCNN_VENV        venv in cui installarli (obbligatorio)
#   NCNN_VERSION     tag dei sorgenti, default: la versione del pacchetto ncnn del venv
#   NCNN_BUILD_ROOT  sorgenti e build (default ~/.cache/yolo-bench/ncnn)
#
# La versione segue il pacchetto Python, che fa la mAP: benchncnn di un'altra release misurerebbe un
# runtime diverso. Idempotente: con la versione gia' installata non fa nulla.
set -euo pipefail

say() { echo "[ncnn] $*"; }
die() { echo "[ncnn] $*" >&2; exit 1; }
retry() {
  local what=$1; shift
  local deadline=$(( $(date +%s) + ${RETRY_MINUTES:-30} * 60 )) attempt=1
  until "$@"; do
    (( $(date +%s) >= deadline )) && die "$what: fallito per ${RETRY_MINUTES:-30} minuti"
    say "$what fallito (tentativo $attempt), riprovo fra 30 s"; attempt=$((attempt + 1)); sleep 30
  done
}

VENV="${NCNN_VENV:?NCNN_VENV non impostato}"
BUILD_ROOT="${NCNN_BUILD_ROOT:-$HOME/.cache/yolo-bench/ncnn}"
VERSION="${NCNN_VERSION:-$("$VENV/bin/python" -c 'import ncnn; print(ncnn.__version__)' 2>/dev/null || true)}"
VERSION="${VERSION##*.}"   # il wheel e' 1.0.YYYYMMDD, il tag e' YYYYMMDD
[[ -n "$VERSION" ]] || die "versione di ncnn non determinabile: installarlo nel venv o impostare NCNN_VERSION"

STAMP="$VENV/bin/.ncnn_tools_version"
if [[ "$(cat "$STAMP" 2>/dev/null)" == "$VERSION" ]]; then say "tool $VERSION gia' installati"; exit 0; fi

SRC="$BUILD_ROOT/ncnn-$VERSION"
[[ -d "$SRC" ]] || retry "git clone ncnn $VERSION" \
  git clone --depth 1 --branch "$VERSION" https://github.com/Tencent/ncnn.git "$SRC"
# SIMPLEOCV: ncnn2table legge le immagini di calibrazione senza OpenCV di sistema
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_SIMPLEOCV=ON \
  -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_PYTHON=OFF
cmake --build "$SRC/build" -j"$(nproc)" --target benchncnn ncnn2table ncnn2int8
install -m755 "$SRC/build/benchmark/benchncnn" "$SRC/build/tools/quantize/ncnn2table" \
  "$SRC/build/tools/quantize/ncnn2int8" "$VENV/bin/"
echo "$VERSION" > "$STAMP"
say "installati benchncnn, ncnn2table, ncnn2int8 ($VERSION) in $VENV/bin"
