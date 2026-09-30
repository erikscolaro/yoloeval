#!/usr/bin/env bash
# Compila `onnxruntime_perf_test` e lo installa nel venv del tool.
#
# Nessun pacchetto lo distribuisce: ne' il wheel pip ne' i .tgz delle release
# di ONNX Runtime lo contengono. Si compila dai sorgenti, una volta per board,
# e lo chiamano gli script di provisioning con le variabili qui sotto.
#
#   ORT_VENV         venv in cui installarlo (obbligatorio)
#   ORT_CUDA_HOME    toolkit CUDA; vuoto = build solo CPU
#   ORT_CUDNN_HOME   cuDNN: cartella con include/ e lib/, oppure la lib dir
#                    di sistema (Jetson)
#   ORT_CUDA_ARCHS   compute capability senza punto, es. 89 (RTX Ada), 87 (Orin)
#   ORT_VERSION      default: la versione del pacchetto onnxruntime del venv
#   ORT_BUILD_ROOT   sorgenti e build (default ~/.cache/yolo-bench/ort)
#   ORT_BUILD_JOBS   default: limitati dalla RAM, non solo dai core
#
# La versione segue quella del pacchetto Python perche' la validazione
# numerica gira con quello: perf_test di un'altra release misurerebbe un
# runtime diverso da quello validato.
#
# Idempotente: se il binario installato ha gia' versione e build (CPU/CUDA)
# richieste non fa nulla.
set -euo pipefail

say() { echo "[ort-perf-test] $*"; }
die() { echo "[ort-perf-test] $*" >&2; exit 1; }

# Il proxy della rete interna cade anche per minuti: ogni passo che scarica si riprova
# per RETRY_MINUTES (da stage.retry_minutes, default 30) prima di arrendersi.
retry() {
  local what=$1; shift
  local deadline=$(( $(date +%s) + ${RETRY_MINUTES:-30} * 60 )) attempt=1
  until "$@"; do
    (( $(date +%s) >= deadline )) && die "$what: fallito per ${RETRY_MINUTES:-30} minuti"
    say "$what fallito (tentativo $attempt), riprovo fra 30 s"
    attempt=$((attempt + 1))
    sleep 30
  done
}

VENV="${ORT_VENV:?ORT_VENV non impostato}"
CUDA_HOME="${ORT_CUDA_HOME:-}"
CUDNN_HOME="${ORT_CUDNN_HOME:-}"
CUDA_ARCHS="${ORT_CUDA_ARCHS:-}"
BUILD_ROOT="${ORT_BUILD_ROOT:-$HOME/.cache/yolo-bench/ort}"

VERSION="${ORT_VERSION:-$("$VENV/bin/python" -c 'import onnxruntime as o; print(o.__version__)' 2>/dev/null || true)}"
VERSION="${VERSION%%+*}"   # i wheel Jetson hanno suffissi locali tipo +cu126
[[ -n "$VERSION" ]] || die "versione di onnxruntime non determinabile: installarlo nel venv o impostare ORT_VERSION"

if [[ -n "$CUDA_HOME" ]]; then
  [[ -x "$CUDA_HOME/bin/nvcc" ]] || die "nvcc non trovato in $CUDA_HOME/bin"
  [[ -n "$CUDNN_HOME" ]] || die "build CUDA senza ORT_CUDNN_HOME"
  [[ -n "$CUDA_ARCHS" ]] || die "build CUDA senza ORT_CUDA_ARCHS"
  FLAVOR="cuda-sm${CUDA_ARCHS}"
else
  FLAVOR="cpu"
fi

INSTALL_DIR="$VENV/opt/onnxruntime_perf_test"
WRAPPER="$VENV/bin/onnxruntime_perf_test"
STAMP="$INSTALL_DIR/.build"
WANT="$VERSION $FLAVOR"

if [[ -x "$WRAPPER" && -f "$STAMP" && "$(cat "$STAMP")" == "$WANT" ]]; then
  say "gia' installato ($WANT)"
  exit 0
fi
say "compilo onnxruntime_perf_test $WANT"

# Toolchain in un venv a parte: ORT 1.2x vuole CMake >= 3.28, che ne' Ubuntu
# 22.04 (JetPack 6) ne' Raspberry Pi OS bookworm hanno da apt.
TOOLS="$BUILD_ROOT/toolenv"
[[ -x "$TOOLS/bin/cmake" && -x "$TOOLS/bin/ninja" ]] || {
  python3 -m venv "$TOOLS"
  retry "installazione di cmake e ninja" \
    "$TOOLS/bin/python" -m pip install --quiet --upgrade pip "cmake>=3.28" ninja
}
export PATH="$TOOLS/bin:$PATH"

SRC="$BUILD_ROOT/onnxruntime-$VERSION"
# Un clone interrotto lascia una cartella a meta' (magari con .git ma senza tutti
# i submodule): conta solo il marker scritto a clone riuscito.
clone_ort() {
  rm -rf "$SRC"
  git clone --depth 1 --branch "v$VERSION" --recursive --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git "$SRC" && touch "$SRC/.clone-ok"
}
[[ -f "$SRC/.clone-ok" ]] || retry "clone di onnxruntime" clone_ort

# Parallelismo limitato dalla RAM: i sorgenti CUDA arrivano a 3 GB per job, e
# su un Raspberry da 4 GB `-j4` finisce in OOM a meta' build.
if [[ -z "${ORT_BUILD_JOBS:-}" ]]; then
  mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  per_job=$([[ -n "$CUDA_HOME" ]] && echo 3 || echo 2)
  ORT_BUILD_JOBS=$(( mem_gb / per_job ))
  (( ORT_BUILD_JOBS > $(nproc) )) && ORT_BUILD_JOBS=$(nproc)
  (( ORT_BUILD_JOBS < 1 )) && ORT_BUILD_JOBS=1
fi
say "job paralleli: $ORT_BUILD_JOBS"

ARGS=(--build_dir "$SRC/build" --config Release --update
      --cmake_generator Ninja --skip_tests --compile_no_warning_as_error)
CUDNN_LIB=""
if [[ -n "$CUDA_HOME" ]]; then
  # Il cuDNN dei wheel pip ha solo libcudnn*.so.9, senza il link .so che serve
  # al linker: gli si affianca una cartella con i link, senza toccare il venv.
  if [[ -d "$CUDNN_HOME/include" && -d "$CUDNN_HOME/lib" && ! -e "$CUDNN_HOME/lib/libcudnn.so" ]]; then
    shim="$BUILD_ROOT/cudnn-home"
    rm -rf "$shim" && mkdir -p "$shim/lib"
    ln -s "$CUDNN_HOME/include" "$shim/include"
    for so in "$CUDNN_HOME"/lib/libcudnn*.so.*; do
      ln -s "$so" "$shim/lib/$(basename "$so")"
      ln -sf "$so" "$shim/lib/$(basename "${so%.so.*}").so"
    done
    CUDNN_LIB="$CUDNN_HOME/lib"
    CUDNN_HOME="$shim"
  fi
  ARGS+=(--use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME"
         --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS")
fi

# build.py configura soltanto; la build si limita ai target che servono,
# invece di compilare anche le migliaia di unit test. La configurazione scarica
# una trentina di dipendenze (FetchContent) e non riprova da sola: si rilancia,
# quelle gia' scaricate restano in build/Release/_deps.
configure_ort() { (cd "$SRC" && python3 tools/ci_build/build.py "${ARGS[@]}"); }
retry "configurazione di onnxruntime" configure_ort
TARGETS=(onnxruntime_perf_test onnxruntime_providers_shared)
[[ -n "$CUDA_HOME" ]] && TARGETS+=(onnxruntime_providers_cuda)
cmake --build "$SRC/build/Release" --target "${TARGETS[@]}" -j "$ORT_BUILD_JOBS"

# Binario e provider insieme: ORT carica libonnxruntime_providers_*.so dalla
# cartella dell'eseguibile. Nel bin/ del venv va solo un wrapper, che porta
# anche il path del cuDNN pip (fuori dal path del loader di sistema).
rm -rf "$INSTALL_DIR" && mkdir -p "$INSTALL_DIR"
cp "$SRC/build/Release/onnxruntime_perf_test" "$INSTALL_DIR/"
cp "$SRC"/build/Release/libonnxruntime_providers_*.so "$INSTALL_DIR/"
{
  echo '#!/bin/sh'
  echo "# generato da build_ort_perf_test.sh ($WANT)"
  [[ -n "$CUDNN_LIB" ]] && echo "export LD_LIBRARY_PATH=\"$CUDNN_LIB\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
  echo "exec \"$INSTALL_DIR/onnxruntime_perf_test\" \"\$@\""
} > "$WRAPPER"
chmod +x "$WRAPPER"

# Verifica: una libreria mancante qui diventa una cella failed a meta' sweep.
for f in "$INSTALL_DIR"/onnxruntime_perf_test "$INSTALL_DIR"/*.so; do
  missing=$(LD_LIBRARY_PATH="${CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$f" | grep "not found" || true)
  [[ -z "$missing" ]] || die "$(basename "$f"): librerie mancanti:
$missing"
done

echo "$WANT" > "$STAMP"
say "installato in $WRAPPER ($WANT)"
