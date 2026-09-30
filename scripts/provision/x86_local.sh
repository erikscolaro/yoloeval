#!/usr/bin/env bash
# Provisioning della workstation x86_64.
#
# La workstation e' locale: non passa da SSH e non ha un file `remote` nel
# config. Questo script serve a preparare il venv e a verificare che i tool di
# misura ci siano davvero, prima di scoprirlo a meta' sweep.
set -euo pipefail

WORKDIR="${BENCH_WORKDIR:-$(pwd)}"
REQUIREMENTS="${BENCH_REQUIREMENTS:-scripts/requirements/x86_64.txt}"
VENV="${BENCH_VENV:-$WORKDIR/.venv}"

say() { echo "[provision][x86] $*"; }

[[ -d "$VENV" ]] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r "$REQUIREMENTS"

# trtexec non arriva da pip: e' nel pacchetto apt `libnvinfer-bin` del
# repository CUDA di NVIDIA, e si installa in /usr/src/tensorrt/bin (fuori dal
# PATH, il backend tensorrt lo cerca anche li').
#
# Versione fissata a TensorRT 10, per compatibilita' con Ultralytics e con
# conf/quantization: l'11 ha tolto --fp16/--int8 da trtexec (la precisione va
# scritta nell'ONNX, e Ultralytics lo fa solo passando da NVIDIA ModelOpt). Deve
# coincidere con tensorrt-cu13 in scripts/requirements/x86_64.txt, o l'engine
# costruito da trtexec non si carica in validazione. Il metapacchetto
# `tensorrt` non va usato: si tira dietro i binding Python di sistema, che su
# Ubuntu 24.04 vanno in conflitto con Python 3.12.
TRT_VERSION=10.16.1.11-1+cuda13.2
TRT_PACKAGES=(libnvinfer-bin libnvinfer10 libnvinfer-lean10 libnvinfer-plugin10
              libnvinfer-vc-plugin10 libnvinfer-dispatch10 libnvonnxparsers10)
TRTEXEC=/usr/src/tensorrt/bin/trtexec
installed=$(dpkg-query -W -f='${Version}' libnvinfer-bin 2>/dev/null || true)
if [[ "$installed" != "$TRT_VERSION" ]]; then
  # Output salvato prima del grep: con `pipefail`, `grep -q` chiude la pipe
  # appena trova la riga, apt-cache muore di SIGPIPE e il test risulta falso.
  available=$(LC_ALL=C apt-cache madison libnvinfer-bin 2>/dev/null || true)
  if grep -qF "$TRT_VERSION" <<<"$available"; then
    say "installo TensorRT $TRT_VERSION (trtexec)${installed:+, al posto di $installed}"
    sudo apt-mark unhold "${TRT_PACKAGES[@]}" >/dev/null 2>&1 || true
    sudo apt-get install -y --allow-downgrades \
      "${TRT_PACKAGES[@]/%/=$TRT_VERSION}"
    # senza hold, il primo `apt upgrade` riporterebbe TensorRT 11
    sudo apt-mark hold "${TRT_PACKAGES[@]}"
  else
    say "ATTENZIONE: libnvinfer-bin $TRT_VERSION non disponibile, manca il repository CUDA di NVIDIA"
  fi
fi

# onnxruntime_perf_test si compila dai sorgenti (vedi lo script). Con una GPU
# NVIDIA la build include il provider CUDA, contro il toolkit di sistema e il
# cuDNN pip del venv, cioe' lo stesso che usa onnxruntime-gpu in validazione.
# BENCH_ORT_CUDA=0 forza la build solo CPU.
ORT_ENV=(ORT_VENV="$VENV")
if [[ "${BENCH_ORT_CUDA:-1}" != 0 ]] && nvidia-smi >/dev/null 2>&1; then
  CUDA_HOME="${BENCH_CUDA_HOME:-/usr/local/cuda}"
  CUDNN_HOME=$(python -c 'import nvidia.cudnn as c; print(c.__path__[0])' 2>/dev/null || true)
  [[ -n "$CUDNN_HOME" ]] || { echo "cuDNN pip non trovato nel venv (nvidia-cudnn-cu13)" >&2; exit 1; }
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  ORT_ENV+=(ORT_CUDA_HOME="$CUDA_HOME" ORT_CUDNN_HOME="$CUDNN_HOME" ORT_CUDA_ARCHS="$ARCH")
fi
env "${ORT_ENV[@]}" bash "$WORKDIR/scripts/remote/build_ort_perf_test.sh"

for tool in onnxruntime_perf_test trtexec benchmark_app pandoc; do
  if [[ "$tool" == trtexec && -x "$TRTEXEC" ]]; then
    say "$tool: $TRTEXEC"
  elif command -v "$tool" >/dev/null 2>&1; then
    say "$tool: $(command -v $tool)"
  else
    say "ATTENZIONE: $tool non nel PATH — le celle che lo usano falliranno"
  fi
done

say "ok"
