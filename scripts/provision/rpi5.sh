#!/usr/bin/env bash
# Provisioning di un Raspberry Pi 5 per le celle CPU: venv dell'host e onnxruntime_perf_test.
# Con la scheda Axelera Metis (hardware=rpi5_axelera) gira dopo rpi5_axelera.sh.
set -euo pipefail

WORKDIR="${BENCH_WORKDIR:-$HOME/bench}"
REQUIREMENTS="${BENCH_REQUIREMENTS:-/tmp/requirements.txt}"
VENV="$WORKDIR/.venv"

say() { echo "[provision][rpi5] $*"; }

# proxy instabile: pip e apt riprovano di piu' prima di arrendersi
export PIP_RETRIES=10 PIP_TIMEOUT=60
echo 'Acquire::Retries "10";' | sudo tee /etc/apt/apt.conf.d/80retries >/dev/null

mkdir -p "$WORKDIR"

# 1. libgl1: richiesto da OpenCV, non incluso via pip.
sudo apt-get install -y libgl1 python3-venv

# 2. venv host, per le celle CPU-only che non usano il container.
if [[ ! -d "$VENV" ]]; then
  say "creo il venv host in $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
# torch prima del resto e dall'indice CPU: quello di PyPI per aarch64 e' la build CUDA e si
# porta dietro GB di pacchetti nvidia-* inutili sul Pi (li tira dentro ultralytics).
python -m pip install --no-cache-dir torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r "$REQUIREMENTS"
# residui di un'installazione precedente con torch CUDA
NVIDIA=$(python -m pip freeze | grep -iE '^(nvidia-|triton)' | cut -d= -f1 | cut -d' ' -f1 || true)
if [[ -n "$NVIDIA" ]]; then
  say "rimuovo i pacchetti CUDA: $(echo $NVIDIA | tr '\n' ' ')"
  echo "$NVIDIA" | xargs python -m pip uninstall -y
fi

# 3. onnxruntime_perf_test, solo CPU: sul Pi non c'e' CUDA. La build dura
#    qualche ora; se si interrompe, rilanciare riprende da dove era.
sudo apt-get install -y git build-essential
ORT_VENV="$VENV" bash "$WORKDIR/tools/build_ort_perf_test.sh"

sudo -n true 2>/dev/null || \
  say "ATTENZIONE: sudo senza password non configurato (serve per cpufreq)"

say "ok"
