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

for tool in onnxruntime_perf_test trtexec benchmark_app pandoc; do
  if command -v "$tool" >/dev/null 2>&1; then
    say "$tool: $(command -v $tool)"
  else
    say "ATTENZIONE: $tool non nel PATH — le celle che lo usano falliranno"
  fi
done

say "ok"
