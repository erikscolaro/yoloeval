#!/usr/bin/env bash
# Inverso esatto di prepare_board.sh.
#
# Viene invocato dal guard quando lo heartbeat si ferma, e a mano quando uno
# sweep e' stato interrotto in modo brutale. Se una cella lascia la macchina in
# uno stato diverso da come l'ha trovata, deve essere visibile subito: questo
# script e' il modo per rimetterla a posto senza riavviarla.
set -euo pipefail

STATE_FILE="${YOLO_BENCH_STATE:-/tmp/yolo_bench_guard.state}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --state) STATE_FILE="$2"; shift 2 ;;
    *) echo "argomento sconosciuto: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$STATE_FILE" ]] || { echo "nessuno stato salvato in $STATE_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$STATE_FILE"

if [[ "${GOVERNOR_PREV:-unknown}" != "unknown" ]]; then
  echo "$GOVERNOR_PREV" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
fi
if [[ "${FREQ_PREV:-unknown}" != "unknown" && "${GOVERNOR_PREV:-}" == "userspace" ]]; then
  echo "$FREQ_PREV" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed >/dev/null 2>&1 || true
fi
if [[ "${SWAP_PREV:-0}" -gt 0 ]]; then
  sudo swapon -a || true
fi
if [[ -n "${NVPMODEL_PREV:-}" ]] && command -v nvpmodel >/dev/null 2>&1; then
  echo NO | sudo nvpmodel -m "$NVPMODEL_PREV" >/dev/null 2>&1 || true
fi

echo "stato ripristinato da $STATE_FILE"
