#!/usr/bin/env bash
# Tuning temporaneo della board, con il suo inverso in restore_board.sh.
#
# Ogni azione qui dentro e' reversibile e passa da sysfs o da comandi runtime:
# niente /boot/firmware/config.txt, niente unit systemd, niente parametri
# kernel. Le modifiche persistenti sono fuori dalle regole del tool.
#
# Uso:
#   prepare_board.sh --governor performance [--freq-khz 1500000]
#                    [--nvpmodel 2] [--swap-off] [--drop-caches]
#                    [--guard-minutes 30]
#
# Emette su stdout un JSON con lo stato applicato **davvero** (frequenza
# riletta, core online, governor effettivo): un valore non supportato viene
# arrotondato in silenzio dal kernel, e la differenza fra richiesto ed
# effettivo deve finire nel risultato.
#
# Il tool applica il tuning dai propri controller Python (src/measure/): questo
# script serve per l'uso manuale, per il guard degli sweep non presidiati e
# come documentazione eseguibile di cosa viene toccato.
set -euo pipefail

GOVERNOR=""
FREQ_KHZ=""
NVPMODEL=""
SWAP_OFF=0
DROP_CACHES=0
GUARD_MINUTES=""
STATE_FILE="${YOLO_BENCH_STATE:-/tmp/yolo_bench_guard.state}"
HEARTBEAT="${YOLO_BENCH_HEARTBEAT:-/tmp/yolo_bench_guard.heartbeat}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --governor) GOVERNOR="$2"; shift 2 ;;
    --freq-khz) FREQ_KHZ="$2"; shift 2 ;;
    --nvpmodel) NVPMODEL="$2"; shift 2 ;;
    --swap-off) SWAP_OFF=1; shift ;;
    --drop-caches) DROP_CACHES=1; shift ;;
    --guard-minutes) GUARD_MINUTES="$2"; shift 2 ;;
    *) echo "argomento sconosciuto: $1" >&2; exit 2 ;;
  esac
done

cpu0=/sys/devices/system/cpu/cpu0/cpufreq

# --- salva lo stato, per poterlo ripristinare -------------------------------
{
  echo "GOVERNOR_PREV=$(cat $cpu0/scaling_governor 2>/dev/null || echo unknown)"
  echo "FREQ_PREV=$(cat $cpu0/scaling_setspeed 2>/dev/null || echo unknown)"
  echo "SWAP_PREV=$(swapon --show --noheadings 2>/dev/null | wc -l)"
  if command -v nvpmodel >/dev/null 2>&1; then
    echo "NVPMODEL_PREV=$(sudo nvpmodel -q 2>/dev/null | tail -1 | tr -dc '0-9')"
  fi
} > "$STATE_FILE"

# --- nvpmodel prima di cpufreq ----------------------------------------------
# nvpmodel impone il tetto, il governor sfrutta il tetto: scrivere una
# frequenza superiore al tetto non ha effetto.
if [[ -n "$NVPMODEL" ]] && command -v nvpmodel >/dev/null 2>&1; then
  echo NO | sudo nvpmodel -m "$NVPMODEL" >/dev/null 2>&1 || true
  sleep 30   # stabilizzazione dopo il cambio di profilo
fi

if [[ -n "$GOVERNOR" ]]; then
  available=$(cat "$cpu0/scaling_available_governors" 2>/dev/null || echo "")
  if [[ -n "$available" && "$available" != *"$GOVERNOR"* ]]; then
    echo "governor $GOVERNOR non disponibile ($available)" >&2
    exit 3
  fi
  echo "$GOVERNOR" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
fi

if [[ -n "$FREQ_KHZ" ]]; then
  echo "$FREQ_KHZ" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed >/dev/null 2>&1 || true
fi

# Meglio un OOM visibile che una cella in swap.
[[ $SWAP_OFF -eq 1 ]] && sudo swapoff -a || true
# Cosi' il primo caricamento del modello e' confrontabile fra celle.
[[ $DROP_CACHES -eq 1 ]] && { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; } || true

# --- guard per sweep non presidiati -----------------------------------------
# Il `finally` di Python non copre kill -9 ne' una connessione che cade: se lo
# heartbeat si ferma, la board torna com'era da sola.
if [[ -n "$GUARD_MINUTES" ]]; then
  touch "$HEARTBEAT"
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  setsid nohup bash -c "
    while true; do
      sleep 60
      age=\$(( \$(date +%s) - \$(stat -c %Y '$HEARTBEAT' 2>/dev/null || echo 0) ))
      if (( age > $((GUARD_MINUTES * 60)) )); then
        '$here/restore_board.sh' --state '$STATE_FILE'
        exit 0
      fi
    done" >/dev/null 2>&1 &
fi

# --- stato effettivo, riletto ------------------------------------------------
cur_gov=$(cat $cpu0/scaling_governor 2>/dev/null || echo null)
cur_freq=$(cat $cpu0/scaling_cur_freq 2>/dev/null || echo null)
online=$(nproc 2>/dev/null || echo null)
temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo null)
swap=$(swapon --show --noheadings 2>/dev/null | wc -l)

printf '{"governor":"%s","freq_actual_khz":%s,"freq_requested_khz":%s,' \
  "$cur_gov" "${cur_freq:-null}" "${FREQ_KHZ:-null}"
printf '"cores_online":%s,"temp_c":%s,"swap_off":%s,"nvpmodel_id":%s}\n' \
  "${online:-null}" \
  "$([[ "$temp" == null ]] && echo null || echo "scale=1; $temp/1000" | bc)" \
  "$([[ "$swap" -eq 0 ]] && echo true || echo false)" \
  "${NVPMODEL:-null}"
