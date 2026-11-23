#!/usr/bin/env bash
# Perche' INTER_AREA costa di piu' con alcuni fattori di riduzione (misura_area.py): solo cv2.resize, un thread
# (e quattro nella variante area4), governor performance, 2 passate, lock del Pi. Risultati in results/area_fattori.jsonl.
#   experiments/latenza/track_costo_ritaglio/rifai_area.sh
set -euo pipefail
cd "$(dirname "$0")"
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results; : > results/area_fattori.jsonl
scp -q misura_area.py rpi5:/tmp/misura_area.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1; do
  ssh rpi5 "/home/bench/bench/.venv/bin/python /tmp/misura_area.py 300" | sed "s/^{/{\"passata\": $r, /" | tee -a results/area_fattori.jsonl
done
