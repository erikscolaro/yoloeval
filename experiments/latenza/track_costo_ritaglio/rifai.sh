#!/usr/bin/env bash
# Costo di taglio + resize + pre del crop adattativo sul Pi 5 (misura.py): un core, governor performance, 3 passate,
# sotto il lock del Pi. Risultati in results/crop_resize.jsonl; rimette il governor com'era.
#   experiments/latenza/track_costo_ritaglio/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results
scp -q misura.py rpi5:/tmp/misura_crop_resize.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "taskset -c 0 /home/bench/bench/.venv/bin/python /tmp/misura_crop_resize.py" | sed "s/^{/{\"passata\": $r, /" | tee -a results/crop_resize.jsonl
done
