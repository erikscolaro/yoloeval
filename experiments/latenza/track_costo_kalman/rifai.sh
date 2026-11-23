#!/usr/bin/env bash
# Costo di KalmanTracker.step sul Pi 5 (misura.py), un core, governor performance, 3 passate, sotto il lock del Pi.
# Risultati in results/kalman.jsonl; rimette il governor com'era. Serve scipy nel venv (scripts/requirements).
#   experiments/latenza/track_costo_kalman/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results
ssh rpi5 mkdir -p /tmp/track_costo_kalman
scp -q misura.py ../../../src/tracking.py rpi5:/tmp/track_costo_kalman/
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "taskset -c 0 /home/bench/bench/.venv/bin/python /tmp/track_costo_kalman/misura.py" | sed "s/^{/{\"passata\": $r, /" | tee -a results/kalman.jsonl
done
