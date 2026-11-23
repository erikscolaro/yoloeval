#!/usr/bin/env bash
# Budget del modello di SCAN a 720p senza cambiare le proporzioni (misura.py): un core, governor performance,
# 3 passate, sotto il lock del Pi. Risultati in results/budget_scan.jsonl; rimette il governor com'era.
#   experiments/latenza/scan_budget_720p/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results; : > results/budget_scan.jsonl
scp -q misura.py rpi5:/tmp/misura_budget_scan.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "taskset -c 0 /home/bench/bench/.venv/bin/python /tmp/misura_budget_scan.py" | sed "s/^{/{\"passata\": $r, /" | tee -a results/budget_scan.jsonl
done
