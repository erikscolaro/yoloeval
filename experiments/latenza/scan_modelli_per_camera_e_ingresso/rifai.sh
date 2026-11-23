#!/usr/bin/env bash
# Celle camera x ingresso dello SCAN: costo fuori dal modello sul Pi (misura.py, 3 passate, lock del Pi), poi la tabella
# (tabella.py: miglior modello per cella). Risultati in results/.
#   experiments/latenza/scan_modelli_per_camera_e_ingresso/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results; : > results/fuori_modello.jsonl
scp -q misura.py rpi5:/tmp/misura_celle.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "taskset -c 0 /home/bench/bench/.venv/bin/python /tmp/misura_celle.py" | sed "s/^{/{\"passata\": $r, /" | tee -a results/fuori_modello.jsonl
done
exec 9>&-
../../../.venv/bin/python tabella.py
