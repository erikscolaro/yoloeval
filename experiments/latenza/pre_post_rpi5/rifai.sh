#!/usr/bin/env bash
# B1-2 sul Pi 5 a 640: pre e post-processing fuori dal grafo (misura.py), governor performance, 3 passate.
# Risultati in results/pre_post.jsonl, una riga per passata.
#   experiments/latenza/pre_post_rpi5/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
scp -q misura.py rpi5:/tmp/misura_pre_post.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "taskset -c 0 /home/bench/bench/.venv/bin/python /tmp/misura_pre_post.py 640" | tee -a results/pre_post.jsonl
done
