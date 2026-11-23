#!/usr/bin/env bash
# Pre e post a ogni ingresso di deploy sul Pi (misura.py): un core, governor performance, 3 passate, sotto il lock del
# Pi (aspetta le altre misure). Poi rifa' la figura del budget della griglia (../modelli_latenza_per_ingresso/report) con il budget
# di ogni ingresso: 33,3 ms - pre - post.
#   experiments/latenza/modelli_budget_per_ingresso/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
R=results/pre_post.jsonl
(
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results; : > $R
scp -q misura.py rpi5:/tmp/misura_pre_risoluzioni.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "taskset -c 0 /home/bench/bench/.venv/bin/python /tmp/misura_pre_risoluzioni.py" | sed "s/^{/{\"passata\": $r, /" | tee -a $R
done
)
cd ../../..
.venv/bin/python tools/figure_sintesi.py experiments/latenza/modelli_latenza_per_ingresso/report --risultati results --board rpi5 \
  --precisioni fp16 --tesi --budget experiments/latenza/modelli_budget_per_ingresso/$R
