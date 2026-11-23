#!/usr/bin/env bash
# Come rifai.sh, ma con crop piu' grandi (fino a 1024 px) e due controlli (384, 512, 720): AREA con potenze di 2 o
# no? Risultati in results/crop_resize_grandi.jsonl. Un core, governor performance, 3 passate, lock del Pi.
#   experiments/latenza/track_costo_ritaglio/rifai_grandi.sh
set -euo pipefail
cd "$(dirname "$0")"
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results
scp -q misura.py rpi5:/tmp/misura_crop_resize.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "taskset -c 0 /home/bench/bench/.venv/bin/python /tmp/misura_crop_resize.py 500 384,512,640,720,768,1000,1024" \
    | sed "s/^{/{\"passata\": $r, /" | tee -a results/crop_resize_grandi.jsonl
done
