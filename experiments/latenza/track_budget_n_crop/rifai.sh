#!/usr/bin/env bash
# Costo del crop tracker fuori dal modello sul Pi 5 (misura.py), N = 1, 2, 4, 8 crop 256x256 da un fotogramma 1080p,
# governor performance, un core, 3 passate. Copia src/tracking.py e src/crop_tracker.py cosi' come sono nel repo.
# Risultati in results/crop_kalman.jsonl, una riga per N e passata. Serve per il margine di ../track_modelli_batch_crop256.
#   experiments/latenza/track_budget_n_crop/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
R=/tmp/crop_kalman
ssh rpi5 "rm -rf $R && mkdir -p $R/yoloeval_src && touch $R/yoloeval_src/__init__.py"
scp -q misura.py rpi5:$R/
scp -q ../../../src/tracking.py ../../../src/crop_tracker.py rpi5:$R/yoloeval_src/
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
for r in 0 1 2; do
  ssh rpi5 "cd $R && taskset -c 0 /home/bench/bench/.venv/bin/python misura.py" | tee -a results/crop_kalman.jsonl
done
