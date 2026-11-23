#!/usr/bin/env bash
# Tempo di caricamento e di cambio fra i due modelli del tracker sul Pi 5 (misura.py): SCAN PN- 640x384, TRACK S- 256x256.
# OpenVINO fp16, 4 thread, governor performance, lock del Pi. Caricamento: 5 processi a freddo (cache dei file svuotata)
# e 5 a caldo per modello, piu' compile_model con la cache di OpenVINO (1 miss e 4 hit). Cambio: 3 passate.
# Risultati in results/carico_cambio.jsonl.
#   experiments/latenza/modelli_carico_e_cambio/rifai.sh
set -euo pipefail
cd "$(dirname "$0")"
EXP=../../../artifacts/exports
SCAN=yolo26w0.1875_D101_5d00ca6b_640x384_fp16_openvino_aarch64/yolo26w0.1875_D101_fp16_openvino_model/best.xml
TRACK=yolo26w0.5_D101_834ac92a_crop256_b1_fp16_openvino_aarch64/yolo26w0.5_D101_fp16_openvino_model/best.xml
exec 9>../../../runs/.pi.lock; flock 9
mkdir -p results; : > results/carico_cambio.jsonl
for m in ${SCAN%%/*} ${TRACK%%/*}; do rsync -aq $EXP/$m/ rpi5:/home/bench/bench/exports/$m/; done
scp -q misura.py rpi5:/tmp/misura_carico.py
G=$(ssh rpi5 cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)
ssh rpi5 'echo performance | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'
trap "ssh rpi5 'echo $G | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null'" EXIT
R="sudo -n prlimit --rtprio=99:99 --pid \$\$ >/dev/null 2>&1; cd /home/bench/bench/exports; env OMP_NUM_THREADS=4 chrt -f 80 taskset -c 0-3 /home/bench/bench/.venv/bin/python /tmp/misura_carico.py"
agg() { sed "s/^{/{\"prova\": \"$1\", /" | tee -a results/carico_cambio.jsonl; }
for nome in scan track; do
  [[ $nome == scan ]] && X=$SCAN S=384x640 || { X=$TRACK; S=256x256; }
  for i in 1 2 3 4 5; do
    ssh rpi5 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'
    ssh rpi5 "$R carico $X $S" | agg "${nome}_freddo"
  done
  for i in 1 2 3 4 5; do ssh rpi5 "$R carico $X $S" | agg "${nome}_caldo"; done
  ssh rpi5 'rm -rf /tmp/ov_cache'
  for i in 1 2 3 4 5; do ssh rpi5 "$R carico $X $S /tmp/ov_cache" | agg "${nome}_cache_$([[ $i == 1 ]] && echo miss || echo hit)"; done
done
for i in 1 2 3; do ssh rpi5 "$R cambio $SCAN 384x640 $TRACK 256x256" | agg "cambio_$i"; done
