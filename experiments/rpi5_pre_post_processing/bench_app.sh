#!/bin/bash
# Sul Pi: benchmark_app sui due IR (max_det 300 e 50), stesso setup di bench.sh, per separare
# il costo del TopK dall'overhead di Python. Due passate alternate da 300 iterazioni.
D=/home/bench/bench/prepost
G=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
restore() { echo $G | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null; echo "governor ripristinato: $G"; }
trap restore EXIT
echo performance | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
for i in $(seq 30); do t=$(cat /sys/class/thermal/thermal_zone0/temp); [ $t -lt 50000 ] && break; echo "attesa raffreddamento: $((t/1000)) C"; sleep 10; done
echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), temp $((t/1000)) C"
sudo -n prlimit --rtprio=99:99 --pid $$ >/dev/null 2>&1
BA=/home/bench/bench/.venv/bin/benchmark_app
run() { env OMP_NUM_THREADS=4 chrt -f 80 $BA -m $1 -d CPU -b 1 -hint none -nstreams 1 -niter $2 -nthreads 4 -report_type no_counters 2>&1; }
for pass in 1 2; do
  for md in 300 50; do
    m=$D/models/max_det_${md}_openvino_model/best.xml
    run $m 50 >/dev/null
    echo "== passata $pass max_det $md"; run $m 300 | grep -E "Median|Average|Max:"
  done
done
echo "temp fine $(( $(cat /sys/class/thermal/thermal_zone0/temp)/1000 )) C"
