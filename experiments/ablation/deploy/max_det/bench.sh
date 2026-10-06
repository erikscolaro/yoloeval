#!/bin/bash
# Sul Pi: governor performance, attesa termica sotto 50 C (max 5 min), FIFO 80; benchmark_app
# su ogni IR (due passate in ordine opposto, 300 iterazioni dopo 50 di riscaldamento), poi
# pi_bench.py. Il governor torna com'era all'uscita.
D=/home/bench/bench/maxdet
G=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
restore() { echo $G | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null; echo "governor ripristinato: $G"; }
trap restore EXIT
echo performance | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
for i in $(seq 30); do t=$(cat /sys/class/thermal/thermal_zone0/temp); [ $t -lt 50000 ] && break; echo "attesa raffreddamento: $((t/1000)) C"; sleep 10; done
echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), freq $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz, temp $((t/1000)) C"
sudo -n prlimit --rtprio=99:99 --pid $$ >/dev/null 2>&1
BA=/home/bench/bench/.venv/bin/benchmark_app
MDS=$(ls -d $D/models/max_det_*_openvino_model | sed 's/.*max_det_\([0-9]*\)_.*/\1/' | sort -n)
ba() { env OMP_NUM_THREADS=4 chrt -f 80 $BA -m $D/models/max_det_$1_openvino_model/best.xml -d CPU -b 1 -hint none -nstreams 1 -niter $2 -nthreads 4 -report_type no_counters 2>&1; }
: > $D/benchmark_app.csv
echo "pass,max_det,median_ms,mean_ms,max_ms" >> $D/benchmark_app.csv
for pass in 1 2; do
  [ $pass = 2 ] && MDS=$(echo "$MDS" | sort -rn)
  for md in $MDS; do
    ba $md 50 >/dev/null
    out=$(ba $md 300)
    v() { echo "$out" | grep -m1 "$1" | awk '{print $(NF-1)}'; }
    echo "$pass,$md,$(v Median:),$(v Average:),$(v Max:)" | tee -a $D/benchmark_app.csv
  done
done
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    chrt -f 80 /home/bench/bench/.venv/bin/python -u $D/pi_bench.py $D/images $D/models $D/timings.csv
echo "temp fine $(( $(cat /sys/class/thermal/thermal_zone0/temp)/1000 )) C"
