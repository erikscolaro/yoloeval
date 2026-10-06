#!/bin/bash
# Sul Pi: governor performance, attesa termica sotto 50 C (max 5 min), FIFO 80, poi pi_bench.py.
# Il governor torna com'era all'uscita.
D=/home/bench/bench/prepost
G=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
restore() { echo $G | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null; echo "governor ripristinato: $G"; }
trap restore EXIT
echo performance | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
for i in $(seq 30); do t=$(cat /sys/class/thermal/thermal_zone0/temp); [ $t -lt 50000 ] && break; echo "attesa raffreddamento: $((t/1000)) C"; sleep 10; done
echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), freq $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz, temp $((t/1000)) C"
sudo -n prlimit --rtprio=99:99 --pid $$ >/dev/null 2>&1
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    chrt -f 80 /home/bench/bench/.venv/bin/python -u $D/pi_bench.py $D/images $D/models $D/timings.csv
echo "temp fine $(( $(cat /sys/class/thermal/thermal_zone0/temp)/1000 )) C"
