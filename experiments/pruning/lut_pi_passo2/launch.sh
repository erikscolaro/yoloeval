#!/bin/bash
# Full LUT run, detached from the shell (setsid nohup): survives Claude/terminal closing.
# After an error it restarts (run.py resumes from results/raw.csv), at most 20 times.
#   setsid nohup ./launch.sh > /dev/null 2>&1 &     log: runs/run.log, pid: runs/launch.pid
cd "$(dirname "$0")"
mkdir -p runs
echo $$ > runs/launch.pid
for i in $(seq 1 20); do
  ITERS=200 ../../../.venv/bin/python run.py run >> runs/run.log 2>&1 && exit 0
  echo "$(date -Is) RIAVVIO $i dopo errore, fra 60 s" >> runs/run.log
  sleep 60
done
echo "$(date -Is) troppi riavvii, mi fermo" >> runs/run.log
