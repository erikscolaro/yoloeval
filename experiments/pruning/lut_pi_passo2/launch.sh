#!/bin/bash
# Full LUT run, detached from the shell (setsid nohup): survives Claude/terminal closing.
# After an error it restarts (run.py resumes from results/raw.csv), at most 20 times.
#   systemd-run --user --unit=lut-pi-passo2 --working-directory=$PWD ./launch.sh     log: runs/run.log
#   stop: systemctl --user stop lut-pi-passo2   (SIGTERM to all: run.py restores the board, no restart)
cd "$(dirname "$0")"
mkdir -p runs
echo $$ > runs/launch.pid
trap 'echo "$(date -Is) STOP richiesto" >> runs/run.log; exit 0' INT TERM   # no restart after a stop
for i in $(seq 1 20); do
  ITERS=200 ../../../.venv/bin/python run.py run >> runs/run.log 2>&1 && exit 0
  echo "$(date -Is) RIAVVIO $i dopo errore, fra 60 s" >> runs/run.log
  sleep 60
done
echo "$(date -Is) troppi riavvii, mi fermo" >> runs/run.log
