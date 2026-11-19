#!/usr/bin/env bash
# Pilota: la ricerca finale (N-, 30,3 ms, pendenza best, forza 2, testa chiusa) con le maschere su Adam 5e-4 come nel
# notebook di PLiNIO, invece di SGD 0,01. Solo la ricerca: si guarda se la curva della latenza e' piu' liscia e dove
# si ferma. Parte quando il finetuning da 100 epoche della catena libera un posto sulla GPU.
#   systemd-run --user --unit=pilota-adam --working-directory=$PWD scripts/pilota_adam.sh
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "fine-tuning completato\|Traceback" runs/notte_ft100.log 2>/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] pilota Adam 5e-4: ricerca da N-" >> runs/catena_notte.log
.venv/bin/python run.py stage=search model=yolo26 model.width=0.25 model.depth=D101 strategy=pit_latency target=rpi5_640 \
  logging=local strategy.search.nas.optimizer=Adam strategy.search.nas.lr0=0.0005 strategy.search.regularizer.strength=2 \
  strategy.search.pit.latency.estimator=best strategy.search.workers=8 > runs/pilota_adam_search.log 2>&1
echo "[$(date '+%F %T')] pilota Adam 5e-4: FINE (exit $?)" >> runs/catena_notte.log
