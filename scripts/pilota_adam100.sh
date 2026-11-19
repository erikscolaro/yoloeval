#!/usr/bin/env bash
# Pilota: come pilota_adam.sh (N-, 30,3 ms, best, forza 2, Adam 5e-4, testa chiusa) ma con 100 epoche di ricerca
# invece di 30: PIT ferma la ricerca con early stopping a pazienza 20, quindi cerca molto piu a lungo di noi.
# Solo la ricerca. Parte quando finisce pilota-adam.
#   systemd-run --user --unit=pilota-adam100 --working-directory=$PWD scripts/pilota_adam100.sh
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active -q pilota-adam; do sleep 120; done
echo "[$(date '+%F %T')] pilota Adam 5e-4, 100 epoche: ricerca da N-" >> runs/catena_notte.log
.venv/bin/python run.py stage=search model=yolo26 model.width=0.25 model.depth=D101 strategy=pit_latency target=rpi5_640 \
  logging=local strategy.search.nas.optimizer=Adam strategy.search.nas.lr0=0.0005 strategy.search.regularizer.strength=2 \
  strategy.search.pit.latency.estimator=best strategy.search.workers=8 strategy.search.epochs=100 > runs/pilota_adam100_search.log 2>&1
echo "[$(date '+%F %T')] pilota Adam 5e-4, 100 epoche: FINE (exit $?)" >> runs/catena_notte.log
