#!/usr/bin/env bash
# Ricerca da PN- (w0.1875 D101, 30,3 ms) con la ricetta SGD di catena_pn.sh, ma con il lr delle maschere che scende
# (coseno da 0,01 a 0,0001: nas.lrf=0.01) e la rete presa all'ultima epoca (pit.select=last, yolopit >= 5176375, allora ramo esplora/maschere-last)
# invece della migliore dentro il budget. Poi da zero 300 epoche e misura sul Pi.
# Debug: tutti i checkpoint (save_period=1) della ricerca (con le maschere) e del training da zero.
#   systemd-run --user --unit=maschere-lr --working-directory=$PWD experiments/pruning/maschere_lr/rifai.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=${M:-$HOME/Desktop/yoloeval}; PY=$M/.venv/bin/python; E=experiments/pruning/maschere_lr; L=$E/runs/diario.log
export PYTHONPATH=${YOLOPIT:-$HOME/Desktop/yolopit}   # yolopit >= 5176375 (pit.select, pit.temperature)
D=$M/data/air-detection/all
passo() { echo "[$(date '+%F %T')] $*" | tee -a $L; }
X="model=yolo26 model.width=0.1875 model.depth=D101 strategy=pit_latency target=rpi5_640 logging=local
   strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=0.01 strategy.search.regularizer.strength=2
   strategy.search.pit.latency.estimator=best
   strategy.search.nas.lrf=0.01 strategy.search.nas.cos_lr=true strategy.search.pit.select=last ~strategy.search.pit.temperature
   +strategy.search.save_period=1 artifacts_dir=$M/artifacts dataset.local_path=$D pretrain_dataset.local_path=$D target.lut_dir=$M/artifacts/lut/rpi5"
X=$(echo $X)   # una riga sola: bash -c in flock non deve vedere a capo
$PY -c "from yolopit.config import DEFAULTS; assert 'temperature' in DEFAULTS['pit'], 'yolopit senza pit.temperature'" || { passo "yolopit sbagliato"; exit 2; }
passo "ricerca (lr maschere a coseno, select=last)"
$PY run.py stage=search $X > $E/runs/search.log 2>&1 || { passo "ricerca FALLITA"; exit 1; }
passo "da zero 300 epoche"
$PY run.py stage=finetune $X finetune=scratch +finetune.save_period=1 > $E/runs/scratch.log 2>&1 || { passo "da zero FALLITO"; exit 1; }
passo "Pi"
flock $M/runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $X finetune=scratch +finetune.save_period=1 results_dir=$PWD/$E/results || break; done" > $E/runs/pi.log 2>&1
passo "FINE (exit $?)"
