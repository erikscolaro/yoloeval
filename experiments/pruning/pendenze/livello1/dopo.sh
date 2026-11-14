#!/usr/bin/env bash
# Dopo la ricerca di una pendenza (rifai.sh): finetuning, export e misura sul Pi. Con il finetuning corto (10 epoche, default)
# e' il protocollo con cui si giudicano le pendenze (la mAP durante la ricerca non conta: i pesi non seguono le maschere).
#   experiments/pruning/pendenze/livello1/dopo.sh best [lut ...]      # stessi LR, FORZA e SEED di rifai.sh
#   TAG=ft100 FT="" dopo.sh best            # finetuning intero (conf/finetune/default.yaml)
#   TAG=scratch FT="finetune=scratch" dopo.sh best     # da zero, 300 epoche
set -uo pipefail
cd "$(dirname "$0")/../../../.."
D=experiments/pruning/pendenze/livello1
T=${TAG:-ft10}${SEED:+_s$SEED}
for e in "$@"; do
  O="model=yolo26n strategy=pit_latency target=rpi5_640 target.model_budget_ms=28 logging=local strategy.search.epochs=30
     strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=${LR:-0.01} strategy.search.regularizer.strength=${FORZA:-2}
     strategy.search.pit.latency.estimator=$e ${SEED:+strategy.search.seed=$SEED} ${FT-finetune.epochs=10}"
  .venv/bin/python run.py stage=finetune $O > $D/runs/${T}_$e.log 2>&1 || continue
  # il Pi misura una rete alla volta: lucchetto condiviso con scripts/catena_notte.sh
  flock runs/.pi.lock bash -c "for s in export benchmark; do .venv/bin/python run.py stage=\$s $(echo $O) || break; done" > $D/runs/pi_${T}_$e.log 2>&1
done
