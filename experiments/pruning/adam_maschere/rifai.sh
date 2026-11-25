#!/bin/bash
# Maschere su Adam: ricerca a 30 e 100 epoche, poi da zero 300 epoche e Pi 5 (gli script stanno in scripts/).
#   bash experiments/pruning/adam_maschere/rifai.sh      lungo (GPU per giorni e Pi): meglio staccato con systemd-run
set -euo pipefail
cd "$(dirname "$0")/../../.."
scripts/pilota_adam.sh       # aspetta la fine di runs/notte_ft100.log: senza quel log resta fermo
scripts/pilota_adam100.sh
scripts/pilota_adam_scratch.sh
