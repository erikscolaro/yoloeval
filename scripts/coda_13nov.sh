#!/usr/bin/env bash
# Coda del 13/11/2026, staccata dalla sessione (systemd-run --user): finisce la griglia con M a 640, misura pre e
# post-processing (B1-2), unisce i report del budget in reports/budget_latenza_rpi5. Log in runs/coda_13nov.log.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f experiments/latenza/griglia_rpi5/rifai.sh >/dev/null; do sleep 60; done
experiments/latenza/griglia_rpi5/rifai.sh || { echo "griglia fallita"; exit 1; }   # idempotente: completa o salta
experiments/latenza/pre_post_rpi5/rifai.sh || echo "pre_post fallita"
[[ -f reports/budget_latenza_rpi5_640/latenza_640.csv ]] && mv reports/budget_latenza_rpi5_640/latenza_640.csv reports/budget_latenza_rpi5/
[[ -f reports/budget_latenza_rpi5/A_budget_dettaglio.png ]] && rm -rf reports/budget_latenza_rpi5_4nov reports/budget_latenza_rpi5_640
echo "coda finita $(date -Is)"
