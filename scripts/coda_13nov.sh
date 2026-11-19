#!/usr/bin/env bash
# Coda del 13/11/2026, staccata dalla sessione (systemd-run --user): finisce la griglia con M a 640, misura pre e
# post-processing (B1-2), unisce i report del budget in experiments/latenza/griglia_rpi5/report. Log in runs/coda_13nov.log.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f experiments/latenza/griglia_rpi5/rifai.sh >/dev/null; do sleep 60; done
experiments/latenza/griglia_rpi5/rifai.sh || { echo "griglia fallita"; exit 1; }   # idempotente: completa o salta
experiments/latenza/pre_post_rpi5/rifai.sh || echo "pre_post fallita"
[[ -f experiments/latenza/griglia_rpi5/report_640/latenza_640.csv ]] && mv experiments/latenza/griglia_rpi5/report_640/latenza_640.csv experiments/latenza/griglia_rpi5/report/
[[ -f experiments/latenza/griglia_rpi5/report/A_budget_dettaglio.png ]] && rm -rf experiments/latenza/griglia_rpi5/report_4nov experiments/latenza/griglia_rpi5/report_640
echo "coda finita $(date -Is)"
