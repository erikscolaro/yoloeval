#!/usr/bin/env bash
# Coda del 13/11/2026, staccata dalla sessione (systemd-run --user): finisce la griglia con M a 640, misura pre e
# post-processing (B1-2), unisce i report del budget in experiments/latenza/modelli_latenza_per_ingresso/report. Log in runs/coda_13nov.log.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f experiments/latenza/modelli_latenza_per_ingresso/rifai.sh >/dev/null; do sleep 60; done
experiments/latenza/modelli_latenza_per_ingresso/rifai.sh || { echo "griglia fallita"; exit 1; }   # idempotente: completa o salta
experiments/latenza/modelli_pre_post_640/rifai.sh || echo "pre_post fallita"
[[ -f experiments/latenza/modelli_latenza_per_ingresso/report_640/latenza_640.csv ]] && mv experiments/latenza/modelli_latenza_per_ingresso/report_640/latenza_640.csv experiments/latenza/modelli_latenza_per_ingresso/report/
[[ -f experiments/latenza/modelli_latenza_per_ingresso/report/A_budget_dettaglio.png ]] && rm -rf experiments/latenza/modelli_latenza_per_ingresso/report_4nov experiments/latenza/modelli_latenza_per_ingresso/report_640
echo "coda finita $(date -Is)"
