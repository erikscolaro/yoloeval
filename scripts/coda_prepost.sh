#!/usr/bin/env bash
# Seguito della coda del 13/11 senza la terza passata di M+ D170 (il Pi si riavvia sotto quel carico): misura pre e
# post-processing, rigenera la tabella del budget con le celle presenti, unisce i report. Log in runs/coda_prepost.log.
set -uo pipefail
cd "$(dirname "$0")/.."
experiments/latenza/pre_post_rpi5/rifai.sh || echo "pre_post fallita"
.venv/bin/python tools/figure_sintesi.py experiments/latenza/griglia_rpi5/report --risultati results --board rpi5 --precisioni fp16 --tesi || echo "figura fallita"
[[ -f experiments/latenza/griglia_rpi5/report_640/latenza_640.csv ]] && mv experiments/latenza/griglia_rpi5/report_640/latenza_640.csv experiments/latenza/griglia_rpi5/report/
[[ -f experiments/latenza/griglia_rpi5/report/A_budget_dettaglio.png ]] && rm -rf experiments/latenza/griglia_rpi5/report_4nov experiments/latenza/griglia_rpi5/report_640
echo "coda finita $(date -Is)"
