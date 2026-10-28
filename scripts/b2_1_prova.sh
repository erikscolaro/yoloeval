#!/bin/bash
# Run di prova B2-1 (griglia discreta di latenza sul PI, fp16 OpenVINO, pesi casuali), 13/10/2026.
# Lanciata il 13/10 da ~/Desktop/yoloeval_prova, spostata in repo il 14/10: risultati in results/, artefatti in
# artifacts/, multirun in multirun/, log e figure in reports/b2_1_prova/.
# Il confronto numerico dopo l'export vuole una cartella di immagini, in I (default il calib del dataset, 511 immagini).
# I entra nella chiave delle celle: con una cartella diversa da quella del 13/10 (data/b2_1_prova/immagini, tolta il
# 28/10) le celle non si saltano e la griglia si rifà.
# Export con hardware=rpi5: la chiave dell'export contiene l'architettura (aarch64), senza il benchmark non lo trova.
# Ingressi in D (default i tre 16:9), es. D=crop128,crop256,crop512 scripts/b2_1_prova.sh
# Solo le figure dai risultati gia' misurati, senza toccare il PI: scripts/b2_1_prova.sh figure
# Per fermarla: pkill -f "b2_1_prova.sh"; pkill -f "run.py -m stage="
set -u
cd ~/Desktop/yoloeval
R=.venv/bin/python
F="$R tools/figure_sintesi.py reports/b2_1_prova --risultati results --board rpi5 --precisioni fp16"
if [ "${1:-}" = figure ]; then $F; exit; fi
A="model=yolo26 model.width=0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1.0 model.depth=D101,D126,D170"
I=${I:-$PWD/data/air-detection/all/images/calib}
O="logging=local dataset.local_path=$I hydra.sweep.dir=multirun/\${now:%Y-%m-%d_%H-%M-%S}"
D=${D:-480p,720p,1080p}
echo "== train $(date)";     $R run.py -m stage=train train=random $A $O
echo "== export $(date)";    $R run.py -m stage=export train=random $A $O deploy=$D quantization=fp16 backend=openvino hardware=rpi5 freq_target=max compute_target=cpu_4
echo "== benchmark $(date)"; $R run.py -m stage=benchmark rep=0,1,2 train=random $A $O deploy=$D \
  quantization=fp16 backend=openvino hardware=rpi5 freq_target=max compute_target=cpu_4 stage.thermal.threshold_c=60
echo "== figure $(date)";    $F
echo "== fine $(date)"
