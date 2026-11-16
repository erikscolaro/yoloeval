#!/usr/bin/env bash
# ncnn sul Pi 5: P e N standard (D126), fp16 e int8, latenza (3 ripetizioni) e mAP sul val misurate sul Pi
# (eval.map_on_target: senza, la mAP degli artefatti portabili si fa sulla workstation, in fp32 x86).
# Riferimento: OpenVINO fp16 (deploy), rimisurato qui per avere la media (benchncnn non da' la mediana) e la mAP
# sul Pi. onnxruntime int8 tolto il 16/11 (deciso con l'utente; su N- fa anche riavviare il Pi).
# Pesi: P 59d8534a e N 9dca8d15, gli stessi della griglia (tools/figure_griglia.py). Artefatti nella cache
# della repo principale (M), risultati in results/. Si lancia dalla root della repo (o del worktree):
#   systemd-run --user --unit=ncnn-rpi5 --working-directory=$PWD experiments/deploy/ncnn_rpi5/rifai.sh
#   tail -f experiments/deploy/ncnn_rpi5/runs/avanzamento.log
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=${M:-$HOME/Desktop/yoloeval}; PY=$M/.venv/bin/python; E=experiments/deploy/ncnn_rpi5
D=$M/data/air-detection/all; mkdir -p $E/runs $E/results
O="target=rpi5_640 artifacts_dir=$M/artifacts dataset.local_path=$D pretrain_dataset.local_path=$D
   results_dir=$PWD/$E/results logging=local retry_failed=true eval.map_on_target=true"
CELLE=("yolo26p fp16 ncnn" "yolo26p int8 ncnn" "yolo26n fp16 ncnn" "yolo26n int8 ncnn"
       "yolo26p fp16 openvino" "yolo26n fp16 openvino")
for c in "${CELLE[@]}"; do read -r m q b <<< "$c"; L=$E/runs/${m}_${q}_$b.log
  echo "$(date '+%H:%M') $c" >> $E/runs/avanzamento.log
  $PY run.py stage=export model=$m quantization=$q backend=$b $O > $L 2>&1 &&
    flock $M/runs/.pi.lock $PY run.py -m stage=benchmark rep=0,1,2 model=$m quantization=$q backend=$b $O \
      hydra.sweep.dir=$E/runs/multirun/${m}_${q}_$b >> $L 2>&1 || echo "  FALLITA, vedi $L" >> $E/runs/avanzamento.log
done
$PY $E/run.py >> $E/runs/avanzamento.log 2>&1
echo "$(date '+%H:%M') fine" >> $E/runs/avanzamento.log
