#!/usr/bin/env bash
# ncnn fp32 sul Pi 5, P e N (D126): latenza con benchncnn_fp32 (fp16 spento, vedi build_fp32.sh). La mAP, se c e, e quella della workstation (x86 fp32): il
# pacchetto ncnn Python non e' lo stesso runtime. Prima volta: ./build_fp32.sh. Dalla root della repo:
#   experiments/deploy/ncnn_rpi5/rifai_fp32.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=${M:-$HOME/Desktop/yoloeval}; PY=$M/.venv/bin/python; E=experiments/deploy/ncnn_rpi5
D=$M/data/air-detection/all; mkdir -p $E/runs $E/results
O="target=rpi5_640 artifacts_dir=$M/artifacts dataset.local_path=$D pretrain_dataset.local_path=$D
   results_dir=$PWD/$E/results logging=local retry_failed=true eval.map_on_target=false"
for m in yolo26p yolo26n; do L=$E/runs/${m}_fp32_ncnn.log
  $PY run.py stage=export model=$m quantization=fp32 backend=ncnn $O > $L 2>&1 &&
    flock $M/runs/.pi.lock $PY run.py -m stage=benchmark rep=0,1,2 model=$m quantization=fp32 backend=ncnn $O \
      backend.benchmark.tool=benchncnn_fp32 "backend.benchmark.cmd='benchncnn_fp32 {iters} {threads} 0 -1 0 param={param} shape={shape}'" \
      hydra.sweep.dir=$E/runs/multirun/${m}_fp32_ncnn >> $L 2>&1 || echo "FALLITA $m, vedi $L"
done
$PY $E/run.py
