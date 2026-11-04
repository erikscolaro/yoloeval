#!/bin/bash
# Training della griglia larghezza × profondità (31/10/2026): le 8 larghezze fino a 0.75 (SM) per le 3 profondità, ricetta default (quella del pico
# 59d8534a), 300 epoche, seed 0. M (1.0) no.
# Pico, nano e small D126 ci sono già, allenati come yolo26p/n/s (59d8534a, 9dca8d15, 9ada3b33): non si rifanno.
# Fino al nano due alla volta (il collo è il dataloader, la GPU resta mezza vuota), sopra uno alla volta.
# Un log per cella in logs/train_yolo26w<larghezza>_<profondità>_<data>.log.
set -u
cd ~/Desktop/yoloeval
t() {
  L=logs/train_yolo26w$1_$2_$(date +%Y%m%d_%H%M).log
  .venv/bin/python run.py stage=train model=yolo26 model.width=$1 model.depth=$2 train=default train.epochs=300 > $L 2>&1 < /dev/null
  echo "$(date '+%d/%m %H:%M') fine $1 $2 (exit $?)"
}
export -f t
celle() {
  for w in $1; do for d in D101 D126 D170; do
    case "$w $d" in "0.125 D126"|"0.25 D126"|"0.5 D126") ;; *) echo "$w $d" ;; esac
  done; done
}
echo "$(date '+%d/%m %H:%M') inizio"
celle "0.0625 0.09375 0.125 0.1875 0.25" | xargs -P 2 -L 1 bash -c 't $0 $1'
celle "0.375 0.5 0.75" | xargs -P 1 -L 1 bash -c 't $0 $1'
echo "$(date '+%d/%m %H:%M') fine"
