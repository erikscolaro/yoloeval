#!/usr/bin/env bash
# Demo sul Raspberry Pi 5 (alias ssh rpi5): tools/demo_pi.py, Kalman con crop tarato (P- in SCAN a 640, S- su crop 256),
# OpenVINO fp16 su 4 core, video con il lato lungo a 640, nessun salvataggio.
#   scripts/demo_pi.sh avvia [MODO] [k=v ...]         finestra sulla workstation (tools/vedi_demo_pi.py): il Pi gira senza
#                                                     desktop e manda solo i metadati, la workstation disegna sugli stessi video
#   scripts/demo_pi.sh avvia-sul-pi [MODO] [k=v ...]  finestra sullo schermo HDMI del Pi
#   scripts/demo_pi.sh prepara [k=v ...]              copia tutto senza avviare; poi dal Pi, senza workstation: icone
#                                                     "Demo ..." o ~/bench/demo/avvia.sh MODO
#   scripts/demo_pi.sh fine                           chiude la demo e spegne il desktop: il Pi torna libero per i benchmark
#   scripts/demo_pi.sh oc on|off|stato                overclock per un avvio (scripts/rpi5_oc.sh)
# MODO: f2 (default, il deploy), f1 (prova 445), e f2-maschera, f1-maschera (in SCAN solo box sopra la F1 per classe di P-).
# Sempre rescan=0 (niente SCAN periodico) e vote=0.95 (classe stabile); k=v cambia un parametro (es. min_hits=2).
# Ogni avvio copia sul Pi programma, modelli, parametri e i video nuovi (~/bench/demo, ~/Videos/demo e demo640) e tiene il
# lock del Pi (runs/.pi.lock) finche' la demo e' aperta. Tasti nella finestra: spazio pausa, n video dopo, m maschera, q esce.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SCAN=artifacts/exports/yolo26w0.125_D101_e7d00a5f_640_fp16_openvino_aarch64/yolo26w0.125_D101_fp16_openvino_model
TRACK=artifacts/exports/yolo26w0.5_D101_21d2b4bd_crop256_fp16_openvino_aarch64/yolo26w0.5_D101_fp16_openvino_model
D=experiments/tracking/taratura_kalman_crop/results
declare -A PRESET=([f2]=$D/kalman_crop_f2_val.json [f1]=$D/kalman_crop_f1_val_prova445.json)   # f1: prova 445, ricerca in corso
SOGLIE=artifacts/weights/yolo26w0.125_D101_air_detection_all_e300_e7d00a5f/soglie_f1_val.json   # maschera: F1 per classe di P-
R=/home/bench/bench/demo
pi() { ssh -o ConnectTimeout=10 -o BatchMode=yes rpi5 "$@"; }

prepara() {  # prepara [k=v ...]: i tre parametri_<modo>.json (f2, f2-maschera, f1), avvia.sh, le icone, modelli e video
  [[ -f $SCAN/best.xml ]] || { echo "manca l'export di P- a 640: $SCAN" >&2; exit 1; }
  [[ -f $TRACK/best.xml ]] || { echo "manca l'export di S- a crop 256: python run.py stage=export model=yolo26 model.width=0.5 model.depth=D101 deploy=crop256 quantization=fp16 backend=openvino hardware=rpi5" >&2; exit 1; }
  local t; t=$(mktemp -d); trap 'rm -rf "$t"' RETURN
  mkdir -p "$t/src" "$t/video"
  local modo pr ma
  for modo in f2 f2-maschera f1 f1-maschera; do
    pr=${modo%%-*}; ma=$([[ $modo == *maschera ]] && echo "$SOGLIE" || echo no)
    "$PY" -c 'import json, sys; from dataclasses import asdict; from tools.predict_video import applica_par, preset
k, t, c = applica_par(*preset(sys.argv[1]), ["rescan=0", "vote=0.95", *sys.argv[4:]])
m = {c: v[0] for c, v in json.load(open(sys.argv[3]))["per_classe"].items()} if sys.argv[3] != "no" else None
json.dump({"kalman": asdict(k), "track": t, "crop": c, "maschera": m}, open(sys.argv[2], "w"), indent=1)' \
      "${PRESET[$pr]}" "$t/parametri_$modo.json" "$ma" "$@"
    printf '[Desktop Entry]\nType=Application\nName=Demo %s\nComment=Tracker Kalman con crop, %s\nExec=%s/avvia.sh %s\nIcon=video-display\nTerminal=false\nCategories=AudioVideo;\n' \
      "$modo" "$modo" "$R" "$modo" > "$t/demo-$modo.desktop"
  done
  printf '[Desktop Entry]\nType=Application\nName=Demo fine\nComment=Chiude la demo e spegne il desktop\nExec=%s/avvia.sh fine\nIcon=system-shutdown\nTerminal=false\nCategories=AudioVideo;\n' "$R" > "$t/demo-fine.desktop"
  cp tools/demo_pi_avvia.sh "$t/avvia.sh"
  cp tools/demo_pi.py "$t/"; cp src/tracking.py src/crop_tracker.py "$t/src/"; : > "$t/src/__init__.py"   # senza l'__init__ della repo (omegaconf)
  for d in data/tracking/val/V_*/; do v=$(basename "$d"); v=${v%_s*}.mp4; [[ -f data/tracking/_mp4/$v ]] && cp --update=none "data/tracking/_mp4/$v" "$t/video/"; done
  cp data/tracking/video/*/*.mp4 "$t/video/" 2>/dev/null || true   # ogni cartella di video (es. dvids: droni di val e test di VEDETTA)
  pi "mkdir -p $R/modelli/scan $R/modelli/track ~/Videos/demo"
  rsync -a "$t/demo_pi.py" "$t/avvia.sh" "$t"/parametri_*.json "rpi5:$R/"; rsync -a --delete "$t/src" "rpi5:$R/"
  pi "mkdir -p ~/.local/share/applications ~/Desktop"
  rsync -a "$t"/demo-*.desktop rpi5:.local/share/applications/; rsync -a "$t"/demo-*.desktop rpi5:Desktop/
  pi 'chmod +x ~/Desktop/demo-*.desktop; for f in ~/Desktop/demo-*.desktop; do gio set "$f" metadata::trusted true 2>/dev/null || true; done' 
  rsync -a --delete "$SCAN/" "rpi5:$R/modelli/scan/"; rsync -a --delete "$TRACK/" "rpi5:$R/modelli/track/"
  rsync -a "$t/video/" rpi5:Videos/demo/
  mkdir -p data/tracking/_demo640   # gli stessi video con il lato lungo a 640: niente riduzione ne' decodifica pesante sul Pi
  for f in "$t"/video/*.mp4; do o=data/tracking/_demo640/$(basename "$f"); [[ -f $o ]] && continue
    ffmpeg -nostdin -loglevel error -i "$f" -vf "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2" \
      -an -c:v libx264 -preset veryfast -crf 20 -threads 4 "$o"; done
  pi "mkdir -p ~/Videos/demo640"; rsync -a data/tracking/_demo640/ rpi5:Videos/demo640/
  echo "pronto: $(ls "$t/video" | wc -l) video in ~/Videos/demo"
}

case ${1:-} in
  avvia)
    shift; modo=f2
    [[ ${1:-} == f2 || ${1:-} == f2-maschera || ${1:-} == f1 || ${1:-} == f1-maschera ]] && { modo=$1; shift; }
    prepara "$@"
    exec flock runs/.pi.lock "$PY" tools/vedi_demo_pi.py --modo "$modo" ;;   # finestra qui, Pi senza desktop
  avvia-sul-pi)   # finestra sullo schermo HDMI del Pi
    shift; modo=f2
    [[ ${1:-} == f2 || ${1:-} == f2-maschera || ${1:-} == f1 || ${1:-} == f1-maschera ]] && { modo=$1; shift; }
    prepara "$@"
    exec flock runs/.pi.lock ssh -o BatchMode=yes rpi5 "$R/avvia.sh $modo" ;;
  prepara) shift; prepara "$@" ;;
  fine)
    pi 'pkill -f "[d]emo_pi.py"; sudo -n systemctl isolate multi-user.target; sleep 3
        pgrep -u bench -x labwc >/dev/null && { echo "desktop ancora attivo" >&2; exit 1; }; echo "Pi libero, desktop spento"' ;;
  oc) scripts/rpi5_oc.sh "${2:-stato}" ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 1 ;;
esac
