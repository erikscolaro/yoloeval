#!/usr/bin/env bash
# Sul Pi, senza la workstation: avvia la demo (tools/demo_pi.py) con i parametri gia' copiati da scripts/demo_pi.sh.
#   ~/bench/demo/avvia.sh f2|f2-maschera|f1|f1-maschera [pieni|meta]   da console, da ssh o dalle icone "Demo ..." (desktop e menu)
#                                     video a 640 (~/Videos/demo640); pieni: alla risoluzione originale (~/Videos/demo)
#                                     meta: niente finestra ne' desktop, metadati su stdout (tools/vedi_demo_pi.py)
#   ~/bench/demo/avvia.sh fine                chiude la demo e spegne il desktop (il Pi torna libero per i benchmark)
# Da console o ssh accende il desktop se e' spento. Non tiene il lock dei benchmark della workstation (runs/.pi.lock).
set -eu
cd "$(dirname "$0")"
m=${1:-f2}
if [[ $m == fine ]]; then
  pkill -f "[d]emo_pi.py" || true
  sudo -n systemctl isolate multi-user.target
  exit 0
fi
[[ -f parametri_$m.json ]] || { echo "manca parametri_$m.json: dalla workstation, scripts/demo_pi.sh avvia" >&2; exit 1; }
if [[ ${2:-} == meta ]]; then
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo -n tee "$f" >/dev/null || true; done
  pkill -f "[d]emo_pi.py" || true
  exec ~/bench/.venv/bin/python demo_pi.py --scan modelli/scan/best.xml --track modelli/track/best.xml \
    --parametri "parametri_$m.json" --video ~/Videos/demo640 --meta
fi
if [[ -z ${DISPLAY:-} ]]; then
  systemctl is-active -q lightdm || sudo -n systemctl isolate graphical.target
  for _ in $(seq 40); do pgrep -u "$USER" -x Xwayland >/dev/null && break; sleep 1; done
  export DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority
fi
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo -n tee "$f" >/dev/null || true; done
pkill -f "[d]emo_pi.py" || true   # una demo sola
QT_QPA_PLATFORM=xcb exec ~/bench/.venv/bin/python demo_pi.py --scan modelli/scan/best.xml --track modelli/track/best.xml \
  --parametri "parametri_$m.json" --video ~/Videos/demo$([[ ${2:-} == pieni ]] || echo 640)
