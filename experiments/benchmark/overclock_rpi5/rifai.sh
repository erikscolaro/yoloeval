#!/usr/bin/env bash
# Overclock del Pi 5: latenza OpenVINO fp16 di P e N (3 ripetizioni) per frequenza di CPU (arm_freq) e GPU (gpu_freq).
#   A: CPU 2400..3000 a passi di 100, GPU di serie (910)
#   B: GPU 500..1000 a passi di 100, CPU 2400
#   C: GPU 500..1000, CPU 3000 (o la piu' alta stabile in A); prima 3000/1000, e se la GPU non conta niente scansioni
#   D: NCNN fp16 a CPU 3000, GPU di serie
# Ogni configurazione si prova con tryboot: tryboot.txt = config.txt + overclock, valido per un solo avvio. Se il
# Pi non torna entro 5 minuti lo script si ferma: togliere e ridare corrente riavvia con il config.txt normale.
# Alla fine riavvio normale e controllo che arm_freq sia tornato a 2400. Tiene il lock del Pi per tutta la durata.
#   systemd-run --user --unit=overclock-rpi5 --working-directory=$PWD experiments/benchmark/overclock_rpi5/rifai.sh
#   tail -f experiments/benchmark/overclock_rpi5/runs/avanzamento.log
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=${M:-$HOME/Desktop/yoloeval}; PY=$M/.venv/bin/python; E=experiments/benchmark/overclock_rpi5
D=$M/data/air-detection/all; mkdir -p $E/runs $E/results; A=$E/runs/avanzamento.log; C=$E/results/clock.csv
[ -f $C ] || echo "prova,arm_freq,gpu_freq,over_voltage_delta,stato,arm_max_khz,core_hz,v3d_hz,volt_core,throttled" > $C
log() { echo "$(date '+%H:%M:%S') $*" | tee -a $A; }
pi() { ssh -o ConnectTimeout=10 -o BatchMode=yes rpi5 "$@"; }

# shortcut: tensione per gradini fissi, presi dalle guide comuni (+50 mV a 3 GHz); se una frequenza non e' stabile,
# alzare il gradino di quella frequenza e rilanciare solo quella
ov() { if (( $1 <= 2600 )); then echo 0; elif (( $1 <= 2800 )); then echo 25000; else echo 50000; fi; }

attendi() {  # il Pi torna raggiungibile entro 5 minuti?
  local t=$(( $(date +%s) + 300 )); sleep 20
  until pi true 2>/dev/null; do (( $(date +%s) > t )) && return 1; sleep 10; done
}

prova() {  # $1 sigla, $2 arm_freq, $3 gpu_freq (vuoto = di serie)
  local p=$1 arm=$2 gpu=$3 v; v=$(ov $arm); local R=$E/results/${p}_arm${arm}_gpu${gpu:-def}
  if [ -d $R ] && [ "$(ls $R/*.json 2>/dev/null | wc -l)" -ge 6 ]; then log "$p $arm/${gpu:-def}: gia' misurata"; return 0; fi
  pi true 2>/dev/null || attendi || { log "FERMO: Pi non raggiungibile prima di $p $arm/${gpu:-def}"; exit 2; }
  log "$p arm_freq=$arm gpu_freq=${gpu:-di serie} over_voltage_delta=$v: tryboot"
  pi "{ cat /boot/firmware/config.txt; echo '[all]'; echo arm_freq=$arm; echo over_voltage_delta=$v; \
       ${gpu:+echo gpu_freq=$gpu;} } | sudo tee /boot/firmware/tryboot.txt >/dev/null && sudo -n reboot '0 tryboot'" || true
  if ! attendi; then
    log "FERMO: il Pi non torna dopo $p $arm/${gpu:-def}. Togliere e ridare corrente (riparte con config.txt)."
    echo "$p,$arm,$gpu,$v,non riparte,,,,," >> $C; exit 2
  fi
  local st; st=$(pi "cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq; sudo -n vcgencmd measure_clock core; \
       sudo -n vcgencmd measure_clock v3d; sudo -n vcgencmd measure_volts core; sudo -n vcgencmd get_throttled" | \
       sed 's/.*=//' | tr '\n' ',')
  local maxk=${st%%,*}
  if [ "$maxk" != "$((arm * 1000))" ]; then
    log "  non applicata: cpuinfo_max_freq $maxk"; echo "$p,$arm,$gpu,$v,non applicata,$st" >> $C; return 0
  fi
  for m in yolo26p yolo26n; do
    $PY run.py -m stage=benchmark rep=0,1,2 model=$m quantization=fp16 backend=${BK:-openvino} target=rpi5_640 \
      hardware.freq.max.khz=$((arm * 1000)) artifacts_dir=$M/artifacts dataset.local_path=$D pretrain_dataset.local_path=$D \
      results_dir=$PWD/$R logging=local eval.map_on_target=true hydra.sweep.dir=$R/../../runs/multirun/${p}_${arm}_${gpu:-def}_$m \
      >> $E/runs/${p}_arm${arm}_gpu${gpu:-def}.log 2>&1 || log "  $m FALLITA (vedi log)"
  done
  local thr; thr=$(pi "sudo -n vcgencmd get_throttled" | sed 's/.*=//')
  echo "$p,$arm,$gpu,$v,ok,$st" | sed 's/,$//' >> $C
  log "  fatta (throttled dopo la misura: $thr)"
}

log "inizio"
for a in 2400 2500 2600 2700 2800 2900 3000; do prova A $a ""; done
# Prima CPU e GPU al massimo: se la GPU non sposta la latenza oltre l'1% (rumore) le scansioni B e C non servono
prova C ${CMAX:-3000} 1000
[ -d $E/results/C_arm${CMAX:-3000}_gpu1000 ] || { log "FERMO: test CPU e GPU al massimo non misurato"; exit 2; }
if $PY - "$E" "${CMAX:-3000}" <<'PYEOF'
import json, glob, statistics, sys
e, cmax = sys.argv[1:3]
def m(d, mod):
    v = [json.load(open(f))["latency"]["mean_ms"] for f in glob.glob(f"{e}/results/{d}/*.json")
         if json.load(open(f))["config"]["model"]["name"].startswith(mod)]
    return statistics.mean(v)
d = max(abs(m(f"C_arm{cmax}_gpu1000", x) / m(f"A_arm{cmax}_gpudef", x) - 1) for x in ("yolo26p", "yolo26n"))
print(f"effetto della GPU a 1000 MHz: {100 * d:.2f}%")
sys.exit(0 if d < 0.01 else 1)
PYEOF
then
  log "GPU senza effetto (<1%): salto le scansioni B e C"
else
  log "GPU con effetto: scansioni B e C"
  for g in 500 600 700 800 900 1000; do prova B 2400 $g; done
  for g in 500 600 700 800 900; do prova C ${CMAX:-3000} $g; done
fi
# D: NCNN fp16 alla CPU migliore, GPU di serie (experiments/deploy/ncnn_rpi5)
BK=ncnn prova D ${CMAX:-3000} ""
log "riavvio normale con config.txt"
pi "sudo -n reboot" || true
attendi && log "tornato: arm_freq $(pi 'sudo -n vcgencmd get_config arm_freq')" || log "FERMO: il Pi non torna dal riavvio normale"
$PY $E/run.py >> $A 2>&1
log "fine"
