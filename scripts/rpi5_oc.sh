#!/usr/bin/env bash
# Overclock del Pi 5 per un solo avvio, con i parametri del profilo freq.oc (conf/hardware/rpi5.yaml):
# tryboot.txt = config.txt + arm_freq, gpu_freq, over_voltage_delta, poi `reboot '0 tryboot'`. config.txt resta
# intatto: il riavvio dopo torna a 2,4 GHz, e se il Pi non ripartisse basta togliere e ridare corrente.
# Come experiments/benchmark/overclock_rpi5/rifai.sh. Tiene il lock del Pi durante il riavvio.
#   scripts/rpi5_oc.sh on       # riavvia in overclock, poi: run.py ... freq_target=oc
#   scripts/rpi5_oc.sh off      # riavvio normale
#   scripts/rpi5_oc.sh stato    # frequenza massima, clock, tensione, throttling
set -euo pipefail
cd "$(dirname "$0")/.."
pi() { ssh -o ConnectTimeout=10 -o BatchMode=yes rpi5 "$@"; }
attendi() {  # il Pi torna raggiungibile entro 5 minuti?
  local t=$(( $(date +%s) + 300 )); sleep 20
  until pi true 2>/dev/null; do (( $(date +%s) > t )) && { echo "il Pi non torna: togliere e ridare corrente" >&2; exit 2; }; sleep 10; done
}
stato() {
  pi "echo max_khz=\$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq); sudo -n vcgencmd measure_clock arm; \
      sudo -n vcgencmd measure_volts core; sudo -n vcgencmd get_throttled"
}
case ${1:-} in
  on)
    B=$(.venv/bin/python -c "import yaml; b = yaml.safe_load(open('conf/hardware/rpi5.yaml'))['freq']['oc']['boot']
print(*(f'{k}={v}' for k, v in b.items()))")
    echo "tryboot con: $B"
    flock runs/.pi.lock bash -c "
      ssh -o ConnectTimeout=10 -o BatchMode=yes rpi5 \"{ cat /boot/firmware/config.txt; echo '[all]'; printf '%s\n' $B; } \
        | sudo tee /boot/firmware/tryboot.txt >/dev/null && sudo -n reboot '0 tryboot'\" || true
      $(declare -f pi attendi); attendi"
    stato
    want=$(( $(sed -n 's/.*arm_freq=\([0-9]*\).*/\1/p' <<< "$B") * 1000 ))
    [[ $(pi cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq) == "$want" ]] \
      || { echo "overclock non applicato" >&2; exit 1; } ;;
  off)
    flock runs/.pi.lock bash -c "ssh -o ConnectTimeout=10 -o BatchMode=yes rpi5 'sudo -n reboot' || true; $(declare -f pi attendi); attendi"
    stato ;;
  stato) stato ;;
  *) sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 1 ;;
esac
