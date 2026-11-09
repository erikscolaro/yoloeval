"""LUT progress: overall % and, per family, measurements done vs planned (results/raw.csv vs points()).
    watch -n 30 ../../../.venv/bin/python progress.py
"""
import csv
import subprocess
from collections import Counter
from datetime import datetime

from run import OUT, REPEATS, RES, points

plan = Counter(f for f, *_ in points())
rows = [r for r in csv.DictReader(open(RES / "raw.csv")) if r["remeasured"] == "0" and int(r["repeat"]) < REPEATS]   # extra repeats of unstable points not counted
done = Counter(r["family"] for r in rows)
seen = Counter(f for f, *_ in {(r["family"], r["c_in"], r["c_out"]) for r in rows})

unit = subprocess.run(["systemctl", "--user", "is-active", "lut-pi-passo2"], capture_output=True, text=True).stdout.strip()
try:
    pi = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "rpi5", "cat /sys/class/thermal/thermal_zone0/temp "
                         "/sys/class/hwmon/hwmon2/fan1_input; pgrep -f bench_dump.py >/dev/null && echo 1 || echo 0"],
                        capture_output=True, text=True, timeout=15).stdout.split()
    temp, rpm, busy = int(pi[0]) / 1000, pi[1], pi[2] == "1"
except (subprocess.TimeoutExpired, IndexError, ValueError):
    temp = rpm = busy = None
log = (OUT / "runs" / "run.log").read_text().splitlines()[-1:] if (OUT / "runs" / "run.log").exists() else [""]
state = ("FERMA (unit " + unit + ")" if unit != "active" else "Pi non raggiungibile" if busy is None
         else "COMPUTING" if busy else "RESTING (pausa termica)" if log[0].startswith("caldo") else "RESTING (fra due misure)")
print(f"stato: {state}" + (f"   Pi ora: {temp:.1f} °C, ventola {rpm} rpm" if temp is not None else ""))

tot, n = sum(plan.values()) * REPEATS, len(rows)
t = [datetime.fromisoformat(r["timestamp"]) for r in (rows[-200], rows[-1])] if len(rows) > 200 else []
rate = 199 / max((t[1] - t[0]).total_seconds(), 1) if t else 0   # pace of the last 200 measures
eta = f", fine stimata {datetime.fromtimestamp(t[1].timestamp() + (tot - n) / rate):%a %d/%m %H:%M}" if rate else ""
print(f"LUT: {n} / {tot} misure = {100 * n / tot:.2f}%{eta}")
if rows:
    print(f"ultima: {rows[-1]['timestamp'][11:19]}  {float(rows[-1]['temp_c']):.1f} °C  throttled {rows[-1]['throttled']}\n")
print(f"{'famiglia':<10}{'punti':>8}{'punti visti':>13}{'%':>7}{'misure':>10}{'su':>9}{'%':>7}")
for f, p in plan.items():
    print(f"{f:<10}{p:>8}{seen[f]:>13}{100 * seen[f] / p:>6.1f}%{done[f]:>10}{p * REPEATS:>9}{100 * done[f] / (p * REPEATS):>6.1f}%")
print(f"{'totale':<10}{sum(plan.values()):>8}{sum(seen.values()):>13}{100 * sum(seen.values()) / sum(plan.values()):>6.1f}%"
      f"{n:>10}{tot:>9}{100 * n / tot:>6.1f}%")
