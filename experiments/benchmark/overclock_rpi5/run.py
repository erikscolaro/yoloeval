"""Tabella dell'overclock: latenza media OpenVINO fp16 di P e N per configurazione (results/<prova>_arm<f>_gpu<g>/),
con i clock letti sul Pi (results/clock.csv) e lo scarto dalla configurazione di serie (A, 2400, GPU di serie)."""

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
clock = {(r["prova"], r["arm_freq"], r["gpu_freq"] or "def"): r for r in csv.DictReader(open(HERE / "results/clock.csv"))}
righe = []
for d in sorted(p for p in (HERE / "results").iterdir() if p.is_dir()):
    prova, arm, gpu = d.name.split("_")
    arm, gpu = arm[3:], gpu[3:]
    lat = {"yolo26p_D126": [], "yolo26n_D126": []}
    for f in d.glob("*.json"):
        j = json.loads(f.read_text())
        if j.get("status") == "ok" and j.get("latency"):
            lat.setdefault(j["config"]["model"]["name"], []).append(j["latency"]["mean_ms"])
    c = clock.get((prova, arm, gpu), {})
    righe.append({"prova": prova, "arm_mhz": int(arm), "gpu_mhz": gpu, "core_mhz": round(int(c["core_hz"]) / 1e6)
                  if c.get("core_hz") else "", "volt_core": c.get("volt_core", ""), "throttled": c.get("throttled", ""),
                  **{f"{m}_ms": round(statistics.mean(v), 2) if v else "" for m, v in
                     (("P", lat["yolo26p_D126"]), ("N", lat["yolo26n_D126"]))}})
base = next((r for r in righe if r["prova"] == "A" and r["arm_mhz"] == 2400), None)
for r in righe:
    for m in ("P", "N"):
        ok = base and r[f"{m}_ms"] != "" and base[f"{m}_ms"] != ""
        r[f"d{m}_%"] = round(100 * (r[f"{m}_ms"] / base[f"{m}_ms"] - 1), 1) if ok else ""
assert righe, "nessuna configurazione misurata"
with open(HERE / "results/tabella.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(righe[0]))
    w.writeheader()
    w.writerows(righe)
for r in righe:
    print("\t".join(str(v) for v in r.values()))
