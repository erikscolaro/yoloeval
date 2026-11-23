"""Miglior modello dello SCAN per ogni cella camera x ingresso. Budget = 33,3 ms - costo fuori dal modello (p99 della misura
in results/fuori_modello.jsonl) - post (0,009 ms) - Kalman con un track (0,21 ms). Un modello sta nel budget se il massimo
delle sue passate (stima prudente del p99) e' sotto. Fra i modelli nel budget: mAP massima; sotto 0,01 di differenza si prende
il piu' veloce. Latenze di ../modelli_latenza_per_ingresso (solo reti baseline, fp16, Pi 5); mAP50-95 del val 640x640."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.selection import arch_di, map_modelli  # noqa: E402
from tools.figura_pareto import nome  # noqa: E402

FRAME, POST, KALMAN, PARI = 1000 / 30, 0.009, 0.21, 0.01
DEPLOY = {"640x384": "640x384", "HD": "720p", "FHD": "1080p"}
LARG = {"640": 640, "HD": 1280, "FHD": 1920}
LARG_IN = {"640x384": 640, "HD": 1280, "FHD": 1920}

extra = defaultdict(list)
for riga in open(HERE / "results/fuori_modello.jsonl"):
    r = json.loads(riga)
    extra[(r["camera"], r["ingresso"], r["interp"])].append(r["p99_ms"])

lat = defaultdict(lambda: {"med": [], "max": []})
for p in (ROOT / "results").rglob("*.json"):
    rec = json.loads(p.read_text())
    ax = rec.get("axes") or {}
    if rec.get("status") != "ok" or ax.get("board") != "rpi5" or ax.get("strategy", "baseline") != "baseline":
        continue
    if ax.get("quantization") != "fp16" or ax.get("deploy") not in DEPLOY.values():
        continue
    m, l = rec["config"]["model"], rec["latency"]
    k = (float(m["width"]), m["depth"], ax["deploy"])
    lat[k]["med"].append(l["median_ms"])
    lat[k]["max"].append(l.get("max_ms") or l["median_ms"])

mp = {k: float(np.mean([m[1] for m in v])) for k, v in map_modelli(ROOT / "artifacts/weights").items()}
righe = []
for cam in LARG:
    for inp in DEPLOY:
        if LARG_IN[inp] > LARG[cam]:
            continue
        for interp in ("area", "linear"):
            e = float(np.median(extra[(cam, inp, interp)]))
            budget = FRAME - e - POST - KALMAN
            ok = [(float(np.median(v["med"])), max(v["max"]), mp[arch_di(w, d)], w, d)
                  for (w, d, dep), v in lat.items()
                  if dep == DEPLOY[inp] and arch_di(w, d) in mp and max(v["max"]) <= budget]
            if not ok:
                righe.append([cam, inp, interp, round(e, 2), round(budget, 2), "-", None, None, None])
                continue
            migliore = max(o[2] for o in ok)
            t, mx, mapp, w, d = min((o for o in ok if o[2] >= migliore - PARI), key=lambda o: o[0])
            righe.append([cam, inp, interp, round(e, 2), round(budget, 2), nome(w, d), round(t, 1), round(mx, 1), round(mapp, 3)])

head = ["camera", "ingresso", "interp", "fuori_modello_ms", "budget_ms", "modello", "lat_mediana_ms", "lat_max_ms", "mAP50-95"]
with open(HERE / "results/celle.csv", "w") as f:
    f.write(",".join(head) + "\n")
    for r in righe:
        f.write(",".join("" if v is None else str(v) for v in r) + "\n")
print("| camera | ingresso | interp. | fuori modello | budget | modello | latenza (mediana / max) | mAP50-95 |\n|---|---|---|---|---|---|---|---|")
for r in righe:
    print(f"| {r[0]} | {r[1]} | {r[2].upper()} | {r[3]} ms | {r[4]} ms | {r[5]} | "
          + (f"{r[6]} / {r[7]} ms | {r[8]} |" if r[6] else "— | — |"))
