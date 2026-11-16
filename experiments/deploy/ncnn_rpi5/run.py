"""Tabella dell'esperimento dai risultati in results/: latenza media sulle ripetizioni (media di ogni
ripetizione: benchncnn non da' la mediana) e mAP sul val misurata sul Pi, con scarti da OpenVINO fp16."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
NOMI = {"yolo26p_D126": "P", "yolo26n_D126": "N"}

celle = defaultdict(list)
for p in sorted((HERE / "results").glob("*.json")):
    d = json.loads(p.read_text())
    if d.get("status") != "ok" or not d.get("latency"):
        continue
    c = d["config"]
    celle[(NOMI.get(c["model"]["name"], c["model"]["name"]), c["backend"]["name"], c["quantization"]["name"])].append(d)

righe = []
for (m, b, q), ds in sorted(celle.items()):
    lat = [d["latency"]["mean_ms"] for d in ds]
    acc = next((d["accuracy"] for d in ds if d.get("accuracy")), {}) or {}
    righe.append({"modello": m, "backend": b, "precisione": q, "rip": len(lat),
                  "media_ms": round(statistics.mean(lat), 2),
                  "dev_ms": round(statistics.stdev(lat), 2) if len(lat) > 1 else None,
                  "map50_95": round(acc["map50_95"], 4) if "map50_95" in acc else None,
                  "map50": round(acc["map50"], 4) if "map50" in acc else None})
rif = {r["modello"]: r for r in righe if r["backend"] == "openvino" and r["precisione"] == "fp16"}
for r in righe:
    base = rif.get(r["modello"])
    r["d_lat_%"] = round(100 * (r["media_ms"] / base["media_ms"] - 1), 1) if base else None
    r["d_map50_95"] = (round(r["map50_95"] - base["map50_95"], 4)
                       if base and r["map50_95"] is not None and base["map50_95"] is not None else None)

assert righe, "nessun risultato in results/"
with open(HERE / "results" / "tabella.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(righe[0]))
    w.writeheader()
    w.writerows(righe)
for r in righe:
    print("\t".join(str(v) for v in r.values()))
