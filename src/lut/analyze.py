"""Analysis of the LUT measures (src.lut.pi lut): one LUT per family and a summary.
Figures: experiments/pruning/lut_pi_passo2/figures.py lut.

  python -m src.lut.analyze <dir>   # <dir>/results/raw.csv -> <dir>/results/lut_<family>.csv, summary.json
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .pi import REPEATS

ANOMALY = 5            # a point is anomalous if its spread is > ANOMALY x the family noise (and > 10%)


def main(res: Path):
    raw = res / "raw.csv"
    rows = list(csv.DictReader(open(raw)))
    by = {}
    for r in rows:
        by.setdefault((r["family"], int(r["c_in"]), int(r["c_out"])), []).append(float(r["ms"]))
    pts = {}
    for k, v in by.items():
        med = float(np.median(v))
        pts[k] = dict(ms=med, min=min(v), max=max(v), spread=(max(v) - min(v)) / med, n=len(v))

    fams = sorted({k[0] for k in pts})
    ts = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    summary = {"measures": len(rows), "points": len(pts),
               "start": min(ts).isoformat(), "end": max(ts).isoformat(),
               "hours": (max(ts) - min(ts)).total_seconds() / 3600,
               "remeasured": sum(int(r["remeasured"]) for r in rows),
               "throttled_nonzero": sum(r["throttled"] != "0x0" for r in rows),
               "temp_max_c": max(float(r["temp_c"]) for r in rows),
               "unstable_remeasured": sum(p["n"] > REPEATS for p in pts.values()),
               "freq_khz": sorted({int(r["freq_khz"]) for r in rows}),
               "families": {}}
    for f in fams:
        P = {(ci, co): p for (ff, ci, co), p in pts.items() if ff == f}
        noise = float(np.median([p["spread"] for p in P.values()]))
        anom = sorted(((ci, co, round(p["spread"], 3)) for (ci, co), p in P.items()
                       if p["spread"] > max(ANOMALY * noise, 0.10)), key=lambda x: -x[2])
        summary["families"][f] = {"points": len(P), "noise_rel": noise,
                                  "ms_min": min(p["ms"] for p in P.values()),
                                  "ms_max": max(p["ms"] for p in P.values()),
                                  "n_anomalous": len(anom), "anomalous": anom[:50]}
        with open(res / f"lut_{f}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["c_in", "c_out", "median_ms", "min_ms", "max_ms", "spread_rel", "n"])
            for (ci, co), p in sorted(P.items()):
                w.writerow([ci, co, p["ms"], p["min"], p["max"], round(p["spread"], 5), p["n"]])
    (res / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "families"}, indent=1))
    for f, d in summary["families"].items():
        print(f, d["points"], f"rumore {100 * d['noise_rel']:.1f}%", f"{d['ms_min']:.3f}-{d['ms_max']:.3f} ms",
              f"anomali {d['n_anomalous']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) / "results")
