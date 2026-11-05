"""Analysis of results/raw.csv (run.py): one LUT per family, summary, heatmaps.

  python analyze.py [raw.csv]   -> results/lut_<family>.csv, results/lut_identity.csv, results/summary.json,
                                   figures/heatmap_<family>.png
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RES, FIG = OUT / "results", OUT / "figures"
sys.path.insert(0, str(OUT))
from run import FAMILIES  # noqa: E402  (run.py adds the repo root to sys.path)

ANOMALY = 5            # a point is anomalous if its spread is > ANOMALY x the family noise (and > 10%)
W1 = 3.5               # one-column figure width [in]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from tools.plots import STYLE
    plt.style.use(STYLE)

    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "raw.csv"
    rows = list(csv.DictReader(open(raw)))
    by = {}
    for r in rows:
        by.setdefault((r["family"], int(r["c_in"]), int(r["c_out"])), []).append(float(r["ms"]))
    pts = {}
    for k, v in by.items():
        med = float(np.median(v))
        pts[k] = dict(ms=med, min=min(v), max=max(v), spread=(max(v) - min(v)) / med, n=len(v))

    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    fams = sorted({k[0] for k in pts}, key=lambda f: (f.startswith("id"), f))
    ts = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    summary = {"measures": len(rows), "points": len(pts),
               "start": min(ts).isoformat(), "end": max(ts).isoformat(),
               "hours": (max(ts) - min(ts)).total_seconds() / 3600,
               "remeasured": sum(int(r["remeasured"]) for r in rows),
               "throttled_nonzero": sum(r["throttled"] != "0x0" for r in rows),
               "temp_max_c": max(float(r["temp_c"]) for r in rows),
               "freq_khz": sorted({int(r["freq_khz"]) for r in rows}),
               "families": {}}
    ident = [(k, p) for k, p in pts.items() if k[0].startswith("id")]
    with open(RES / "lut_identity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["map_in", "c", "median_ms", "min_ms", "max_ms", "spread_rel", "n"])
        for (f, ci, _), p in sorted(ident, key=lambda x: (int(x[0][0][2:]), x[0][1])):
            w.writerow([int(f[2:]), ci, p["ms"], p["min"], p["max"], round(p["spread"], 5), p["n"]])
    for f in fams:
        P = {(ci, co): p for (ff, ci, co), p in pts.items() if ff == f}
        noise = float(np.median([p["spread"] for p in P.values()]))
        anom = sorted(((ci, co, round(p["spread"], 3)) for (ci, co), p in P.items()
                       if p["spread"] > max(ANOMALY * noise, 0.10)), key=lambda x: -x[2])
        summary["families"][f] = {"points": len(P), "noise_rel": noise,
                                  "ms_min": min(p["ms"] for p in P.values()),
                                  "ms_max": max(p["ms"] for p in P.values()),
                                  "n_anomalous": len(anom), "anomalous": anom[:50]}
        if f.startswith("id"):
            continue
        with open(RES / f"lut_{f}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["c_in", "c_out", "median_ms", "min_ms", "max_ms", "spread_rel", "n"])
            for (ci, co), p in sorted(P.items()):
                w.writerow([ci, co, p["ms"], p["min"], p["max"], round(p["spread"], 5), p["n"]])
        cis, cos = sorted({c for c, _ in P}), sorted({c for _, c in P})
        M = np.full((len(cis), len(cos)), np.nan)
        for (ci, co), p in P.items():
            M[cis.index(ci), cos.index(co)] = p["ms"]
        fig, ax = plt.subplots(figsize=(W1, 2.9), constrained_layout=True)
        im = ax.pcolormesh(cos, cis, M, norm=LogNorm(), shading="nearest", cmap="viridis")
        k, s, hw, _ = FAMILIES[f]
        ax.set_title(f"conv {k}x{k} s{s}, output {hw}x{hw}", fontsize=8)
        ax.set_xlabel("c_out")
        ax.set_ylabel("c_in")
        fig.colorbar(im, ax=ax).set_label("median latency [ms]", fontsize=7)
        fig.savefig(FIG / f"heatmap_{f}.png", dpi=200)
        plt.close(fig)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "families"}, indent=1))
    for f, d in summary["families"].items():
        print(f, d["points"], f"rumore {100 * d['noise_rel']:.1f}%", f"{d['ms_min']:.3f}-{d['ms_max']:.3f} ms",
              f"anomali {d['n_anomalous']}")


if __name__ == "__main__":
    main()
