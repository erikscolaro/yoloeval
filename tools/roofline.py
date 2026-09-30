#!/usr/bin/env python3
"""Roofline dei modelli misurati: efficienza e confronto baseline / modelli potati.

    python -m tools.roofline [--results results] [--out reports/roofline]

Legge le celle `ok` del benchmark (results/*.json, con i blocchi `complexity` e
`model_info` scritti dall'export) e i tetti misurati da `stage=roofline`
(results/roofline/*.json). Per ogni gruppo board x compute target x precisione x backend:

- intensita' aritmetica del modello I = MAC / byte (src/measure/complexity.py);
- prestazione ottenuta P = MAC / latenza mediana;
- tetto raggiungibile R = min(picco, banda x I) e efficienza P / R;
- memory-bound o compute-bound, a seconda di dove cade I rispetto al ridge point;
- rispetto alla baseline dello stesso modello: speedup, rapporto di MAC, delta di mAP.

Scrive roofline.csv (una riga per cella), un PNG per gruppo e stampa le tabelle.
Se per un gruppo non c'e' un roofline misurato con lo stesso backend, usa quello di un altro
backend sulla stessa board, compute target e precisione, e lo dichiara nella colonna
`roof_backend`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_roofs(results: Path) -> list[dict]:
    return [r for p in sorted((results / "roofline").glob("*.json")) if (r := _read(p))]


def find_roof(roofs: list[dict], board, compute, quant, backend) -> dict | None:
    same = [r for r in roofs if (r["board"], r["compute_target"], r["quantization"]) ==
            (board, compute, quant)]
    exact = [r for r in same if r["backend"] == backend]
    pool = exact or same
    return max(pool, key=lambda r: r.get("created_at") or "") if pool else None


def label(model_info: dict | None, strategy: str | None) -> str:
    info = model_info or {}
    strategy = info.get("strategy") or strategy or "baseline"
    if strategy == "baseline":
        return "baseline"
    n = info.get("pit_n")
    return f"{strategy} N={n}" if n else strategy


def rows_from_results(results: Path, roofs: list[dict]) -> list[dict]:
    rows = []
    for path in sorted(results.glob("*.json")):
        rec = _read(path) or {}
        if rec.get("status") != "ok" or not rec.get("complexity"):
            continue
        ax, lat, cx = rec.get("axes") or {}, rec.get("latency") or {}, rec["complexity"]
        ms = lat.get("median_ms") or lat.get("mean_ms")
        if not ms:
            continue
        roof = find_roof(roofs, ax.get("board"), ax.get("compute_target"),
                         ax.get("quantization"), ax.get("backend"))
        macs, nbytes = cx["macs"], cx["bytes"]
        intensity = macs / nbytes if nbytes else None
        achieved = macs / (ms * 1e-3) / 1e9
        row = {
            "cell_id": rec.get("cell_id"), "board": ax.get("board"),
            "compute_target": ax.get("compute_target"), "quantization": ax.get("quantization"),
            "backend": ax.get("backend"), "model": ax.get("model"),
            "strategy": ax.get("strategy") or "baseline",
            "label": label(rec.get("model_info"), ax.get("strategy")),
            "pit_n": (rec.get("model_info") or {}).get("pit_n"),
            "latency_ms": ms, "gmac": macs / 1e9, "mbytes": nbytes / 1e6,
            "intensity_mac_per_byte": intensity, "achieved_gmacs": achieved,
            "map50_95": (rec.get("accuracy") or {}).get("map50_95"),
        }
        if roof:
            attainable = min(roof["peak_gmacs"], roof["bandwidth_gbs"] * intensity)
            row.update(
                peak_gmacs=roof["peak_gmacs"], bandwidth_gbs=roof["bandwidth_gbs"],
                ridge_mac_per_byte=roof["ridge_mac_per_byte"], roof_backend=roof["backend"],
                attainable_gmacs=attainable, efficiency=achieved / attainable,
                bound="memory" if intensity < roof["ridge_mac_per_byte"] else "compute")
        rows.append(row)
    return rows


def add_baseline_comparison(rows: list[dict]) -> None:
    """speedup, rapporto di MAC e delta di mAP rispetto alla baseline dello stesso gruppo."""
    key = lambda r: (r["board"], r["compute_target"], r["quantization"], r["backend"],
                     r["model"])
    base = {key(r): r for r in rows if r["strategy"] == "baseline"}
    for r in rows:
        b = base.get(key(r))
        if not b:
            continue
        r["speedup_vs_baseline"] = b["latency_ms"] / r["latency_ms"]
        r["mac_ratio_vs_baseline"] = r["gmac"] / b["gmac"]
        if r.get("map50_95") is not None and b.get("map50_95") is not None:
            r["map_delta_vs_baseline"] = r["map50_95"] - b["map50_95"]


def group_key(r: dict) -> tuple:
    return (r["board"], r["compute_target"], r["quantization"], r["backend"])


def plot_group(rows: list[dict], roof: dict | None, datasheet: dict | None, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [r["intensity_mac_per_byte"] for r in rows]
    lo, hi = min(xs + [1]) / 4, max(xs + [100]) * 4
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    if roof:
        ax.plot(grid, np.minimum(roof["peak_gmacs"], roof["bandwidth_gbs"] * grid), "k-",
                linewidth=2, label=f"measured roof ({roof['backend']})")
        ax.axvline(roof["ridge_mac_per_byte"], color="k", alpha=.2, linestyle=":")
    if datasheet and datasheet.get("gmacs") and datasheet.get("bandwidth_gbs"):
        ax.plot(grid, np.minimum(datasheet["gmacs"], datasheet["bandwidth_gbs"] * grid), "k--",
                alpha=.5, label="datasheet roof")
    markers = "osD^v<>p*h"
    for i, r in enumerate(sorted(rows, key=lambda r: (r["strategy"] != "baseline",
                                                      r["label"]))):
        ax.scatter(r["intensity_mac_per_byte"], r["achieved_gmacs"], s=60, zorder=3,
                   marker=markers[i % len(markers)],
                   label=f"{r['model']} {r['label']}"
                         + (f" ({100 * r['efficiency']:.0f}%)" if "efficiency" in r else ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity [MAC/byte]")
    ax.set_ylabel("performance [GMAC/s]")
    b, c, q, be = group_key(rows[0])
    ax.set_title(f"{b} · {c} · {q} · {be}")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


COLUMNS = ["model", "label", "latency_ms", "gmac", "intensity_mac_per_byte", "achieved_gmacs",
           "attainable_gmacs", "efficiency", "bound", "speedup_vs_baseline",
           "mac_ratio_vs_baseline", "map_delta_vs_baseline"]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return "" if v is None else str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=Path("results"), type=Path)
    ap.add_argument("--out", default=Path("reports/roofline"), type=Path)
    args = ap.parse_args(argv)

    roofs = load_roofs(args.results)
    rows = rows_from_results(args.results, roofs)
    if not rows:
        print("nessuna cella ok con la complessita' del modello: esegui export e benchmark "
              "(gli export fatti prima di questa versione non la hanno: +force_reexport=true)",
              file=sys.stderr)
        return 1
    if not roofs:
        print("! nessun roofline misurato (stage=roofline): niente tetti ne' efficienza",
              file=sys.stderr)
    add_baseline_comparison(rows)
    args.out.mkdir(parents=True, exist_ok=True)

    import csv

    fields = sorted({k for r in rows for k in r})
    with open(args.out / "roofline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    for key, grp in sorted(groups.items()):
        roof = find_roof(roofs, *key)
        datasheet = (roof or {}).get("datasheet") or {}
        ds = None
        if datasheet:
            gm = (datasheet.get("gmacs") or {}).get(key[2])
            ds = {"gmacs": gm, "bandwidth_gbs": datasheet.get("bandwidth_gbs")}
        png = args.out / ("_".join(str(k) for k in key) + ".png")
        plot_group(grp, roof, ds, png)
        print(f"\n== {' / '.join(str(k) for k in key)}  -> {png}")
        if roof:
            print(f"   roof: picco {roof['peak_gmacs']:.1f} GMAC/s, banda "
                  f"{roof['bandwidth_gbs']:.1f} GB/s, ridge {roof['ridge_mac_per_byte']:.1f} "
                  f"MAC/byte ({roof['backend']})")
        print("   " + " | ".join(COLUMNS))
        for r in sorted(grp, key=lambda r: (r["model"], r["strategy"] != "baseline",
                                            r["label"])):
            print("   " + " | ".join(_fmt(r.get(c)) for c in COLUMNS))
    print(f"\n{len(rows)} celle -> {args.out / 'roofline.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
