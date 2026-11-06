"""B4-R figures, in the common style (tools/plots.py): PNG + PDF in figures/.

  python figures.py nets      # what was generated: MACs of the nets, kept fraction per family (results/*.csv)
  python figures.py fit       # predicted vs measured, a_f per family, error vs latency (results/fit.json, predictions.csv)
  python figures.py all
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "lut_pi_passo2"))
import run as lut  # noqa: E402  (adds the repo root to sys.path)

RES, FIG = OUT / "results", OUT / "figures"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.plots import STYLE
    plt.style.use(STYLE)
    return plt


def _panel(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8)


def nets():
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    rows = list(csv.DictReader(open(RES / "nets.csv")))
    full = next(float(r["cost_ops"]) for r in rows if r["net"] == "full")
    frac = [float(r["cost_ops"]) / full for r in rows if r["net"] != "full"]
    layers = list(csv.DictReader(open(RES / "layers.csv")))
    ref = {(r["order"]): (int(r["c_in"]), int(r["c_out"])) for r in layers if r["net"] == "full"}
    fams = [f for f in lut.FAMILIES if any(r["family"] == f for r in layers)]
    kept = {f: [] for f in fams}
    for r in layers:
        if r["net"] != "full":
            kept[r["family"]].append(int(r["c_out"]) / ref[r["order"]][1])
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 0.38 * WIDTH), gridspec_kw={"width_ratios": [1, 2]})
    axs[0].hist(frac, bins=20, color=color("x", 0))
    axs[0].set_xlabel("MAC / MAC del nano")
    axs[0].set_ylabel("reti")
    _panel(axs[0], "a", f"{len(frac)} reti")
    axs[1].boxplot([kept[f] for f in fams], showfliers=False, widths=0.6,
                   medianprops={"color": color("x", 1)})
    axs[1].set_xticks(range(1, len(fams) + 1), [f.replace("_", " @") for f in fams], rotation=60, fontsize=7)
    axs[1].set_ylabel("c_out tenuti / c_out del nano")
    axs[1].axhline(1, color=GRAY, lw=0.6)
    _panel(axs[1], "b", "per famiglia")
    save(fig, FIG / "nets")


def fit():
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    out = json.loads((RES / "fit.json").read_text())
    rows = list(csv.DictReader(open(RES / "predictions.csv")))
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 0.36 * WIDTH), gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]})
    ax = axs[0]
    for split, mk in (("train", "o"), ("test", "s")):
        r = [x for x in rows if x["split"] == split]
        ax.scatter([float(x["measured_ms"]) for x in r], [float(x["composition_ms"]) for x in r], s=10,
                   marker=mk, color=color("x", 0 if split == "train" else 1), alpha=0.75, label=split)
    lo = min(float(x["measured_ms"]) for x in rows) * 0.95
    hi = max(float(x["measured_ms"]) for x in rows) * 1.05
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.6)
    ax.set_xlabel("misurata [ms]")
    ax.set_ylabel("stimata [ms]")
    ax.legend(loc="upper left")
    _panel(ax, "a", "composizione")
    ax = axs[1]
    names = list(out["a"])
    ax.bar(range(len(names)), list(out["a"].values()), color=color("x", 0))
    ax.axhline(1, color=GRAY, lw=0.6)
    ax.set_xticks(range(len(names)), [n.replace("_", " @") for n in names], rotation=90, fontsize=7)
    ax.set_ylabel("a_f")
    _panel(ax, "b", "correzione per famiglia")
    ax = axs[2]
    for m, i in (("composition_ms", 0), ("lut_sum_ms", 2), ("macs_ms", 1)):
        r = [x for x in rows if x["split"] == "test"]
        ax.scatter([float(x["measured_ms"]) for x in r],
                   [100 * (float(x[m]) / float(x["measured_ms"]) - 1) for x in r], s=10, color=color("x", i),
                   alpha=0.75, label={"composition_ms": "composizione", "lut_sum_ms": "somma LUT + C",
                                      "macs_ms": "MAC"}[m])
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel("misurata [ms]")
    ax.set_ylabel("errore sul test [%]")
    ax.legend(loc="best")
    _panel(ax, "c", "confronto")
    save(fig, FIG / "fit")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("nets", nets), ("fit", fit)):
        if what in (name, "all"):
            if name == "fit" and not (RES / "fit.json").exists():
                print("fit: nessun results/fit.json, salto")
                continue
            fn()
            print("ok", name)
