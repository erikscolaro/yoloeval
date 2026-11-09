"""B4-R figures, in the common style (tools/plots.py): PNG + PDF in figures/.

  python figures.py nets      # what was generated: MACs of the nets, kept fraction per family (results/*.csv)
  python figures.py mac       # measured latency against MACs of every net, with C + beta*MAC (results/raw.csv)
  python figures.py fit       # predicted vs measured, a_f per family, error vs latency (results/fit.json, predictions.csv)
  python figures.py all
  python figures.py fit --dummy   # same figure from made-up data, watermarked, in reports/stimatore_latenza/
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
    full = {r["arch"]: float(r["cost_ops"]) for r in rows if r["net"].endswith("_full")}   # one per architecture
    frac = [float(r["cost_ops"]) / full[r["arch"]] for r in rows if not r["net"].endswith("_full")]
    layers = list(csv.DictReader(open(RES / "layers.csv")))
    ref = {(r["net"].rsplit("_", 1)[0], r["order"]): int(r["c_out"]) for r in layers if r["net"].endswith("_full")}
    fams = [f for f in lut.FAMILIES if any(r["family"] == f for r in layers)]
    kept = {f: [] for f in fams}
    for r in layers:
        if not r["net"].endswith("_full"):
            kept[r["family"]].append(int(r["c_out"]) / ref[(r["net"].rsplit("_", 1)[0], r["order"])])
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 0.38 * WIDTH), gridspec_kw={"width_ratios": [1, 2]})
    axs[0].hist(frac, bins=20, color=color("x", 0))
    axs[0].set_xlabel("MAC / MAC della rete intera")
    axs[0].set_ylabel("reti")
    _panel(axs[0], "a", f"{len(frac)} reti, {len(full)} architetture")
    axs[1].boxplot([kept[f] for f in fams], showfliers=False, widths=0.6,
                   medianprops={"color": color("x", 1)})
    axs[1].set_xticks(range(1, len(fams) + 1), [f.replace("_", " @") for f in fams], rotation=60, fontsize=7)
    axs[1].set_ylabel("frazione di c_out tenuta")
    axs[1].axhline(1, color=GRAY, lw=0.6)
    _panel(axs[1], "b", "per famiglia")
    save(fig, FIG / "nets")


def mac():
    """Measured latency of every net against its MACs (median of the repeats measured so far), with C + beta*MAC.
    Colour = width, marker = depth: nets with the same MACs and different latency are what MACs miss."""
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    nets = {r["net"]: r for r in csv.DictReader(open(RES / "nets.csv"))}
    ms = {}
    for r in csv.DictReader(open(RES / "raw.csv")):
        ms.setdefault(r["net"], []).append(float(r["ms"]))
    names = sorted(ms)
    y = np.array([np.median(ms[n]) for n in names])
    x = np.array([float(nets[n]["cost_ops"]) for n in names]) / 1e9
    c, b = np.polyfit(x, y, 1)[::-1]
    widths = sorted({nets[n]["width"] for n in names}, key=float)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.5 * WIDTH))
    for n_, xi, yi in zip(names, x, y):
        w, d = nets[n_]["width"], nets[n_]["depth"]
        ax.scatter(xi, yi, s=10, alpha=0.75, lw=0, color=color("x", widths.index(w)),
                   marker={"D101": "o", "D126": "s", "D170": "^"}[d])
    xs = np.linspace(0, x.max() * 1.02, 2)
    ax.plot(xs, c + b * xs, color=GRAY, lw=0.8)
    for i, w in enumerate(widths):
        ax.scatter([], [], s=10, color=color("x", i), label=f"larghezza {w}")
    for d, mk in (("D101", "o"), ("D126", "s"), ("D170", "^")):
        ax.scatter([], [], s=10, color=GRAY, marker=mk, label=d)
    ax.plot([], [], color=GRAY, lw=0.8, label="C + β·MAC")
    ax.set(xlabel="GMAC", ylabel="latenza misurata in ms")
    ax.legend(fontsize=6, ncol=2)
    err = 100 * ((c + b * x) / y - 1)
    ax.set_title(f"{len(names)} reti su 765; errore di C + β·MAC: mediana {np.median(abs(err)):.1f}%, "
                   f"p95 {np.percentile(abs(err), 95):.1f}%", loc="left", fontsize=8)
    save(fig, FIG / "mac")


def _dummy() -> tuple[dict, list[dict]]:
    """Made-up fit.json and predictions.csv (same keys), for the report draft before B4-R is measured."""
    rng = np.random.default_rng(0)
    fams = list(lut.FAMILIES)
    a = {f: float(1 - 0.08 * np.log2(lut.FAMILIES[f][2] / 20) + rng.normal(0, 0.03)) for f in fams}
    rows = []
    for i, y in enumerate(rng.uniform(8, 60, 300)):
        rows.append({"split": "test" if i % 5 == 0 else "train", "measured_ms": y,
                     "composition_ms": y * (1 + rng.normal(-0.004, 0.02)),
                     "lut_sum_ms": y * (1 + rng.normal(0.038, 0.04)), "macs_ms": y * (1 + rng.normal(-0.06, 0.12))})
    return {"a": a}, rows


def fit(dummy: bool = False):
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    if dummy:
        out, rows = _dummy()
    else:
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
    ax.set_xlabel("misurata in ms")
    ax.set_ylabel("stimata in ms")
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
    ax.set_xlabel("misurata in ms")
    ax.set_ylabel("errore sul test in %")
    ax.legend(loc="upper right", fontsize=6)
    _panel(ax, "c", "confronto")
    if dummy:
        fig.text(0.5, 0.5, "DATI FITTIZI", ha="center", va="center", fontsize=28, color=GRAY, alpha=0.25, rotation=20)
    save(fig, lut.REPO / "reports" / "stimatore_latenza" / "fit_fittizio" if dummy else FIG / "fit")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if "--dummy" in sys.argv:
        fit(dummy=True)
        print("ok fit --dummy")
        sys.exit()
    for name, fn in (("nets", nets), ("mac", mac), ("fit", fit)):
        if what in (name, "all"):
            if name == "fit" and not (RES / "fit.json").exists():
                print("fit: nessun results/fit.json, salto")
                continue
            fn()
            print("ok", name)
