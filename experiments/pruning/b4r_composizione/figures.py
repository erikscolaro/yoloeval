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
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=9)


DEPTH_MARKER = {"D101": "o", "D126": "s", "D170": "^"}


def _fam(f: str) -> str:
    """Three-line family label for tick marks: 13 of them fit the text width at 8 pt."""
    return lut.family_label(f, "\n").replace(" ", "\n", 1)


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
    fig, axs = plt.subplots(2, 1, figsize=(WIDTH, 0.72 * WIDTH), gridspec_kw={"height_ratios": [1, 1.25]})
    axs[0].hist(frac, bins=np.linspace(0, 1, 21), color=color("x", 0), edgecolor="white", lw=0.5)
    axs[0].set_xlabel("MAC della rete potata / MAC della rete intera")
    axs[0].set_ylabel("reti")
    _panel(axs[0], "a", f"{len(frac)} reti potate da {len(full)} architetture")
    axs[1].boxplot([kept[f] for f in fams], showfliers=False, widths=0.55, patch_artist=True,
                   boxprops={"facecolor": "white", "edgecolor": GRAY}, whiskerprops={"color": GRAY},
                   capprops={"color": GRAY}, medianprops={"color": color("x", 1), "lw": 1.5})
    axs[1].set_xticks(range(1, len(fams) + 1), [_fam(f) for f in fams])
    axs[1].set_ylabel("canali tenuti / canali\ndella rete intera")
    axs[1].set_ylim(0, 1.05)
    axs[1].grid(axis="x", visible=False)
    _panel(axs[1], "b", "canali di uscita tenuti, per famiglia")
    save(fig, FIG / "nets")


def mac():
    """Measured latency of every net against its MACs (median of the repeats), with C + beta*MAC.
    Colour = width, marker = depth: nets with the same MACs and different latency are what MACs miss."""
    plt = _style()
    from tools.figure_griglia import LETTERA
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
    fig, ax = plt.subplots(figsize=(WIDTH, 0.62 * WIDTH))
    for n_, xi, yi in zip(names, x, y):
        w, d = nets[n_]["width"], nets[n_]["depth"]
        ax.scatter(xi, yi, s=14, alpha=0.8, lw=0, color=color("x", widths.index(w)), marker=DEPTH_MARKER[d])
    xs = np.linspace(0, x.max() * 1.02, 2)
    ax.plot(xs, c + b * xs, color=GRAY, lw=1, ls="--", zorder=0)
    h = [ax.scatter([], [], s=14, color=color("x", i), label=f"{LETTERA[float(w)]} (larghezza {float(w):g})")
         for i, w in enumerate(widths)]
    h.append(ax.plot([], [], color=GRAY, lw=1, ls="--", label="retta C + β·MAC")[0])
    h += [ax.scatter([], [], s=14, color=GRAY, marker=mk, label=f"profondità {d}") for d, mk in DEPTH_MARKER.items()]
    ax.set(xlabel="MAC in miliardi", ylabel="latenza misurata in ms", xlim=(0, None), ylim=(0, None))
    fig.legend(handles=h, loc="outside lower center", ncol=3, columnspacing=1.2, handletextpad=0.3)  # by column
    save(fig, FIG / "mac")


ESTIMATORS = (("macs_ms", 1, "retta sui MAC"), ("lut_sum_ms", 2, "somma della LUT"),
              ("composition_ms", 0, "stimatore"))


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
    """(a) ECDF of |error| on the held-out nets for the three estimators, with the target (5% at p99); (b) signed
    error against measured latency; (c) a_f per family."""
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    if dummy:
        out, rows = _dummy()
    else:
        out = json.loads((RES / "fit.json").read_text())
        rows = list(csv.DictReader(open(RES / "predictions.csv")))
    test = [x for x in rows if x["split"] == "test"]
    y = np.array([float(x["measured_ms"]) for x in test])
    err = {m: 100 * (np.array([float(x[m]) for x in test]) / y - 1) for m, _, _ in ESTIMATORS}
    fig = plt.figure(figsize=(WIDTH, 0.82 * WIDTH))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.85])
    axs = top.subplots(1, 2)
    ax = axs[0]
    for m, i, label in ESTIMATORS:
        e = np.sort(abs(err[m]))
        ax.step(e, 100 * np.arange(1, len(e) + 1) / len(e), where="post", color=color("x", i), lw=1.5,
                label=label)
    ax.axvline(5, color=GRAY, lw=1, ls="--")
    ax.axhline(99, color=GRAY, lw=1, ls="--")  # target: 99% of the nets within 5%
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 5, 10, 30], ["0,1", "0,3", "1", "3", "5", "10", "30"])
    ax.set(xlim=(0.05, 40), ylim=(0, 101), xlabel="|errore| in %", ylabel="reti con errore minore, in %")
    ax.legend(loc="upper left")
    _panel(ax, "a", "distribuzione dell'errore")
    ax = axs[1]
    for m, i, label in ESTIMATORS:
        ax.scatter(y, err[m], s=7, lw=0, alpha=0.7, color=color("x", i))
    ax.axhline(0, color=GRAY, lw=1)
    ax.set(xlabel="latenza misurata in ms", ylabel="errore in %")
    _panel(ax, "b", "errore per rete")
    ax = bottom.subplots()
    names = [f for f in lut.FAMILIES if f in out["a"]]
    ax.bar(range(len(names)), [out["a"][f] for f in names], width=0.6, color=color("x", 0))
    ax.axhline(1, color=GRAY, lw=1, ls="--")
    ax.set_xticks(range(len(names)), [_fam(n) for n in names])
    ax.set_ylabel("$a_f$")
    ax.grid(axis="x", visible=False)
    _panel(ax, "c", "correzione per famiglia")
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
