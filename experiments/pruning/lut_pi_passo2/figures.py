"""Figures of the experiment, in the common style (tools/plots.py, thesis.mplstyle): PNG + PDF in figures/.

  python figures.py lut        # LUT heatmaps, one per family + all families (results/lut_<family>.csv)
  python figures.py warmup     # warm-up transient (runs/latencies_warmup.txt) and error before/after the fix
  python figures.py bimodal    # the two regimes of k1s1_160 48x64 against its neighbours (results/bimodal.json)
  python figures.py all

Computation stays in run.py / analyze.py / warmup.py / bimodal.py: here only reading and drawing.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import run  # noqa: E402  (adds the repo root to sys.path)

RES, FIG = run.RES, OUT / "figures"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.plots import STYLE
    plt.style.use(STYLE)
    return plt


def _panel(ax, letter: str, text: str):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8)


def _lut(f: str):
    P = {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"]) for r in csv.DictReader(open(RES / f"lut_{f}.csv"))}
    cis, cos = sorted({c for c, _ in P}), sorted({c for _, c in P})
    M = np.full((len(cis), len(cos)), np.nan)
    for (ci, co), v in P.items():
        M[cis.index(ci), cos.index(co)] = v
    return cis, cos, M


def _family_label(f: str) -> str:
    k, s, hw, _ = run.FAMILIES[f]
    return f"{k}×{k} s{s} @{hw}"


def _draw(ax, f: str, norm):
    """Heatmap c_in x c_out; the stem (c_in fixed at 3) has one row: a curve over c_out instead."""
    cis, cos, M = _lut(f)
    if len(cis) == 1:
        ax.plot(cos, M[0], marker="o", ms=3, lw=1)
        ax.set_yscale("log")
        ax.set_ylabel("latenza in ms", fontsize=7)
        return None
    ax.grid(False)
    return ax.pcolormesh(cos, cis, M, norm=norm, shading="nearest", cmap="viridis", rasterized=True)


def lut():
    plt = _style()
    from matplotlib.colors import LogNorm
    from tools.plots import HALF, WIDTH, save
    fams = [f for f in run.FAMILIES if (RES / f"lut_{f}.csv").exists()]
    if not fams:
        sys.exit("nessuna results/lut_<famiglia>.csv: prima python analyze.py")
    for f in fams:                                      # one per family, half width
        fig, ax = plt.subplots(figsize=(HALF, 0.85 * HALF))
        im = _draw(ax, f, LogNorm())
        ax.set_xlabel("c_out")
        if im is None:
            ax.set_ylabel("latenza mediana in ms")
        else:
            ax.set_ylabel("c_in")
            fig.colorbar(im, ax=ax).set_label("latenza mediana in ms")
        save(fig, FIG / f"heatmap_{f}")
    n, cols = len(fams), 5                              # all families, full width, shared colour scale
    rows = -(-n // cols)
    norm = LogNorm(min(np.nanmin(_lut(f)[2]) for f in fams), max(np.nanmax(_lut(f)[2]) for f in fams))
    fig, axs = plt.subplots(rows, cols, figsize=(WIDTH, 0.25 * WIDTH * rows), squeeze=False)
    im = None
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis("off")
            continue
        im = _draw(ax, fams[i], norm) or im
        ax.tick_params(labelsize=7)
        ax.set_title(f"({'abcdefghijklmnop'[i]}) {_family_label(fams[i])}", loc="left", fontsize=7)
        if i % cols == 0:
            ax.set_ylabel("c_in", fontsize=7)
        if i >= n - cols:
            ax.set_xlabel("c_out", fontsize=7)
    fig.colorbar(im, ax=axs, shrink=0.7).set_label("latenza mediana in ms")
    save(fig, FIG / "heatmap_all")


def warmup():
    plt = _style()
    import warmup as W
    from tools.plots import GRAY, WIDTH, color, save
    # (a, b) relative latency vs iteration and vs time, measures right after the previous one
    curves = [(k, t, v / steady, steady) for k, m, v, t, steady, _ in W.series()
              if k[0] in W.REPS_NOW and int(m["remeasured"]) == 0]
    speeds = np.log10([c[3] for c in curves])
    norm = plt.Normalize(speeds.min(), speeds.max())
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 0.42 * WIDTH), sharey=True)
    for (k, t, rel, _), s in zip(curves, speeds):
        step = max(1, len(rel) // 2000)                 # thin long series
        sm = np.convolve(rel, np.ones(5) / 5, mode="valid")[::step]
        axs[0].plot(np.arange(len(sm)) * step + 1, sm, lw=0.5, alpha=0.6, color=cmap(norm(s)), rasterized=True)
        axs[1].plot(t[2:len(rel) - 2][::step], sm, lw=0.5, alpha=0.6, color=cmap(norm(s)), rasterized=True)
    for ax, lab, letter, title in zip(axs, ("iterazione", "tempo dalla prima inferenza in ms"), "ab",
                                      ("per iterazione", "per tempo")):
        ax.set_xscale("log")
        ax.axhspan(0.98, 1.02, color=GRAY, alpha=0.25, lw=0)
        ax.set_xlabel(lab)
        _panel(ax, letter, title)
    axs[0].set_ylim(0.8, 3)
    axs[0].set_ylabel("latenza / regime")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axs).set_label("log₁₀ del regime in ms")
    save(fig, FIG / "warmup")
    # (a) error of the LUT protocol before/after on the 19 nets, (b) pilot points old vs new
    w = json.loads((RES / "warmup.json").read_text())["measures_detail"]
    before = {}
    for r in w:
        before.setdefault((r["family"], r["c_in"], r["c_out"]), []).append((r["steady_ms"], r["err_niter200"]))
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 0.4 * WIDTH))
    ax = axs[0]
    for (f, ci, co), v in before.items():
        st = np.median([a for a, _ in v])
        ident = f.startswith("id")
        ax.scatter([st] * len(v), [100 * e for _, e in v], s=10, marker="s" if ident else "o",
                   color=color("x", 1), alpha=0.7)
    chk = RES / "warmup_check.json"
    if chk.exists():
        for k, d in json.loads(chk.read_text())["per_net"].items():
            ax.scatter([d["steady_ms"]] * len(d["err"]), [100 * e for e in d["err"]], s=10,
                       marker="s" if k.startswith("id") else "o", color=color("x", 0), alpha=0.7)
    ax.scatter([], [], s=10, color=color("x", 1), label="un'inferenza di warmup")
    ax.scatter([], [], s=10, color=color("x", 0), label="warmup di 60 ms")
    ax.scatter([], [], s=10, marker="s", color=GRAY, label="rete identità")
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("latenza a regime in ms")
    ax.set_ylabel("errore della mediana in %")
    ax.legend(loc="upper right")
    _panel(ax, "a", "protocollo LUT, 19 reti")
    ax = axs[1]
    pil = RES / "warmup_pilot.json"
    if pil.exists():
        pts = json.loads(pil.read_text())["points_detail"]
        ax.scatter([p["new_ms"] for p in pts], [100 * p["old_over_new"] for p in pts], s=10, color=color("x", 1),
                   alpha=0.7)
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=5, linscale=2)
        ax.set_yticks([-5, 0, 5, 20, 100], ["−5", "0", "5", "20", "100"])
        ax.set_xlabel("latenza con il warmup nuovo in ms")
        ax.set_ylabel("vecchio / nuovo − 1, in %")
    _panel(ax, "b", "pilota, 97 conv")
    save(fig, FIG / "warmup_gain")


def bimodal():
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    d = json.loads((RES / "bimodal.json").read_text())["benchmark_app"]
    nets = sorted(d, key=lambda k: np.median([x["median"] for x in d[k]]))
    fig, ax = plt.subplots(figsize=(WIDTH, 0.33 * WIDTH))
    rng = np.random.default_rng(0)
    for i, k in enumerate(nets):
        m = np.array([x["median"] for x in d[k]])
        rel = m / np.median(m) - 1
        target = k == "k1s1_160_48_64"
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(m)), 100 * rel, s=12, alpha=0.8,
                   color=color("x", 1) if target else color("x", 0))
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xticks(range(len(nets)), [k.replace("_", " ", 1).replace("_", "×") for k in nets], fontsize=7)
    ax.set_ylabel("misura / mediana − 1, in %")
    save(fig, FIG / "bimodal")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("lut", lut), ("warmup", warmup), ("bimodal", bimodal)):
        if what in (name, "all"):
            if name == "lut" and what == "all" and not list(RES.glob("lut_*.csv")):
                print("lut: nessuna LUT ancora, salto")
                continue
            fn()
            print("ok", name)
