"""Figures of the experiment, in the common style (tools/plots.py, thesis.mplstyle): PNG + PDF in figures/.

  python figures.py lut        # LUT heatmaps, one per family + all families (results/lut_<family>.csv)
  python figures.py sezioni    # 1D cuts of the LUT (jumps on multiples of 8) and the jump by c_out mod 8
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
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=9)


CIN, COUT = "$c_\\mathrm{in}$", "$c_\\mathrm{out}$"


def _lut(f: str):
    P = {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"]) for r in csv.DictReader(open(RES / f"lut_{f}.csv"))}
    cis, cos = sorted({c for c, _ in P}), sorted({c for _, c in P})
    M = np.full((len(cis), len(cos)), np.nan)
    for (ci, co), v in P.items():
        M[cis.index(ci), cos.index(co)] = v
    return cis, cos, M


def _family_label(f: str) -> str:
    return run.family_label(f)


def _edges(v):
    """Cell edges around grid values (works for a single value too, the stem's c_in = 3)."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return np.array([v[0] - 1, v[0] + 1])
    mid = (v[1:] + v[:-1]) / 2
    return np.r_[2 * v[0] - mid[0], mid, 2 * v[-1] - mid[-1]]


def _draw(ax, f: str, norm):
    """Heatmap c_in x c_out; the stem (c_in fixed at 3) is a one-row strip on the same colour scale."""
    from matplotlib.ticker import MaxNLocator
    cis, cos, M = _lut(f)
    ax.grid(False)
    im = ax.pcolormesh(_edges(cos), _edges(cis), M, norm=norm, cmap="viridis", rasterized=True)
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True) if len(cis) > 1 else plt_fixed(cis))
    return im


def plt_fixed(v):
    from matplotlib.ticker import FixedLocator
    return FixedLocator(v)


def _plain(axis):
    """Log axis with plain numbers (0,01 / 0,1 / 1 / 10) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:g}".replace(".", ",")))
    axis.set_minor_formatter(NullFormatter())


def _ms_colorbar(fig, im, **kw):
    from matplotlib.ticker import FixedLocator, FuncFormatter
    cb = fig.colorbar(im, **kw)
    lo, hi = im.norm.vmin, im.norm.vmax
    cb.ax.yaxis.set_major_locator(FixedLocator([t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10) if lo <= t <= hi]))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:g}".replace(".", ",")))
    cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    cb.set_label("latenza mediana in ms")
    return cb


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
        ax.set(xlabel=COUT, ylabel=CIN)
        _ms_colorbar(fig, im, ax=ax)
        save(fig, FIG / f"heatmap_{f}")
    # all families, full width, shared colour scale: rows = kernel/stride, columns = side of the output map
    kinds = sorted({run.FAMILIES[f][:2] for f in fams})
    sides = sorted({run.FAMILIES[f][2] for f in fams})
    norm = LogNorm(min(np.nanmin(_lut(f)[2]) for f in fams), max(np.nanmax(_lut(f)[2]) for f in fams))
    fig, axs = plt.subplots(len(kinds), len(sides), figsize=(WIDTH, 0.6 * WIDTH), squeeze=False)
    im = None
    for f in fams:
        k, st, hw, _ = run.FAMILIES[f]
        r, c = kinds.index((k, st)), sides.index(hw)
        ax = axs[r, c]
        im = _draw(ax, f, norm)
        ax.set_title(f"{hw}×{hw}", fontsize=8)
    for ax in axs.flat:
        if not ax.has_data():
            ax.axis("off")
    for (k, st), ax in zip(kinds, axs[:, 0]):
        ax.set_ylabel(f"{k}×{k} s{st}\n{CIN}")
    for c in range(len(sides)):                         # x label on the lowest panel of each column
        col = [ax for ax in axs[:, c] if ax.has_data()]
        col[-1].set_xlabel(COUT)
    axs[0, -1].set_ylabel(CIN)
    _ms_colorbar(fig, im, ax=axs, shrink=0.8, aspect=30)
    save(fig, FIG / "heatmap_all")


def warmup():
    plt = _style()
    import warmup as W
    from tools.plots import GRAY, WIDTH, color, save
    # (a, b) relative latency vs iteration and vs time, measures right after the previous one
    curves = [(k, t, v / steady, steady) for k, m, v, t, steady, _ in W.series()
              if k[0] in W.REPS_NOW and int(m["remeasured"]) == 0]
    from matplotlib.colors import LogNorm
    speeds = np.array([c[3] for c in curves])
    norm = LogNorm(speeds.min(), speeds.max())
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
        _plain(ax.xaxis)
        ax.axhspan(0.98, 1.02, color=GRAY, alpha=0.25, lw=0)
        ax.set_xlabel(lab)
        _panel(ax, letter, title)
    axs[0].set_xticks([1, 100, 10000], ["1", "100", "10 000"])
    axs[0].set_ylim(0.8, 3)
    axs[0].set_ylabel("latenza / regime")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axs)
    cb.set_label("latenza a regime in ms")
    _plain(cb.ax.yaxis)
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
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xscale("log")
    _plain(ax.xaxis)
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
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xscale("log")
        _plain(ax.xaxis)
        ax.set_yscale("symlog", linthresh=5, linscale=2)
        ax.set_yticks([-5, 0, 5, 20, 100], ["−5", "0", "5", "20", "100"])
        ax.set_xlabel("latenza con il warmup nuovo in ms")
        ax.set_ylabel("scarto del protocollo vecchio in %")
    _panel(ax, "b", "pilota, 97 conv")
    save(fig, FIG / "warmup_gain")


def sezioni():
    """1D cuts of the LUT (c_in fixed): sawtooth on multiples of 8, lines = the linear interpolation PIT uses;
    then the mean jump against the two neighbours by c_out mod 8, every family."""
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    cuts = [("k3s2_40", 64), ("k1s1_40", 128), ("k3s1_20", 64)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 0.42 * WIDTH), gridspec_kw={"width_ratios": [1.7, 1]})
    for i, (f, ci) in enumerate(cuts):
        cis, cos, M = _lut(f)
        y = M[cis.index(ci)]
        y = y / np.nanmax(y)
        c = color("x", i)
        ax1.plot(cos, y, lw=1, color=c, label=f"{_family_label(f)}, {CIN} = {ci}")
        m8 = np.array(cos) % 8 == 0
        ax1.scatter(np.array(cos)[m8], y[m8], s=12, color=c, zorder=3)
        ax1.scatter(np.array(cos)[~m8], y[~m8], s=6, facecolor="white", edgecolor=c, lw=0.6, zorder=3)
    ax1.set(xlabel=COUT, ylabel="latenza / massimo della sezione", ylim=(0, 1.05))
    ax1.legend(loc="lower right", handlelength=1.2)
    _panel(ax1, "a", f"sezioni a {CIN} fisso")
    dev = {r: [] for r in (0, 2, 4, 6)}
    for f in run.FAMILIES:
        P = {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"]) for r in csv.DictReader(open(RES / f"lut_{f}.csv"))}
        for (ci, co), v in P.items():
            lo, hi = P.get((ci, co - 2)), P.get((ci, co + 2))
            if lo and hi:
                dev[co % 8].append(100 * (v / ((lo + hi) / 2) - 1))
    ax2.boxplot([dev[r] for r in dev], whis=(5, 95), showfliers=False, widths=0.55, medianprops={"color": color("x", 1), "lw": 1.5},
                boxprops={"color": GRAY}, whiskerprops={"color": GRAY}, capprops={"color": GRAY})
    ax2.scatter(range(1, 5), [np.mean(dev[r]) for r in dev], marker="D", s=12, color=color("x", 0), zorder=3,
                label="media")
    ax2.axhline(0, color=GRAY, lw=1)
    ax2.set_xticks(range(1, 5), [f"{r}" for r in dev])
    ax2.set(xlabel=f"resto di {COUT} diviso 8", ylabel="scarto dai vicini in %")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="lower right")
    _panel(ax2, "b", "tutte le famiglie")
    save(fig, FIG / "sezioni")
    for r, d in dev.items():
        print(f"c_out mod 8 = {r}: media {np.mean(d):+.2f}%, n = {len(d)}")


def _net_label(k: str) -> str:
    """'k1s1_160_48_64' -> two lines: family, then c_in and c_out."""
    f, ci, co = k.rsplit("_", 2)
    return f"{_family_label(f)}\n{ci} → {co}"


def bimodal():
    plt = _style()
    from tools.plots import GRAY, WIDTH, color, save
    d = json.loads((RES / "bimodal.json").read_text())["benchmark_app"]
    nets = sorted(d, key=lambda k: np.median([x["median"] for x in d[k]]))
    fig, ax = plt.subplots(figsize=(WIDTH, 0.4 * WIDTH))
    rng = np.random.default_rng(0)
    for i, k in enumerate(nets):
        m = np.array([x["median"] for x in d[k]])
        rel = m / np.median(m) - 1
        target = k == "k1s1_160_48_64"
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(m)), 100 * rel, s=12, alpha=0.8,
                   color=color("x", 1) if target else color("x", 0))
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xticks(range(len(nets)), [_net_label(k) for k in nets])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("scarto dalla mediana in %")
    save(fig, FIG / "bimodal")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("lut", lut), ("sezioni", sezioni), ("warmup", warmup), ("bimodal", bimodal)):
        if what in (name, "all"):
            if name in ("lut", "sezioni") and what == "all" and not list(RES.glob("lut_*.csv")):
                print("lut: nessuna LUT ancora, salto")
                continue
            fn()
            print("ok", name)
