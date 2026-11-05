"""Analysis of results/raw.csv (run.py): noise, two-way additive fit on log t, 1D prediction, verdict.

  python analyze.py      -> results/ (point CSV, summary), figures/ (heatmaps, 1D curves)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RES, FIG = OUT / "results", OUT / "figures"
sys.path.insert(0, str(OUT))
from run import GRID, REFS  # noqa: E402  (run.py adds the repo root to sys.path)

REF = 64
W2 = 7.16                       # two-column figure width [in]


def load(path: Path):
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault((r["family"], r["kind"], int(r["c_in"]), int(r["c_out"])), []).append(float(r["ms"]))
    pts = {}
    for k, v in by.items():
        med = float(np.median(v))
        pts[k] = dict(ms=med, min=min(v), max=max(v), spread=(max(v) - min(v)) / med, n=len(v))
    return pts


def additive_fit(T: np.ndarray):
    """log t = a_i + b_j, least squares on a full grid (rows c_in, cols c_out)."""
    L = np.log(T)
    fit = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean()
    res = L - fit
    r2 = 1 - (res ** 2).sum() / ((L - L.mean()) ** 2).sum()
    return res, np.exp(-res) - 1, r2            # rel. error of the fit: (t_hat - t) / t


def mult_class(c: int) -> str:
    return "x32" if c % 32 == 0 else "x16" if c % 16 == 0 else "x8" if c % 8 == 0 else "altro"


def analyze_family(pts, f, offset: float = 0.0):
    t = lambda ci, co: pts[(f, "conv", ci, co)]["ms"] - offset          # noqa: E731
    G = list(GRID)
    T = np.array([[t(ci, co) for co in G] for ci in G])
    res_b, err_b, r2 = additive_fit(T)
    P = np.array([[t(ci, REF) * t(REF, co) / t(REF, REF) for co in G] for ci in G])
    err_c = P / T - 1
    return dict(T=T, res_b=res_b, err_b=err_b, r2=r2, err_c=err_c)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from tools.plots import STYLE, color
    plt.style.use(STYLE)

    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    pts = load(RES / "raw.csv")
    raw = {}
    for r in csv.DictReader(open(RES / "raw.csv")):
        raw.setdefault((r["family"], r["kind"], int(r["c_in"]), int(r["c_out"])), []).append(
            (float(r["ms"]), float(r["temp_c"])))
    with open(RES / "points.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "kind", "c_in", "c_out", "median_ms", "min_ms", "max_ms", "spread_rel", "n"])
        for (f, kind, ci, co), p in sorted(pts.items()):
            w.writerow([f, kind, ci, co, p["ms"], p["min"], p["max"], round(p["spread"], 5), p["n"]])

    G = list(GRID)
    trivial = np.array([[ci == REF or co == REF for co in G] for ci in G])
    summary, fails, A = {}, [], {}
    for f in ("A", "B"):
        conv = [p for (ff, k, *_), p in pts.items() if ff == f and k == "conv"]
        noise = float(np.median([p["spread"] for p in conv]))
        ident = float(np.median([p["ms"] for (ff, k, *_), p in pts.items() if ff == f and k == "identity"]))
        a = A[f] = analyze_family(pts, f)
        a0 = analyze_family(pts, f, offset=ident)          # variant: minus the fixed cost
        ok = np.abs(a["err_c"]) <= 2 * noise
        nt = ~trivial
        summary[f] = {
            "noise_rel": noise, "identity_ms": ident, "n_points": len(conv),
            "b_R2": a["r2"], "b_err_med": float(np.median(np.abs(a["err_b"]))),
            "b_err_max": float(np.abs(a["err_b"]).max()),
            "c_err_med": float(np.median(np.abs(a["err_c"][nt]))),
            "c_err_max": float(np.abs(a["err_c"][nt]).max()),
            "c_frac_within_2noise": float(ok[nt].mean()),
            "separable": bool(ok[nt].mean() >= 0.9),
            "minus_identity": {"b_R2": a0["r2"], "c_err_med": float(np.median(np.abs(a0["err_c"][nt]))),
                               "c_frac_within_2noise": float((np.abs(a0["err_c"]) <= 2 * noise)[nt].mean())},
        }
        # where it fails: failure rate by channel class of c_in and of c_out, and small c_in
        by = {}
        for i, ci in enumerate(G):
            for j, co in enumerate(G):
                if trivial[i, j]:
                    continue
                for key in (f"c_in {mult_class(ci)}", f"c_out {mult_class(co)}",
                            "c_in<32" if ci < 32 else "c_in>=32"):
                    by.setdefault(key, []).append(not ok[i, j])
                if not ok[i, j]:
                    fails.append([f, ci, co, round(a["T"][i, j], 5), round(a["err_c"][i, j], 4)])
        summary[f]["fail_rate_by_class"] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(by.items())}
        # heating: deviation of each repeat from its point median vs temperature (read right after the run)
        d, T = np.array([[r[0] / pts[(f, "conv", ci, co)]["ms"] - 1, r[1]]
                         for (ff, k, ci, co), rs in raw.items() if ff == f and k == "conv" for r in rs]).T
        summary[f]["temp"] = {"min_c": float(T.min()), "max_c": float(T.max()),
                              "slope_rel_per_c": float(np.polyfit(T, d, 1)[0]), "r": float(np.corrcoef(T, d)[0, 1])}

    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(RES / "fails_c.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["family", "c_in", "c_out", "ms", "rel_err_c"]] + fails)

    # heatmaps: rows = family, cols = (b) fit residual, (c) 1D prediction error, in %
    fig, axes = plt.subplots(2, 2, figsize=(W2, 6.4), constrained_layout=True)
    for r, f in enumerate(("A", "B")):
        for c, (key, title) in enumerate((("err_b", "(b) additive fit"), ("err_c", "(c) 1D prediction"))):
            ax = axes[r][c]
            E = 100 * A[f][key]
            if key == "err_c":
                E = np.where(trivial, np.nan, E)
            lim = max(1.0, np.nanmax(np.abs(E)))
            im = ax.imshow(E, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), origin="lower")
            ax.set_xticks(range(len(G)), G, rotation=90, fontsize=7)
            ax.set_yticks(range(len(G)), G, fontsize=7)
            ax.set_xlabel("c_out")
            ax.set_ylabel("c_in")
            n2 = 200 * summary[f]["noise_rel"]
            ax.set_title(f"{'conv 1x1 @40x40' if f == 'A' else 'conv 3x3 @20x20'}: {title}", fontsize=8)
            cb = fig.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label("relative error [%]", fontsize=7)
            if key == "err_c":         # mark points outside 2x noise
                yy, xx = np.where(np.abs(E) > n2)
                ax.scatter(xx, yy, s=4, c="k", marker="x", linewidths=0.6)
    fig.savefig(FIG / "heatmap_residui.png", dpi=200)
    fig.savefig(FIG / "heatmap_residui.pdf")
    plt.close(fig)

    # 1D curves normalised at the reference: if separable they overlap
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.2), constrained_layout=True, sharex=True)
    for r, f in enumerate(("A", "B")):
        for c, scan in enumerate(("c_out", "c_in")):
            ax = axes[r][c]
            for i, ref in enumerate(REFS):
                xs = list(range(1, 129))
                if scan == "c_out":
                    ys = [pts[(f, "conv", ref, x)]["ms"] / pts[(f, "conv", ref, REF)]["ms"] for x in xs]
                    lab = f"c_in = {ref}"
                else:
                    ys = [pts[(f, "conv", x, ref)]["ms"] / pts[(f, "conv", REF, ref)]["ms"] for x in xs]
                    lab = f"c_out = {ref}"
                ax.plot(xs, ys, lw=0.9, color=color("x", i), label=lab)
            for m in range(16, 129, 16):
                ax.axvline(m, color="#9e9e9e", lw=0.4, alpha=0.6)
            ax.set_xticks(range(0, 129, 16))
            ax.set_xlabel(scan)
            ax.set_ylabel(f"t / t({scan} = {REF})")
            ax.set_title(f"{'conv 1x1 @40x40' if f == 'A' else 'conv 3x3 @20x20'}: scan of {scan}", fontsize=8)
            ax.legend(fontsize=7)
    fig.savefig(FIG / "curve_1d_normalizzate.png", dpi=200)
    fig.savefig(FIG / "curve_1d_normalizzate.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
