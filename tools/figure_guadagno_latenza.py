# Quanto rende ogni ms di latenza sul Pi 5 a 640: (a) guadagno di mAP per 10 ms a ogni passo del fronte mAP50-95;
# (b) spazio di potatura di ogni architettura (rete potata al massimo -> rete intera) rispetto al budget.
#   .venv/bin/python tools/figure_guadagno_latenza.py [OUT_DIR] [--budget MS] [--latency CSV]
# Larghezza del testo (tools.plots.WIDTH), stile della tesi, PNG + PDF. Pavimento: C_arch + sum a_f LUT_f sugli strati della rete
# nano_floor (pavimento_s) riportata sulle architetture; dove c'e' il pavimento misurato vale quello.
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.figura_pareto import fronte  # noqa: E402
from tools.plots import GRAY, STYLE, WIDTH, color, save  # noqa: E402
from src.selection import EXP, dati, sigla  # noqa: E402

LUT = ROOT / "artifacts/lut/rpi5"
SOGLIA = 0.85  # pavimento al massimo all'85 % del budget (NEW_MASTERPLAN)


def guadagno(fr):
    """Punti di mAP per 10 ms fra punti consecutivi del fronte: [(t0, t1, g)]."""
    return [(t0, t1, 100 * (y1 - y0) / (t1 - t0) * 10) for (t0, y0), (t1, y1) in zip(fr, fr[1:])]


def ordine(a):
    """Larghezza crescente, poi profondita' (x-, x, x+)."""
    w, d = re.fullmatch(r"w([\d.]+)_D(\d+)", a).groups()
    return float(w), int(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(ROOT / "reports/guadagno_latenza_640"))
    ap.add_argument("--budget", type=float, help="ms; senza, niente linee del budget")
    ap.add_argument("--latency", help="csv (arch, ms) con le latenze a 640: sostituisce o estende quelle disponibili")
    ap.add_argument("--weights", default=str(ROOT / "artifacts/weights"))
    ap.add_argument("--b4r", default=str(EXP / "b4r_composizione/results"))
    ap.add_argument("--pavimento", default=str(EXP / "pavimento_s/results"))
    args = ap.parse_args()

    lat, mAP, floor = dati(LUT, args.latency, args.weights, args.b4r, args.pavimento)
    arch = sorted(mAP, key=lat.get)
    B = args.budget

    plt.style.use(STYLE)
    ink = plt.rcParams["axes.edgecolor"]
    c50, c5095 = color("map50", 1), color("map50_95", 0)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(WIDTH, 3.9), width_ratios=[1, 1.45])

    # (a) rendimento di ogni passo lungo il fronte mAP50-95: righe in ordine di latenza, dall'alto
    fr = fronte([(lat[a], mAP[a][1], a) for a in arch])
    passi = list(zip(fr, fr[1:]))
    y = np.arange(len(passi))[::-1]
    h = 0.38
    for k, ((t0, _, a0), (t1, _, a1)) in enumerate(passi):
        g95 = 1000 * (mAP[a1][1] - mAP[a0][1]) / (t1 - t0)
        g50 = 1000 * (mAP[a1][0] - mAP[a0][0]) / (t1 - t0)
        axa.barh(y[k] + h / 2, g95, h, color=c5095, label="mAP50-95" if k == 0 else None)
        axa.barh(y[k] - h / 2, g50, h, color=c50, label="mAP50" if k == 0 else None)
        for g, dy, c in ((g95, h / 2, c5095), (g50, -h / 2, c50)):
            axa.annotate(f"{g:.1f}".replace(".", ","), (max(g, 0), y[k] + dy), xytext=(3, 0),
                         textcoords="offset points", va="center", color=c, fontsize="small")
        print(f"{sigla(a0)}->{sigla(a1)} {t0:.1f}->{t1:.1f} ms  mAP50-95 {g95:+.2f}  mAP50 {g50:+.2f} punti / 10 ms")
    axa.set_yticks(y, [f"{sigla(a0)} \u2192 {sigla(a1)}" for (_, _, a0), (_, _, a1) in passi])
    axa.set_ylim(-0.6, len(passi) - 0.4)
    axa.axvline(0, color=ink, lw=0.8)
    axa.set_xlim(right=axa.get_xlim()[1] * 1.15)
    axa.set_xlabel("punti di mAP ogni 10 ms")
    axa.grid(axis="y", visible=False)
    axa.tick_params(axis="y", length=0)
    axa.legend(loc="lower right")

    # (b) spazio di potatura: dalla rete potata al massimo alla rete intera, modelli x-, x, x+ dall'alto
    cat = {"dentro": (color("fp16"), "già nel budget"), "potabile": (color("calib"), "nel budget potando"),
           "fuori": (GRAY, "non adatto alla potatura")}
    rows = sorted((a for a in arch if a in floor), key=ordine)
    yb = {a: i for i, a in enumerate(rows[::-1])}
    for a in rows:
        f, misurato = floor[a]
        k = "fuori" if B is None else "dentro" if lat[a] <= B else "potabile" if f <= SOGLIA * B else "fuori"
        col = cat[k][0] if B else color("rpi5")
        r = yb[a]
        axb.plot([f, lat[a]], [r, r], color=col, lw=2, solid_capstyle="butt", zorder=3)
        axb.plot(lat[a], r, "o", color=col, mec="white", mew=0.6, zorder=4)
        axb.plot(f, r, "D" if misurato else "o", mfc=col if misurato else "white", mec=col, mew=1.2,
                 ms=4.5 if misurato else 4, zorder=4)
        print(f"{sigla(a):3s} {a:13s} pavimento {f:5.1f} ms ({'misurato' if misurato else 'stimato'}) "
              f"intera {lat[a]:5.1f} ms mAP50-95 {mAP[a][1]:.3f} {k if B else ''}")
    for i in range(1, len(rows)):  # separa le larghezze
        if ordine(rows[i])[0] != ordine(rows[i - 1])[0]:
            axb.axhline(yb[rows[i]] + 0.5, color=plt.rcParams["grid.color"], lw=0.8, zorder=0)
    axb.set_yticks([yb[a] for a in rows], [sigla(a) for a in rows])
    axb.set_ylim(-0.6, len(rows) - 0.4)
    ax2 = axb.twinx()  # colonna della mAP50-95 a destra, lontana dalle linee verticali
    ax2.set_ylim(axb.get_ylim())
    ax2.set_yticks([yb[a] for a in rows], [f"{mAP[a][1]:.3f}".replace(".", ",") for a in rows])
    ax2.set_ylabel("mAP50-95")
    ax2.tick_params(axis="y", length=0)
    ax2.grid(False)
    ax2.spines["right"].set_visible(False)
    axb.grid(axis="y", visible=False)
    axb.tick_params(axis="y", length=0)
    # asse lineare; se fra due valori c'e' un vuoto > 40 ms lo si taglia (asse spezzato)
    xs = sorted(v for a in rows for v in (floor[a][0], lat[a]))
    x0 = int(10 * (xs[0] // 10))
    ticks = list(range(x0, int(xs[-1]) + 1, 10))
    lo, hi = max(zip(xs, xs[1:]), key=lambda p: p[1] - p[0])
    if hi - lo > 40:
        G0, G1, W = int(10 * np.ceil(lo / 10)), int(10 * np.floor(hi / 10)), 10  # il vuoto occupa quanto un passo delle tacche
        k = W / (G1 - G0)
        fw = lambda x: np.where(x <= G0, x, np.where(x <= G1, G0 + (x - G0) * k, x - (G1 - G0) + W))
        iv = lambda y: np.where(y <= G0, y, np.where(y <= G0 + W, G0 + (y - G0) / k, y + (G1 - G0) - W))
        axb.set_xscale("function", functions=(fw, iv))
        axb.plot([G0 + 0.38 * (G1 - G0), G0 + 0.62 * (G1 - G0)], [0, 0], ls="", marker=[(-1, -2), (1, 2)], ms=7, mew=0.9, color=ink,
                 transform=axb.get_xaxis_transform(), clip_on=False, zorder=5)
        ticks = [t for t in ticks if t <= G0 or t >= G1]
    axb.set_xticks(ticks, [str(t) for t in ticks])
    axb.minorticks_off()
    axb.set_xlim(x0, xs[-1] + 3)
    axb.set_xlabel("latenza sul Raspberry Pi 5 a 640 (ms)")
    if B:
        for x, ls in ((B, "--"), (SOGLIA * B, ":")):
            axb.axvline(x, color=ink, lw=0.9, ls=ls, zorder=1)
    for ax, l in ((axa, "a"), (axb, "b")):
        ax.set_title(f"({l})", loc="left")

    # legenda in tre colonne: simboli, colori, linee
    sym = [plt.Line2D([], [], color=ink, marker="o", ls="", label="rete intera"),
           plt.Line2D([], [], color=ink, marker="o", mfc="white", ls="", ms=4, label="potata al massimo (stima)"),
           plt.Line2D([], [], color=ink, marker="D", ls="", ms=4.5, label="potata al massimo (misura)")]
    hd = sym
    if B:
        col = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in cat.values()]
        lin = [plt.Line2D([], [], color=ink, ls="--", lw=0.9, label=f"budget {B:g} ms"),
               plt.Line2D([], [], color=ink, ls=":", lw=0.9, label=f"limite {SOGLIA:.0%} del budget".replace("%", " %"))]
        hd = sym + col + lin
    fig.legend(handles=hd, loc="outside lower center", ncol=3, handlelength=1.2, columnspacing=0.8, handletextpad=0.5)
    print(save(fig, Path(args.out) / "guadagno_latenza_640"))


if __name__ == "__main__":
    main()
