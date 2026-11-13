# Quanto rende ogni ms di latenza sul Pi 5 a 640: (a) guadagno di mAP per 10 ms a ogni passo del fronte mAP50-95;
# (b) spazio di potatura di ogni architettura (rete potata al massimo -> rete intera) rispetto al budget.
#   .venv/bin/python tools/figure_guadagno_latenza.py [OUT_DIR] [--budget MS] [--latency CSV]
# Larghezza del testo (tools.plots.WIDTH), stile della tesi, PNG + PDF. Pavimento: C_arch + sum a_f LUT_f sugli strati della rete
# nano_floor (pavimento_s) riportata sulle architetture; dove c'e' il pavimento misurato vale quello.
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.figura_pareto import fronte  # noqa: E402
from tools.figure_griglia import LETTERA  # noqa: E402
from tools.plots import GRAY, STYLE, WIDTH, color, save  # noqa: E402

EXP = ROOT / "experiments/pruning"
LUT = ROOT / "artifacts/lut/rpi5"
SEGNO = {"D101": "-", "D126": "", "D170": "+"}
PROTETTI = ("8.cv2", "9.cv2", "16.cv2", "19.cv2", "22.cv1b", "22.m.0.0.cv2", "22.cv2")  # alimentano C2PSA/PSABlock/Detect
SOGLIA = 0.85  # pavimento al massimo all'85 % del budget (NEW_MASTERPLAN)
SIZE = {"n": 0.25, "p": 0.125, "s": 0.5}  # yolo26n/p/s_D126: i primi modelli, prima della griglia yolo26w<larghezza>


def arch_di(w, d):
    return f"w{w:g}_{d}"


def sigla(arch):
    w, d = re.fullmatch(r"w([\d.]+)_(D\d+)", arch).groups()
    return LETTERA[float(w)] + SEGNO[d]


def righe(p):
    return list(csv.DictReader(open(p)))


def mediane(p):
    acc = defaultdict(list)
    for r in righe(p):
        acc[r["net"]].append(float(r["ms"]))
    return {k: float(np.median(v)) for k, v in acc.items()}


def map_modelli(weights):
    """{arch: [(mAP50, mAP50-95), ...]} dai meta.json delle ricette e300 (un elemento per seme)."""
    out = defaultdict(list)
    for d in sorted(Path(weights).glob("yolo26*_e300_*")):
        m = re.match(r"yolo26(?:w([\d.]+)|([nps]))_(D\d+)_", d.name)
        met = json.load(open(d / "meta.json")).get("metrics") if (d / "meta.json").exists() else None
        if m and met:
            out[arch_di(float(m[1]) if m[1] else SIZE[m[2]], m[3])].append((met["map50"], met["map50_95"]))
    return out


def pavimenti(layers_csv, nano_csv, fit, lut_dir, archs):
    """{arch: ms} del pavimento stimato con C_arch + sum a_f LUT_f. Lo schema dei canali e' quello di nano_floor (74 conv):
    un conv protetto resta pieno (c_out dell'architettura), gli altri a 2 canali; i nomi delle conv extra di D170
    (m.1, m.2) si leggono come m.0, e un ingresso a piu' pezzi da 2 canali (concat) scala col numero di pezzi."""
    from yolopit.latency import FamilyLUT

    fit = json.load(open(fit))
    a = {f: v for k, v in fit["a"].items() for f in k.split("+")}
    nano = [r for r in righe(nano_csv) if r["net"] in ("nano_floor", "nano_full")]
    fl = {r["layer"]: (int(r["c_in"]), int(r["c_out"])) for r in nano if r["net"] == "nano_floor"}
    fu = {r["layer"]: int(r["c_in"]) for r in nano if r["net"] == "nano_full"}
    lut, tot = {}, {}
    full = defaultdict(list)
    for r in righe(layers_csv):
        if r["net"].endswith("_full") and r["net"][:-5] in archs:
            full[r["net"][:-5]].append(r)
    for arch, rs in full.items():
        w = float(re.match(r"w([\d.]+)_", arch)[1])
        s = w / 0.25
        t = fit["C_ms"][arch]
        for r in rs:
            ref = re.sub(r"(layers\.\d+\.m\.)\d+", r"\g<1>0", r["layer"])
            ci_ref, _ = fl[ref]
            ci, co = int(r["c_in"]), int(r["c_out"])
            prot = r["layer"].endswith(tuple(f"layers.{p}.conv" for p in PROTETTI))
            if ci_ref >= 64:   # ingresso con canali protetti: scalano con la larghezza, i pezzi da 2 restano (258 = 256 + 2)
                ci_f = round((ci_ref - ci_ref % 64) * s) + ci_ref % 64
            elif ci_ref % 2:   # immagine in ingresso
                ci_f = ci_ref
            else:              # numero di pezzi da 2 canali: scala come l'ingresso pieno
                ci_f = round(ci_ref * ci / (fu[ref] * s))
            f = r["family"]
            if f not in lut:
                lut[f] = FamilyLUT(f, Path(lut_dir) / f"lut_{f}.csv")
            t += a[f] * float(lut[f](ci_f, co if prot else 2))
        tot[arch] = t
    return tot


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

    b4r, pav = Path(args.b4r), Path(args.pavimento)
    lat = {r["arch"]: float(r["measured_ms"]) for r in righe(b4r / "predictions.csv") if r["net"].endswith("_full")}
    nano = mediane(pav / "raw.csv")  # reti di pavimento_s: s_full, s_floor, nano_floor, nano_full
    lat["w0.5_D126"] = nano["s_full"]
    mis = {"w0.25_D126": nano["nano_floor"], "w0.5_D126": nano["s_floor"]}  # pavimenti misurati
    if args.latency:
        for r in righe(args.latency):
            lat[re.sub(r"_full$", "", r.get("arch") or r["model"])] = float(r.get("ms") or r["measured_ms"])
    stim = pavimenti(b4r / "layers.csv", pav / "layers.csv", LUT / "fit.json", LUT, lat)
    mp = map_modelli(args.weights)
    arch = sorted((a for a in lat if a in mp), key=lat.get)
    mAP = {a: tuple(np.mean(mp[a], 0)) for a in arch}
    floor = {a: (mis[a], True) if a in mis else (stim[a], False) for a in arch if a in mis or a in stim}
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
        G0, G1, W = int(10 * np.ceil(lo / 10)), int(10 * np.floor(hi / 10)), 8  # il vuoto occupa W ms
        k = W / (G1 - G0)
        fw = lambda x: np.where(x <= G0, x, np.where(x <= G1, G0 + (x - G0) * k, x - (G1 - G0) + W))
        iv = lambda y: np.where(y <= G0, y, np.where(y <= G0 + W, G0 + (y - G0) / k, y + (G1 - G0) - W))
        axb.set_xscale("function", functions=(fw, iv))
        axb.axvspan(G0, G1, color=plt.rcParams["axes.facecolor"], lw=0, zorder=3.5)
        axb.plot([G0, G1], [0, 0], ls="", marker=[(-1, -2), (1, 2)], ms=7, mew=0.9, color=ink,
                 transform=axb.get_xaxis_transform(), clip_on=False, zorder=5)
        ticks = [t for t in ticks if t < G0 or t >= G1]
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
