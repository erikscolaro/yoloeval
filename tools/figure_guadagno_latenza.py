# Quanto rende ogni ms di latenza: fronte di Pareto mAP-latenza sul Pi 5 a 640 con il guadagno marginale fra punti
# consecutivi, e spazio di potatura (pavimento -> latenza intera, all'altezza della mAP50-95 dell'architettura).
#   .venv/bin/python tools/figure_guadagno_latenza.py [OUT_DIR] [--budget MS] [--latency CSV]
# Una colonna (3,5 in), stile della tesi, PNG + PDF. Pavimento: C_arch + sum a_f LUT_f sugli strati della rete
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
from tools.figura_pareto import DIRS, etichette, fronte  # noqa: E402
from tools.figure_griglia import LETTERA  # noqa: E402
from tools.plots import STYLE, color, save  # noqa: E402

COLUMN = 3.5
DIRS += [(dx * r / 5, dy * r / 5, ha, va) for r in (12,) for dx, dy, ha, va in DIRS[:8]]  # un anello in più per i punti vicini
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(ROOT / "reports/guadagno_latenza_640"))
    ap.add_argument("--budget", type=float, help="ms; senza, niente linee del budget")
    ap.add_argument("--esempio", action="store_true", help="marca il budget come esempio nella legenda")
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
    for a, v in mp.items():
        if len(v) > 1:
            print(f"{sigla(a)}: {len(v)} semi, mAP50-95 {np.mean([x[1] for x in v]):.4f} (range {min(x[1] for x in v):.4f}-{max(x[1] for x in v):.4f})")
    arch = sorted((a for a in lat if a in mp), key=lat.get)
    mAP = {a: tuple(np.mean(mp[a], 0)) for a in arch}
    floor = {a: (mis[a], "misurato") if a in mis else (stim[a], "stimato") for a in arch if a in mis or a in stim}

    plt.style.use(STYLE)
    ink = plt.rcParams["axes.edgecolor"]
    fig, axs = plt.subplots(4, 1, figsize=(COLUMN, 8.2), sharex=True, height_ratios=[1, 1, 0.8, 3.2])
    c50, c5095 = "#eb6834", "#2a78d6"  # due colori fissi per le due metriche, uguali in tutti i pannelli
    gains = []
    for ax, m, c, nome in zip(axs[:2], (0, 1), (c50, c5095), ("mAP50", "mAP50-95")):
        pts = [(lat[a], mAP[a][m], a) for a in arch]
        fr = fronte(pts)
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color=c, lw=1.4,
                drawstyle="steps-post", zorder=2)
        ax.plot([p[0] for p in pts if p not in fr], [p[1] for p in pts if p not in fr], "o", mfc="white", mec=c, ms=4,
                alpha=0.7, zorder=3)
        ax.plot([p[0] for p in fr], [p[1] for p in fr], "o", color=c, mec="white", mew=0.6, ms=5, zorder=4)
        ax.set_ylabel(nome)
        if m:
            etichette(ax, [(t, y, sigla(a), c) for t, y, a in fr], [(t, y, 3.5) for t, y, _ in pts], [])
        gains.append((nome, c, guadagno([(t, y) for t, y, _ in fr])))
    ax = axs[2]
    for nome, c, g in gains:
        for t0, t1, v in g:
            ax.plot([t0, t1], [v, v], color=c, lw=4 if nome == "mAP50" else 2, solid_capstyle="butt",
                    label=nome if t0 == g[0][0] else None)
            print(f"{nome}: {t0:.1f}-{t1:.1f} ms  {v:+.2f} punti / 10 ms")
    ax.set_ylabel("punti / 10 ms")

    ax = axs[3]
    B = args.budget
    cats = {"dentro": (color("fp16"), "intera già nel budget"), "potabile": (color("calib"), "arriva al budget potando"),
            "fuori": ("#9e9e9e", "non arriva al budget")}
    items, punti, segs = [], [], []
    for a in arch:
        if a not in floor:
            continue
        f, tipo = floor[a]
        k = "fuori" if B is None else "dentro" if lat[a] <= B else "potabile" if f <= SOGLIA * B else "fuori"
        col = cats[k][0] if B else color("rpi5")
        y = mAP[a][1]
        ax.plot([f, lat[a]], [y, y], color=col, lw=2, solid_capstyle="butt", zorder=3)
        ax.plot(lat[a], y, "o", color=col, mec="white", mew=0.5, ms=5, zorder=4)
        ax.plot(f, y, "D" if tipo == "misurato" else "o", mfc=col if tipo == "misurato" else "white", mec=col, ms=4.5 if tipo == "misurato" else 4, zorder=4)
        items.append((lat[a], y, sigla(a), col))
        segs.append(([f, lat[a]], [y, y]))
        punti += [(lat[a], y, 3.5), (f, y, 3.5)]
        print(f"{sigla(a):3s} {a:13s} pavimento {f:5.1f} ms ({tipo}) intera {lat[a]:5.1f} ms mAP50-95 {y:.3f} {k if B else ''}")
    h = [plt.Line2D([], [], color=ink, marker="D", ls="", ms=4.5, label="pavimento misurato"),
         plt.Line2D([], [], color=ink, marker="o", mfc="white", ls="", ms=4, label="pavimento stimato"),
         plt.Line2D([], [], color=ink, marker="o", ls="", ms=5, label="rete intera")]
    if B:
        for x, ls in ((B, "--"), (SOGLIA * B, ":")):
            for a_ in axs:
                a_.axvline(x, color=ink, lw=0.9, ls=ls, zorder=1)
        ex = " (esempio)" if args.esempio else ""
        h += [plt.Line2D([], [], color=ink, ls="--", lw=0.9, label=f"budget {B:g} ms{ex}"),
              plt.Line2D([], [], color=ink, ls=":", lw=0.9, label=f"{SOGLIA:.0%} del budget".replace("%", " %"))]
        h += [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in cats.values()]
    ax.set_ylabel("mAP50-95")
    ax.set_xscale("log")
    ticks = [20, 30, 50, 100]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(10, 175)
    ax.set_ylim(0.526, 0.614)
    ax.set_xlabel("latenza in ms (Raspberry Pi 5, 640)")
    for a_, l in zip(axs, "abcd"):
        a_.set_title(f"({l})", loc="left", fontsize=8, pad=2)
        a_.grid(axis="x", visible=False)
    axs[2].legend(loc="upper right", ncol=1)
    fig.legend(handles=h, loc="outside lower center", ncol=2, columnspacing=1.2, fontsize=7.5)
    etichette(ax, items, punti, segs)
    print(save(fig, Path(args.out) / "guadagno_latenza_640"))


if __name__ == "__main__":
    main()
