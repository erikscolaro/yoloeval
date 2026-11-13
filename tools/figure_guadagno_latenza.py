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
from tools.figura_pareto import fronte  # noqa: E402
from tools.figure_griglia import LETTERA  # noqa: E402
from tools.plots import GRAY, STYLE, color, save  # noqa: E402

COLUMN = 3.5
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
    fig, axs = plt.subplots(2, 1, figsize=(COLUMN, 8.2), sharex=True, height_ratios=[1.7, 2.6])
    c50, c5095 = color("map50", 1), color("map50_95", 0)  # una tinta per metrica

    # (a) guadagno marginale fra punti consecutivi del fronte; la sigla sopra la barra e' il modello che si raggiunge
    # (fronte della mAP50-95; quello della mAP50 ha gli stessi estremi meno qualche passo)
    ax = axs[0]
    fr95 = fronte([(lat[a], mAP[a][1], a) for a in arch])
    for m, c, nome, lw in ((0, c50, "mAP50", 4), (1, c5095, "mAP50-95", 2)):
        fr = fronte([(lat[a], mAP[a][m]) for a in arch])
        for k, (t0, t1, v) in enumerate(guadagno(fr)):
            ax.plot([t0, t1], [v, v], color=c, lw=lw, solid_capstyle="butt", label=nome if k == 0 else None)
            print(f"{nome}: {t0:.1f}-{t1:.1f} ms  {v:+.2f} punti / 10 ms")
    ax.set_ylabel("punti / 10 ms")
    ax.set_ylim(0, 14)
    ax.legend(loc="upper right")

    # (b) spazio di potatura: una riga per architettura, in ordine di mAP50-95; la mAP e' scritta a fine riga
    ax = axs[1]
    B = args.budget
    cats = {"dentro": (color("fp16"), "verde: già nel budget senza potare"), "potabile": (color("calib"), "arancione: ci arriva con la potatura"),
            "fuori": (GRAY, "grigio: non adatto (anche potato supera il limite)")}
    rows = sorted((a for a in arch if a in floor), key=lambda a: mAP[a][1])
    for r, a in enumerate(rows):
        f, tipo = floor[a]
        k = "fuori" if B is None else "dentro" if lat[a] <= B else "potabile" if f <= SOGLIA * B else "fuori"
        col = cats[k][0] if B else color("rpi5")
        ax.plot([f, lat[a]], [r, r], color=col, lw=2, solid_capstyle="butt", zorder=3)
        ax.plot(lat[a], r, "o", color=col, mec="white", mew=0.5, ms=5, zorder=4)
        ax.plot(f, r, "D" if tipo == "misurato" else "o", mfc=col if tipo == "misurato" else "white", mec=col,
                ms=4.5 if tipo == "misurato" else 4, zorder=4)
        ax.annotate(f"{mAP[a][1]:.3f}".replace(".", ","), (lat[a], r), xytext=(5, 0), textcoords="offset points",
                    va="center")
        print(f"{sigla(a):3s} {a:13s} pavimento {f:5.1f} ms ({tipo}) intera {lat[a]:5.1f} ms mAP50-95 {mAP[a][1]:.3f} {k if B else ''}")
    ax.set_yticks(range(len(rows)), [sigla(a) for a in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_ylabel("modello, per mAP50-95 crescente")
    blank = lambda t: plt.Line2D([], [], ls="", label=t)  # titolo di gruppo
    h = [blank("Latenza del modello"),
         plt.Line2D([], [], color=ink, marker="o", ls="", ms=5, label="rete intera (cifra: mAP50-95)"),
         plt.Line2D([], [], color=ink, marker="o", mfc="white", ls="", ms=4, label="potata al massimo (stima da LUT)"),
         plt.Line2D([], [], color=ink, marker="D", ls="", ms=4.5, label="potata al massimo (misurata)")]
    if B:
        for x, ls in ((B, "--"), (SOGLIA * B, ":")):
            for a_ in axs:
                a_.axvline(x, color=ink, lw=0.9, ls=ls, zorder=1)
        for x, ha, txt in ((SOGLIA * B, "right", "limite"), (B, "left", "budget")):  # etichette dirette
            axs[0].annotate(txt, (x, 1), xycoords=("data", "axes fraction"), xytext=(-1 if ha == "right" else 1, 1.5),
                            textcoords="offset points", ha=ha, va="bottom")
        ex = " (esempio)" if args.esempio else ""
        h += [blank("Linee"),
              plt.Line2D([], [], color=ink, ls="--", lw=0.9, label=f"budget {B:g} ms{ex}"),
              plt.Line2D([], [], color=ink, ls=":", lw=0.9,
                         label=f"limite per il potato al massimo ({SOGLIA:.0%} del budget)".replace("%", " %")),
              blank("Colore"), *[plt.Line2D([], [], color=c, lw=2, label=l) for c, l in cats.values()]]
    ax.set_xscale("log")
    ticks = [20, 30, 50, 100]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(10, 230)
    ax.set_xlabel("latenza in ms (Raspberry Pi 5, 640)")
    for a_, l in zip(axs, "ab"):
        a_.set_title(f"({l})", loc="left", pad=2)
        a_.grid(axis="x", visible=False)
    leg = fig.legend(handles=h, loc="outside lower center", ncol=1, handlelength=1.6, labelspacing=0.3)
    for t, hd in zip(leg.get_texts(), h):  # i titoli di gruppo non hanno simbolo
        if not hd.get_marker() or hd.get_marker() == "None":
            if not hd.get_linestyle() or hd.get_linestyle() == "None":
                t.set_fontweight("bold")
    fig.canvas.draw()  # posizioni definitive degli assi, per i livelli delle etichette di (a)
    fs = plt.rcParams["font.size"]
    placed = []  # rettangoli (punti) delle etichette gia' messe: ognuna sale finche' non ne tocca un'altra
    for (t0, y0, _), (t1, y1, a) in zip(fr95, fr95[1:]):
        v, xm = 100 * (y1 - y0) / (t1 - t0) * 10, np.sqrt(t0 * t1)
        X, Y = axs[0].transData.transform((xm, v)) / (fig.dpi / 72)
        w, h = 0.75 * fs * (len(sigla(a)) + 1), 1.3 * fs
        for lev in range(12):
            dy = 8 + 0.8 * fs * lev
            box = (X - w / 2, Y + dy, X + w / 2, Y + dy + h)
            if not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in placed):
                break
        placed.append(box)
        axs[0].annotate("\u2192" + sigla(a), (xm, v), xytext=(0, dy), textcoords="offset points", ha="center",
                        arrowprops=dict(arrowstyle="-", color=ink, lw=0.4, shrinkA=0, shrinkB=1) if lev else None)
    print(save(fig, Path(args.out) / "guadagno_latenza_640"))

if __name__ == "__main__":
    main()
