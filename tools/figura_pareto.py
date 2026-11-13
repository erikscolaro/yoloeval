# Fronte di Pareto mAP-latenza: un punto per ogni modello allenato della griglia, una serie per scheda con latenze
# misurate. Latenza su asse lineare, finestra centrata sulla fascia del budget del modello (figura A, figure_sintesi.py).
#   .venv/bin/python tools/figura_pareto.py OUT_DIR [--deploy 480p] [--precisione fp16] [--risultati results]
# Stile della tesi, una figura con mAP50 sopra e mAP50-95 sotto, PNG + PDF; titolo e note nella didascalia LaTeX.
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.figure_griglia import DEPTHS, LETTERA, WIDTHS, celle  # noqa: E402
from tools.figure_sintesi import A_PELO, BOARDS, BUDGET, lat_misurate  # noqa: E402
from tools.plots import STYLE, WIDTH, color, save  # noqa: E402

# sigle del capitolo 4: lettera della larghezza (LETTERA, figure_griglia) più il segno della profondità
SEGNO = {"D101": "-", "D126": "", "D170": "+"}
MARKERS = "osD^v"  # uno per scheda, in ordine di BOARDS: le serie restano distinguibili anche in bianco e nero
BAND = "#fbe7b8"   # giallo della fascia "a pelo" della figura A
# modello scelto per scheda: il migliore strettamente sotto il budget (conservativo) e il migliore strettamente sotto il
# budget più il margine (con margine): anelli dello stesso raggio, pieno e puntinato (la linea del budget è tratteggiata)
MARGIN = 0.1 * BUDGET
# riquadro bianco con il bordo del colore della serie: l'etichetta si lega alla scheda anche con più serie
LABEL = dict(fontsize=7, bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.4", fc="white", lw=0.7))
# posizioni candidate delle etichette, in punti dal centro del marker: prima vicine, poi più lontane
DIRS = [(dx * r, dy * r, "left" if dx > 0 else "right" if dx < 0 else "center",
         "bottom" if dy > 0 else "top" if dy < 0 else "center")
        for r in (5, 9) for dx, dy in ((-1, 1), (1, 1), (1, -1), (-1, -1), (0, 1.3), (0, -1.3), (1.3, 0), (-1.3, 0))]


def nome(w, d):
    return LETTERA[w] + SEGNO[d]


def fronte(pts):
    """Punti non dominati (latenza minore, mAP maggiore), in ordine di latenza."""
    out, top = [], -1.0
    for p in sorted(pts):
        if p[1] > top:
            out.append(p)
            top = p[1]
    return out


def etichette(ax, items, punti, linee):
    """Mette ogni etichetta (x, y, testo, colore) nella prima posizione di DIRS che resta dentro gli assi e non tocca marker
    (punti: (x, y, raggio in punti)), linee del fronte o etichette già messe; se nessuna è libera, la prima."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    box = ax.get_window_extent(r)
    occ = [Bbox.from_bounds(X - k * pt, Y - k * pt, 2 * k * pt, 2 * k * pt)
           for (X, Y), k in zip(ax.transData.transform([(x, y) for x, y, _ in punti]), [k for *_, k in punti])]
    for xs, ys in linee:  # campioni lungo i segmenti, come piccoli ostacoli
        for (X0, Y0), (X1, Y1) in zip(ax.transData.transform(list(zip(xs, ys)))[:-1],
                                      ax.transData.transform(list(zip(xs, ys)))[1:]):
            occ += [Bbox.from_bounds(X0 + (X1 - X0) * f - pt, Y0 + (Y1 - Y0) * f - pt, 2 * pt, 2 * pt)
                    for f in np.linspace(0, 1, 25)]
    for x, y, testo, col in items:
        stile = dict(LABEL, bbox=dict(LABEL["bbox"], ec=col))
        for i, (dx, dy, ha, va) in enumerate(DIRS):
            t = ax.annotate(testo, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            color=plt.rcParams["axes.labelcolor"], zorder=6, **stile)
            bb = t.get_window_extent(r).padded(2.1 * pt)  # il riquadro (pad 0,3 · 7 pt) sporge dal testo
            libera = box.containsx(bb.x0) and box.containsx(bb.x1) and box.containsy(bb.y0) and box.containsy(bb.y1)
            if libera and not any(bb.overlaps(o) for o in occ):
                occ.append(bb)
                break
            t.remove()
        else:
            dx, dy, ha, va = DIRS[0]
            ax.annotate(testo, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        color=plt.rcParams["axes.labelcolor"], zorder=6, **stile)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=".")
    ap.add_argument("--deploy", default="480p", help="ingresso delle latenze (risoluzione di deploy o ritaglio)")
    ap.add_argument("--precisione", default="fp16")
    ap.add_argument("--risultati", default=str(ROOT / "results"), help="cartella dei JSON del benchmark")
    args = ap.parse_args()
    plt.style.use(STYLE)
    ink = plt.rcParams["axes.edgecolor"]

    # shortcut: le mAP sono quelle del val a 640 anche per le altre risoluzioni, finché non ci sono i set di deploy
    maps = {(WIDTHS[i], DEPTHS[j]): v for (i, j), v in celle().items()}
    serie = {}
    for k, board in enumerate(BOARDS):
        lat = lat_misurate(args.risultati, board)
        pts = [(lat[w, args.deploy, d, args.precisione], v, w, d) for (w, d), v in maps.items()
               if (w, args.deploy, d, args.precisione) in lat]
        if pts:
            serie[board] = (k, pts)
    if not serie:
        sys.exit(f"nessuna latenza {args.precisione} a {args.deploy} in {args.risultati}")

    # finestra: dalla latenza minima meno un margine, simmetrica attorno al budget, così la fascia sta al centro
    x0 = min(min(p[0] for p in pts) for _, pts in serie.values())
    x0 = min(x0, (1 - A_PELO) * BUDGET) - MARGIN
    x1 = 2 * BUDGET - x0
    scelte = (("conservativa", BUDGET, 120, "-"), ("con margine", (1 + A_PELO) * BUDGET, 120, (0, (1, 1.2))))
    # una figura con mAP50 sopra (valori più alti) e mAP50-95 sotto, come i grafici della griglia
    fig, axs = plt.subplots(2, 1, figsize=(WIDTH, 1.05 * WIDTH), sharex=True)
    da_etichettare = []
    for ax, (name, m) in zip(axs, (("mAP50", 0), ("mAP50-95", 1))):
        ax.axvspan((1 - A_PELO) * BUDGET, (1 + A_PELO) * BUDGET, color=BAND, lw=0, zorder=0)
        handles = [ax.axvline(BUDGET, color=ink, lw=1, ls="--", zorder=1, label=f"budget {BUDGET:g} ms"),
                   matplotlib.patches.Patch(color=BAND, label=f"margine ± {A_PELO:.0%}".replace("%", " %"))]
        for lab, lim, size, ls in scelte:  # in legenda anelli più piccoli: a grandezza vera dominerebbero la riga
            handles.append(ax.scatter([], [], s=size / 2, facecolors="none", edgecolors=ink, linewidths=1.3, linestyles=ls,
                                      label=f"scelta {lab} (< {lim:.1f} ms)"))
        ys, items, punti, linee, fuori = [], [], [], [], []
        for board, (k, pts) in serie.items():
            pts = [(t, v[m], w, d) for t, v, w, d in pts]
            col, mk = color(board, k), MARKERS[k % len(MARKERS)]
            fr = fronte(pts)
            inside = [p for p in pts if p[0] <= x1]
            ys += [p[1] for p in inside]
            xs_f, ys_f = [p[0] for p in fr] + [x1], [p[1] for p in fr] + [fr[-1][1]]
            ax.plot(xs_f, ys_f, color=col, lw=1.4, drawstyle="steps-post", zorder=2)
            ax.fill_between(xs_f, ys_f, -1, step="post", color=col, alpha=0.06, lw=0, zorder=1)  # zona dominata
            # gradini del fronte come segmenti, per tenere le etichette lontane dalla linea
            sx, sy = [xs_f[0]], [ys_f[0]]
            for a, b in zip(range(len(xs_f) - 1), range(1, len(xs_f))):
                sx += [xs_f[b], xs_f[b]]; sy += [ys_f[a], ys_f[b]]
            linee.append((sx, sy))
            dom = [p for p in inside if p not in fr]
            ax.plot([p[0] for p in dom], [p[1] for p in dom], mk, mfc="white", mec=col, mew=1, alpha=0.7, zorder=3)
            handles += ax.plot([p[0] for p in fr], [p[1] for p in fr], mk, color=col, mec="white", mew=0.6, ms=6,
                               zorder=4, label=BOARDS[board])
            punti += [(p[0], p[1], 3.5) for p in inside]
            items += [(t, y, nome(w, d), col) for t, y, w, d in fr if t <= x1]
            for lab, lim, size, ls in scelte:
                ok = [p for p in pts if p[0] < lim]
                if ok:
                    t, y, w, d = max(ok, key=lambda p: p[1])
                    ax.scatter(t, y, s=size, facecolors="none", edgecolors=ink, linewidths=1.3, linestyles=ls, zorder=5)
                    punti.append((t, y, (size ** 0.5) / 2 + 1))
                    print(f"{board} {name} scelta {lab}: {nome(w, d)} {t:.1f} ms {y:.3f}")
            if len(pts) > len(inside):
                fuori.append(f"{BOARDS[board]}: {len(pts) - len(inside)} modelli oltre {x1:.0f} ms →")
        if fuori:  # i modelli fuori dalla finestra non spariscono in silenzio
            ax.text(0.99, 0.02, "\n".join(fuori), transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    color=ink)
        ax.set_xlim(x0, x1)
        ax.set_ylim(min(ys) - 0.012, max(ys) + 0.012)
        ax.set_ylabel(name)
        ax.grid(axis="x", visible=False)  # fascia e budget bastano a leggere le latenze
        da_etichettare.append((ax, items, punti, linee))
    axs[-1].set_xlabel(f"latenza in ms ({args.deploy}, {args.precisione})")
    # una legenda sotto i due pannelli: dentro coprirebbe la fascia del budget, che sta al centro
    fig.legend(handles=handles, loc="outside lower center", ncol=3, columnspacing=1.2, scatteryoffsets=[0.5],
               labelspacing=0.9)
    for args_et in da_etichettare:  # dopo la legenda, che sposta gli assi: le collisioni si calcolano sul layout finale
        etichette(*args_et)
    print(save(fig, Path(args.out) / f"pareto_{args.deploy}"))
