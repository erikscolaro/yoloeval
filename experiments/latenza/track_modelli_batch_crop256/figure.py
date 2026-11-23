"""Figure del batch: N crop 256x256 in una sola inferenza (crop tracker, fotogramma TRACK), modelli della griglia.
Budget del modello a 30 fps per N crop = 33,3 ms meno il costo fuori dal modello misurato in ../track_budget_n_crop
(ritaglio, Kalman, pre e post per N crop). Uscite in report/:
    A_batch_serie:     un pannello per profondita', x = batch, y = latenza dell'inferenza, una serie per larghezza
    A_crop_serie:      lo stesso con latenza / N (per immagine); il budget diventa budget(N) / N
    A_pareto:          un pannello per batch, mAP50-95 (val, 640, training e300) contro latenza, solo i modelli entro il
                       budget +5%, le tre profondita' nello stesso pannello, fronte di Pareto. M non c'e': non e'
                       addestrata (costa troppo per il deploy)
    latenza_batch.csv: i numeri
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import tools.figure_sintesi as fs  # noqa: E402

B = {"crop256_b1": 1, "crop256_b2": 2, "crop256_b4": 4, "crop256_b8": 8}
YMAX = 40         # ms: oltre non serve, il budget e' sotto
FRAME_MS = 1000 / 30
EXTRA = HERE.parent / "track_budget_n_crop/results/crop_kalman.jsonl"


def budget():
    """{N: ms al modello} = 33,3 - mediana fra le passate del costo fuori dal modello con N crop."""
    tot = {}
    for line in open(EXTRA):
        r = json.loads(line)
        tot.setdefault(r["n"], []).append(r["totale"]["median_ms"])
    return {n: FRAME_MS - float(np.median(v)) for n, v in tot.items()}


def ramp(plt):
    # sequenziale a piu' tinte (plasma, ordinata per luminosita'): larghezza piccola chiara, grande scura;
    # blu e viridis non separavano le larghezze grandi. Il giallo estremo e' escluso, sul bianco non si vede.
    return plt.cm.plasma(np.linspace(0.88, 0.0, 9))


def main():
    lat = {k: v for k, v in fs.lat_misurate(str(ROOT / "results"), "rpi5").items()
           if k[1] in B and k[3] == "fp16"}
    bud = budget()
    out = HERE / "report"
    per_crop = {k: v / B[k[1]] for k, v in lat.items()}
    figura_serie(out, lat, bud)
    figura_serie(out, per_crop, bud, "A_crop_serie", "latenza per immagine [ms]", per_img=True)
    figura_pareto(out, lat, bud)
    with open(out / "latenza_batch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["width", "depth", "batch", "median_ms", "ms_per_crop", "budget_ms"])
        for (wd, r, d, _), v in sorted(lat.items()):
            w.writerow([wd, d, B[r], round(v, 2), round(v / B[r], 2), round(bud[B[r]], 2)])


def figura_serie(out, lat, bud, nome="A_batch_serie", ylab="latenza dell'inferenza [ms]", per_img=False):
    """Un pannello per profondita', x = batch, y = latenza (lineari, y fino a YMAX), una serie per larghezza.
    Stile della tesi (tools/thesis.mplstyle)."""
    import matplotlib.pyplot as plt
    from src.selection import LETTERA
    from tools.plots import GRAY, STYLE, WIDTH, save
    widths = sorted({k[0] for k in lat})
    depths = sorted({k[2] for k in lat})
    bs = sorted(B.values())
    # budget a 30 fps +-5% (grigio = riferimento); il costo fuori dal modello e' lineare in N, quindi fra i punti
    # misurati si interpola. Per immagine e' diviso fra le N immagini.
    x = np.linspace(1, 8, 100)
    t = np.interp(x, bs, [bud[n] for n in bs])
    if per_img:
        t = t / x
    with plt.style.context(STYLE):
        fig, axs = plt.subplots(1, len(depths), figsize=(WIDTH, 2.6), sharey=True)
        for ax, d in zip(axs, depths):
            ax.fill_between(x, (1 - fs.A_PELO) * t, (1 + fs.A_PELO) * t, color=GRAY, alpha=0.25, lw=0,
                            label=f"budget a 30 fps{' / N' if per_img else ''} ±{fs.A_PELO:.0%}")
            ax.plot(x, t, "--", color=GRAY, lw=1)
            for i, (w, c) in enumerate(zip(widths, ramp(plt))):
                ys = [lat.get((w, r, d, "fp16")) for r in B]
                # forma a rotazione: due larghezze vicine non hanno mai lo stesso segno
                vis = any(y <= YMAX for d2 in depths for y in (lat.get((w, r, d2, "fp16")) for r in B))
                # in legenda solo le larghezze che si vedono in almeno un pannello
                ax.plot(bs, ys, "-", marker="os^"[i % 3], color=c, lw=1.3, ms=3,  # tre pannelli: segni piu' fini dello stile
                        label=LETTERA[w] if vis else "_")
            ax.set_ylim(0, YMAX)
            ax.set_xticks(bs, [str(b) for b in bs])
            ax.set_title(d)
            ax.set_xlabel("batch")
        axs[0].set_ylabel(ylab)
        h, lab = axs[0].get_legend_handles_labels()
        fig.legend(h[1:] + h[:1], lab[1:] + lab[:1], loc="outside lower center", ncol=5)
        save(fig, out / nome)


def figura_pareto(out, lat, bud):
    """Una figura, un pannello per batch (2x2), nello stile di tools/figura_pareto.py: x = latenza dell'inferenza,
    y = mAP50-95 (media fra i semi), solo i modelli entro il budget +5%. Un solo fronte fra le tre profondita'; colore e
    forma = profondita'; punti del fronte pieni ed etichettati, dominati vuoti. Una legenda sola."""
    import matplotlib
    import matplotlib.pyplot as plt
    from src.selection import arch_di, map_modelli
    import tools.figura_pareto as fp
    from tools.figura_pareto import BAND, etichette, fronte, nome
    # quattro pannelli in una larghezza di testo: riquadri delle etichette piu' piccoli, se no non trovano posto
    fp.LABEL = dict(fp.LABEL, fontsize=5.5, bbox=dict(fp.LABEL["bbox"], boxstyle="round,pad=0.15,rounding_size=0.3"))
    from tools.plots import STYLE, WIDTH, color, save
    mp = {k: float(np.mean([m[1] for m in v])) for k, v in map_modelli(ROOT / "artifacts/weights").items()}
    depths = ["D101", "D126", "D170"]
    forma = dict(zip(depths, "os^"))
    col = {d: color(d, i) for i, d in enumerate(depths)}
    visti = set()
    with plt.style.context(STYLE):
        ink = plt.rcParams["axes.edgecolor"]
        fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 0.95 * WIDTH), sharey=True)
        da_etichettare = []
        for ax, (r, n) in zip(axs.flat, B.items()):
            lim = (1 + fs.A_PELO) * bud[n]
            pts = [(lat[k], mp[arch_di(k[0], k[2])], k[0], k[2]) for k in lat
                   if k[1] == r and arch_di(k[0], k[2]) in mp and lat[k] <= lim]
            fr = fronte(pts)
            x1 = lim * 1.08
            ax.axvspan((1 - fs.A_PELO) * bud[n], lim, color=BAND, lw=0, zorder=0)
            ax.axvline(bud[n], color=ink, lw=1, ls="--", zorder=1)
            xs_f, ys_f = [p[0] for p in fr] + [x1], [p[1] for p in fr] + [fr[-1][1]]
            ax.plot(xs_f, ys_f, color=ink, lw=1.2, drawstyle="steps-post", zorder=2)
            ax.fill_between(xs_f, ys_f, -1, step="post", color=ink, alpha=0.06, lw=0, zorder=1)  # zona dominata
            sx, sy = [xs_f[0]], [ys_f[0]]
            for a in range(len(xs_f) - 1):
                sx += [xs_f[a + 1], xs_f[a + 1]]
                sy += [ys_f[a], ys_f[a + 1]]
            for t, m, w, d in pts:
                visti.add(d)
                on = (t, m, w, d) in fr
                ax.plot(t, m, forma[d], ms=5 if on else 4.5, zorder=4 if on else 3,
                        **(dict(color=col[d], mec="white", mew=0.6) if on else dict(mfc="white", mec=col[d], mew=1, alpha=0.7)))
            ax.set_xlim(0, x1)
            ax.set_ylim(0.52, 0.63)        # uguale per tutti i batch, per confrontarli
            ax.set_title(f"batch {n}: budget {bud[n]:.1f} ms".replace(".", ","), fontsize=9)
            ax.set_xlabel("latenza dell'inferenza [ms]")
            ax.grid(axis="x", visible=False)  # fascia e budget bastano a leggere le latenze
            da_etichettare.append((ax, [(t, m, nome(w, d), col[d]) for t, m, w, d in fr],
                                   [(t, m, 3.5) for t, m, *_ in pts], [(sx, sy)]))
        for ax in axs[:, 0]:
            ax.set_ylabel("mAP50-95")
        hd = [plt.Line2D([], [], ls="", marker=forma[d], ms=6, color=col[d], mec="white", label=d) for d in depths if d in visti]
        hd += [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=ink, label="dominato"),
               plt.Line2D([], [], color=ink, ls="--", lw=1, label="budget a 30 fps"),
               matplotlib.patches.Patch(color=BAND, label=f"margine ± {fs.A_PELO:.0%}".replace("%", " %"))]
        fig.legend(handles=hd, loc="outside lower center", ncol=3, columnspacing=1.2)
        for a in da_etichettare:           # dopo la legenda, che sposta gli assi
            etichette(*a)
        save(fig, out / "A_pareto")

if __name__ == "__main__":
    main()
