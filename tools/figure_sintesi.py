"""Figure di sintesi del blocco 1 (NEW_MASTERPLAN.md, sezione 13), una per dispositivo, a piccoli multipli:
righe = precisione (fp16, int8), colonne = profondità (101, 126, 170 conv), in ogni pannello la griglia
discreta larghezza × risoluzione di deploy (480p, 720p, 1080p).

    A. chi sta nel budget         (dopo la griglia di latenza, fase 2a)
    B. dove sta l'ottimo          (dopo il modello di precisione, fase 2c)
    C. candidati alla potatura    (alla decisione 2d, prima del blocco 2)

    python tools/figure_sintesi.py OUT_DIR

I DATI DI ESEMPIO QUI SOTTO SONO INVENTATI (tranne femto e pico a 640 sul Pi, 18,2 e 27,9 ms): vanno sostituiti da
`lat(w, r, d, q)` misurata (fase 2a) e `mAP(w, r, d, q)` del modello di precisione sul set di val della risoluzione di
deploy `r` (fase 2c).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.plots import STYLE, WIDTH, save  # noqa: E402

# --- valori ammessi (sezione 4 del masterplan) --------------------------------------------------------------------
WL = np.array([0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0])
WLAB = ["F", "", "P", "", "N", "", "S", "", "M"]
RL = ["480p", "720p", "1080p"]
RPIX = {"480p": (864, 480), "720p": (1280, 736), "1080p": (1920, 1088)}  # ingresso del modello, multipli di 32
DEPTHS = ["D101", "D126", "D170"]  # profondità = numero di conv (masterplan, sezione 4)
PRECS = ["fp16", "int8"]

# --- parametri delle figure -----------------------------------------------------------------------------------------
BUDGET = 28.0      # ms al modello: 33 ms meno pre/post/trasferimenti (fase 1)
A_PELO = 0.05      # margine "a pelo" attorno al budget
NOISE = 0.01       # 1 punto di mAP50-95: sotto, due modelli sono a pari merito (criterio di scelta, 2c)
MAXCUT = 0.5       # taglio di latenza massimo realistico per la potatura

# --- dati di esempio, INVENTATI -------------------------------------------------------------------------------------
_W = np.array([0.0625, 0.125, 0.25, 0.5, 1.0])
_GMAC640 = np.array([0.245, 0.735, 2.66, 10.0, 37.0])
_MAP640 = np.array([0.40, 0.45, 0.49, 0.51, 0.515])
_DEPTH = {"D101": (0.90, 0.976, -0.005), "D126": (1.0, 1.0, 0.0), "D170": (1.15, 1.23, 0.006)}
_PREC = {"fp16": (1.0, 0.0), "int8": (0.65, -0.010)}


def _area(r):
    return RPIX[r][0] * RPIX[r][1] / 640 ** 2


def lat(w, r, d, q):
    fixed, gm, _ = _DEPTH[d]
    gmac = np.exp(np.interp(np.log2(w), np.log2(_W), np.log(_GMAC640))) * _area(r)
    return 13.35 * fixed + 19.8 * _PREC[q][0] * gm * gmac


def mAP(w, r, d, q):
    # più cielo, più falsi allarmi: poco, e di più per i modelli piccoli
    return (np.interp(np.log2(w), np.log2(_W), _MAP640) - 0.002 * np.log2(_area(r)) * (0.5 / w) ** 0.3
            + _DEPTH[d][2] + _PREC[q][1])


# --- figure ---------------------------------------------------------------------------------------------------------
WTICK = {0.0625: "1/16", 0.125: "1/8", 0.25: "1/4", 0.5: "1/2", 1.0: "1"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#d6d6d6"
OPT, PRUNE = "#d9531e", "#8a5d00"


def _frame():
    """Piccoli multipli a passo uguale in orizzontale e in verticale (aspetto 1:1 della griglia)."""
    fig, axs = plt.subplots(len(PRECS), len(DEPTHS), figsize=(WIDTH, (0.08 + 0.13 * len(PRECS)) * WIDTH), sharex=True,
                            sharey=True, squeeze=False)
    for i, q in enumerate(PRECS):
        for j, d in enumerate(DEPTHS):
            ax = axs[i, j]
            ax.grid(False)
            ax.set_xlim(-0.6, len(WL) - 0.4); ax.set_ylim(-0.6, len(RL) - 0.4)
            ax.set_aspect("equal")
            ax.set_xticks(range(len(WL)), [WTICK.get(w, "") for w in WL], fontsize=7)
            ax.set_yticks(range(len(RL)), RL, fontsize=7)
            ax.tick_params(length=0, colors=MUTED)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"Profondità: {d[1:]} conv", fontsize=8)
            if j == 0:
                ax.set_ylabel(q.upper(), fontsize=8)
            if i == len(PRECS) - 1:
                ax.set_xlabel("Moltiplicatore di larghezza", fontsize=7.5)
    return fig, axs


def _cells():
    for i, q in enumerate(PRECS):
        for j, d in enumerate(DEPTHS):
            for a, r in enumerate(RL):
                for b, w in enumerate(WL):
                    yield i, j, q, d, a, r, b, w


def _optimum():
    """Ottimo entro il budget per ogni risoluzione di deploy (fra tutte le precisioni e profondità): la mAP non
    premia la risoluzione, quindi non c'è un ottimo globale fra risoluzioni."""
    best = {}
    for i, j, q, d, a, r, b, w in _cells():
        if lat(w, r, d, q) <= BUDGET:
            best[r] = max(best.get(r, (-1, 0, 0, 0, 0)), (mAP(w, r, d, q), i, j, b, a))
    return best


def _comma(x, f="{:.3f}"):
    return f.format(x).replace(".", ",")


def _header(fig, handles, labels, title, ncol):
    """Titolo in alto, legenda sotto le figure."""
    fig.suptitle(title, fontsize=8.5)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0), fontsize=7,
               handletextpad=0.4, columnspacing=1.5, frameon=False)


def figura_a(out, device):
    fills = ["#cfeee0", "#fbe7b8", "#eeeeee"]
    fig, axs = _frame()
    for i, j, q, d, a, r, b, w in _cells():
        t = lat(w, r, d, q)
        k = 0 if t <= (1 - A_PELO) * BUDGET else 1 if t <= (1 + A_PELO) * BUDGET else 2
        axs[i, j].add_patch(plt.Rectangle((b - 0.42, a - 0.42), 0.84, 0.84, facecolor=fills[k], edgecolor="none"))
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in fills]
    _header(fig, h, ["Entro il budget", f"Entro ±{A_PELO:.0%} dal budget", "Oltre il budget"],
            f"{device}: configurazioni entro il budget di latenza ({BUDGET:.0f} ms)".replace(".", ","), 3)
    return save(fig, Path(out) / "A_budget", pdf=False)


def _points(axs, norm, cmap, rings):
    best = _optimum()
    for i, j, q, d, a, r, b, w in _cells():
        t, m = lat(w, r, d, q), mAP(w, r, d, q)
        if t <= BUDGET:
            axs[i, j].scatter(b, a, s=26, color=cmap(norm(m)), edgecolor="white", linewidth=0.5, zorder=3)
        elif rings and r in best and m > best[r][0] + NOISE and 1 - BUDGET / t <= MAXCUT:
            axs[i, j].scatter(b, a, s=24, facecolor="white", edgecolor=cmap(norm(m)), linewidth=1.5, zorder=3)
        else:
            axs[i, j].scatter(b, a, s=4, color=GRID, zorder=2)


def _stars(axs):
    best = _optimum()
    for m, i, j, b, a in best.values():
        axs[i, j].scatter(b, a, marker="*", s=110, color=OPT, edgecolor=INK, linewidth=0.7, zorder=5)
    return best


def _colorbar(fig, axs, norm, cmap):
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, pad=0.015, shrink=0.85,
                      aspect=15, extend="both")
    cb.set_label("mAP50-95 (predetta)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _comma(v, "{:.2f}")))


def figura_b(out, device):
    norm, cmap = matplotlib.colors.Normalize(0.36, 0.48), plt.get_cmap("Blues")
    fig, axs = _frame()
    _points(axs, norm, cmap, rings=False)
    _stars(axs)
    _colorbar(fig, axs, norm, cmap)
    h = [plt.Line2D([], [], marker="o", ls="", mfc=cmap(0.6), mec="white", ms=5),
         plt.Line2D([], [], marker="o", ls="", mfc=GRID, mec=GRID, ms=2.5),
         plt.Line2D([], [], marker="*", ls="", mfc=OPT, mec=INK, ms=10)]
    _header(fig, h, ["Entro il budget", "Oltre il budget", "Ottimo della risoluzione"],
            f"{device}: ottimo entro il budget, per risoluzione di deploy", 3)
    return save(fig, Path(out) / "B_ottimo", pdf=False)


def figura_c(out, device):
    norm, cmap = matplotlib.colors.Normalize(0.36, 0.50), plt.get_cmap("Blues")
    best = _optimum()
    cands = {}  # miglior candidato per risoluzione: più mAP per punto di taglio rispetto all'ottimo della stessa
    for i, j, q, d, a, r, b, w in _cells():
        t, m = lat(w, r, d, q), mAP(w, r, d, q)
        cut = 1 - BUDGET / t
        if r in best and t > BUDGET and m > best[r][0] + NOISE and cut <= MAXCUT:
            cands[r] = max(cands.get(r, (-1,)), ((m - best[r][0]) / cut, i, j, b, a, m, cut))
    fig, axs = _frame()
    _points(axs, norm, cmap, rings=True)
    _stars(axs)
    _colorbar(fig, axs, norm, cmap)
    for _, i, j, b, a, m, cut in cands.values():
        ax = axs[i, j]
        ax.scatter(b, a, marker="D", s=55, facecolor="none", edgecolor=PRUNE, linewidth=1.3, zorder=6)
        # una riga per risoluzione: l'etichetta (taglio di latenza da fare) sta a destra, sulla stessa riga
        ax.text(b + 0.5, a, f"−{cut:.0%}", fontsize=5.5, color=INK, va="center", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    h = [plt.Line2D([], [], marker="o", ls="", mfc=cmap(0.6), mec="white", ms=5),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec=cmap(0.75), mew=1.5, ms=5),
         plt.Line2D([], [], marker="o", ls="", mfc=GRID, mec=GRID, ms=2.5),
         plt.Line2D([], [], marker="*", ls="", mfc=OPT, mec=INK, ms=10),
         plt.Line2D([], [], marker="D", ls="", mfc="none", mec=PRUNE, mew=1.3, ms=6)]
    _header(fig, h, ["Entro il budget",
                     f"Candidato alla potatura (ΔmAP > {_comma(NOISE * 100, '{:.1f}')} punti, riduzione ≤ {MAXCUT:.0%})",
                     "Oltre il budget", "Ottimo della risoluzione", "Candidato selezionato (taglio di latenza)"],
            f"{device}: candidati alla potatura, per risoluzione di deploy", 3)
    return save(fig, Path(out) / "C_potabili", pdf=False)


if __name__ == "__main__":
    plt.style.use(STYLE)
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for f in (figura_a, figura_b, figura_c):
        print(f(out, "Raspberry Pi 5, dati di esempio"))
