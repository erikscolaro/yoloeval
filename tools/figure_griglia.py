# Griglia larghezza × profondità: mAP50 e mAP50-95 (miglior epoca sul val), a barre 3D e a linee (una per profondità).
#   .venv/bin/python tools/figure_griglia.py OUT_DIR   (dalla radice di yoloeval)
# Stile della tesi (tools/thesis.mplstyle), una figura per pannello a tutta \textwidth, PNG + PDF; titolo e note nella
# didascalia LaTeX. Rileggibile mentre la griglia gira: le celle senza meta.json sono in corso (valore migliore finora,
# barre tratteggiate e marker vuoti).
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plots import STYLE, WIDTH, color, save  # noqa: E402

W = ROOT / "artifacts/weights"
# sigle delle larghezze (capitolo 4): le intermedie prendono le lettere delle due vicine
LETTERA = {0.0625: "F", 0.09375: "FP", 0.125: "P", 0.1875: "PN", 0.25: "N", 0.375: "NS", 0.5: "S", 0.75: "SM", 1.0: "M"}
WIDTHS = [0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75]
DEPTHS = ["D101", "D126", "D170"]
# pico, nano e small D126 allenati prima della griglia con la stessa ricetta
OLD = {(0.125, "D126"): "yolo26p_D126_air_detection_all_e300_59d8534a",
       (0.25, "D126"): "yolo26n_D126_air_detection_all_e300_9dca8d15",
       (0.5, "D126"): "yolo26s_D126_air_detection_all_e300_9ada3b33"}
# contorni delle barre dal giallo (basso) al blu (alto)
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("neon", ["#FFF200", "#00FF9C", "#00E5FF", "#2F6BFF"])
TINT = 0.15  # facce delle barre: bianco con un velo del colore del contorno, per la stampa
SOFT = 0.35  # gradiente della legenda: schiarito verso il bianco, a metà fra contorni e facce delle barre
CMAP_SOFT = matplotlib.colors.ListedColormap(1 - (1 - SOFT) * (1 - CMAP(np.linspace(0, 1, 256))[:, :3]))
MIN_EP = 100  # celle in corso con meno epoche: il valore è ancora lontano dal plateau e falserebbe la scala dei colori
ELEV, AZIM = 22, -125
ANG_X, ANG_Y = 15, 30  # inclinazione a schermo di larghezza e profondità sull'orizzontale
GAP = 0.4  # distanza fra l'ultima fila di barre e le barre di variazione sulla parete di fondo
# profondità: palette categorica della tesi (blu, arancio, acqua) più tratto e simbolo, leggibili anche in bianco e nero
LINE = {d: (color(d, j), ls, mk) for j, (d, ls, mk) in enumerate(zip(DEPTHS, ("-", "--", ":"), "os^"))}
plt.style.use(STYLE)
INK = plt.rcParams["axes.edgecolor"]


def obliqua(ax, ax_x, ax_y):
    """Assonometria obliqua al posto della vista di mplot3d: asse z verticale, larghezza (x) inclinata di ax_x gradi
    verso destra e profondità (y) di ax_y gradi verso sinistra rispetto all'orizzontale, con la mAP sempre verticale,
    cosa che la vista ortogonale di mplot3d fa solo con la prospettiva. La terza riga resta quella della
    vista ortogonale di elev e azim, che mplot3d usa per scegliere le pareti di fondo e ordinare le facce."""
    orig = ax.get_proj
    a, b = np.radians(ax_x), np.radians(ax_y)
    u, v = np.array([np.cos(a), -np.cos(b), 0]), np.array([np.sin(a), np.sin(b), 1])

    def get_proj():
        orig()  # calcola ax._view_w
        R = 0.5 * np.asarray(ax._box_aspect)
        worldM = proj3d.world_transformation(*ax.get_xlim3d(), *ax.get_ylim3d(), *ax.get_zlim3d(),
                                             pb_aspect=ax._box_aspect)
        w = ax._view_w
        viewM = np.eye(4)
        viewM[:3, :3] = [u, v, w]
        viewM[:3, 3] = [-u @ R, -v @ R, -w @ R - ax._dist]
        return proj3d._ortho_transformation(-ax._dist, ax._dist) @ viewM @ worldM

    ax.set_proj_type("ortho")
    ax.get_proj = get_proj


def best(d):
    r = list(csv.DictReader(open(d / "run/results.csv")))
    k = lambda s: next(c for c in r[0] if c.strip() == s)
    b = max(r, key=lambda x: float(x[k("metrics/mAP50-95(B)")]))  # come Ultralytics 8.4: fitness = mAP50-95
    done = (d / "meta.json").exists()
    if not done and len(r) < MIN_EP:
        return None
    return float(b[k("metrics/mAP50(B)")]), float(b[k("metrics/mAP50-95(B)")]), done


def celle():
    """{(indice larghezza, indice profondità): (mAP50, mAP50-95, finito)} dei training della griglia."""
    cells = {}
    for i, w in enumerate(WIDTHS):
        for j, d in enumerate(DEPTHS):
            p = W / OLD[w, d] if (w, d) in OLD else next(iter(W.glob(f"yolo26w{w}_{d}_air_detection_all_e300_*")), None)
            if p is not None and (p / "run/results.csv").exists():
                if (c := best(p)) is not None:
                    cells[i, j] = c

    return cells


if __name__ == "__main__":
    OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cells = celle()
    ij = np.array(list(cells))
    for name, m in (("mAP50", 0), ("mAP50-95", 1)):
        v = np.array([cells[c][m] for c in cells])
        done = np.array([cells[c][2] for c in cells])
        norm = plt.Normalize(v.min(), v.max())
        lo = v.min() - 0.02
        zt = np.arange(np.ceil(lo * 100), np.floor((v.max() + 0.01) * 100) + 1) / 100  # griglia ogni 0,01, etichette ogni 0,02
        zl = [f"{z:.2f}" if round(z * 100) % 2 == 0 else "" for z in zt]
        stem = f"griglia_{name.replace('-', '_')}"

        # barre 3D: una chiamata per barra, dalla più lontana alla più vicina alla camera
        fig = plt.figure(figsize=(WIDTH, 0.75 * WIDTH), layout="none")  # constrained_layout non regge gli assi 3D
        ax = fig.add_axes((0.06, 0.1, 0.92, 0.9), projection="3d", computed_zorder=False)  # disegno nell'ordine di chiamata
        # profondità in ordine inverso, D170 davanti e D101 in fondo: le barre più basse stanno vicino alla parete
        pos = np.array([(i, len(DEPTHS) - 1 - j) for i, j in ij])
        # sulla parete di fondo, parallela all'asse della larghezza e staccata dalle barre di GAP: per ogni larghezza
        # minimo e massimo fra le profondità, come una barra di variazione con il delta scritto sopra; disegnata prima
        # delle barre, che ci passano davanti
        yb, cap = len(DEPTHS) - 0.5 + GAP, 0.06
        for i in sorted({i for i, _ in ij}):
            zs = [z for (ii, _), z in zip(ij, v) if ii == i]
            if len(zs) > 1:
                ax.plot([i, i], [yb, yb], [min(zs), max(zs)], color=INK, lw=0.6)
                for z in (min(zs), max(zs)):
                    ax.plot([i - cap, i + cap], [yb, yb], [z, z], color=INK, lw=0.6)
                ax.text(i, yb, max(zs) + 0.002, "Δ" + f"{max(zs) - min(zs):.3f}"[1:], ha="center", va="bottom", fontsize=4.5, color=INK)
        cam = np.array([np.cos(np.radians(AZIM)) * 2.2 / len(WIDTHS), np.sin(np.radians(AZIM)) * 1.2 / len(DEPTHS)])
        for (i, j), z, f in sorted(zip(map(tuple, pos), v, done), key=lambda t: np.dot(t[0], cam)):
            c, a_ = CMAP(norm(z)), 1 if f else 0.4
            ax.bar3d(i - 0.3, j - 0.3, lo, 0.6, 0.6, z - lo, color=1 - TINT * (1 - np.array(c[:3])), edgecolor=(*c[:3], a_),
                     linewidth=0.8, shade=False, linestyle="-" if f else (0, (3, 2)))
        for (i, j), z in zip(pos, v):  # valori sopra le barre, dopo tutto il resto perché restino in primo piano
            ax.text(i, j, z, f"{z:.3f}"[1:], ha="center", va="bottom", fontsize=5.5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        # sigle delle larghezze (tabella del capitolo 4) a mano, orizzontali e subito sotto ogni tick:
        # mplot3d le sposterebbe in diagonale, lontane dal loro tick
        ax.set_xticks(range(len(WIDTHS)), [""] * len(WIDTHS))
        sotto = offset_copy(ax.transData, fig=fig, y=-7, units="points")
        for i, w in enumerate(WIDTHS):
            ax.text(i, -0.5, lo, LETTERA[w], transform=sotto, ha="center", va="center", fontsize=7,
                    color=plt.rcParams["xtick.color"])
        ax.set_yticks(range(len(DEPTHS)), DEPTHS[::-1])
        ax.tick_params(axis="z", labelsize=7); ax.tick_params(axis="y", labelsize=7, pad=0)
        ax.set_xlim(-0.5, len(WIDTHS) - 0.5); ax.set_ylim(-0.5, yb); ax.set_zlim(lo, v.max() + 0.01)
        ax.set_xlabel("larghezza", labelpad=2); ax.set_ylabel("profondità", labelpad=2); ax.set_zlabel("")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis.line.set_color(INK)
            axis._axinfo["grid"].update(color=plt.rcParams["grid.color"], linewidth=plt.rcParams["grid.linewidth"])
        ax.set_zticks(zt, zl)
        ax.view_init(elev=ELEV, azim=AZIM)
        obliqua(ax, ANG_X, ANG_Y)
        ax.set_box_aspect((2.2, 0.95, 1), zoom=0.78)  # stessa scala per larghezza e profondità
        # etichetta z a mano: quella di mplot3d finisce fuori dalla figura e il ritaglio "tight" la perde
        fig.canvas.draw()  # le estensioni delle etichette dei tick 3D non sono affidabili: uso il bordo sinistro del contenuto
        x0 = fig.get_tightbbox(fig.canvas.get_renderer()).x0 / fig.get_figwidth()
        y0 = 0.53  # a metà dell'asse z
        fig.text(x0 - 0.005, y0, name, rotation=90, ha="right", va="center", fontsize=plt.rcParams["axes.labelsize"],
                 color=plt.rcParams["axes.labelcolor"])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_SOFT), cax=fig.add_axes((0.25, 0.14, 0.5, 0.022)),
                          orientation="horizontal")
        cb.set_label(name); cb.outline.set_visible(False)
        print(save(fig, OUT / f"{stem}_barre"))

        # linee 2D: una per profondità, larghezze equispaziate; marker vuoti se in allenamento
        # stile del fronte di Pareto: virgola decimale, solo griglia orizzontale, legenda sotto il grafico
        fig, ax2 = plt.subplots(figsize=(WIDTH, 0.55 * WIDTH))
        # fascia fra minimo e massimo delle profondità: la stessa variazione delle barre sulla parete della figura 3D
        rng = sorted((i, min(z for (ii, _), z in zip(ij, v) if ii == i), max(z for (ii, _), z in zip(ij, v) if ii == i))
                     for i in {i for i, _ in ij})
        ax2.fill_between([r[0] for r in rng], [r[1] for r in rng], [r[2] for r in rng], color=INK, alpha=0.08, lw=0,
                         zorder=1, label="variazione fra profondità")
        for j, d in enumerate(DEPTHS):
            col, ls, mk = LINE[d]
            q = sorted((i, z, f) for (i, jj), z, f in zip(ij, v, done) if jj == j)
            ax2.plot([i for i, _, _ in q], [z for _, z, _ in q], color=col, ls=ls, lw=1.4, label=d, zorder=2,
                     marker=mk, ms=5, mfc=col, mec="white", mew=0.6)
            for i, z, f in q:
                if not f:
                    ax2.plot(i, z, mk, ms=5, mfc="white", mec=col, mew=1, zorder=3)
        # sigle delle larghezze: i valori sono nella tabella del capitolo 4
        ax2.set_xticks(range(len(WIDTHS)), [LETTERA[w] for w in WIDTHS])
        ax2.set_xlim(-0.3, len(WIDTHS) - 0.7)
        ax2.set_yticks(zt, [z.replace(".", ",") for z in zl]); ax2.set_ylim(lo, v.max() + 0.01)
        ax2.grid(axis="x", visible=False)
        ax2.set_xlabel("larghezza"); ax2.set_ylabel(name)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, columnspacing=1.5, handlelength=2.5)
        print(save(fig, OUT / f"{stem}_linee"))
