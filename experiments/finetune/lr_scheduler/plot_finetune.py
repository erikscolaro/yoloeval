"""mAP50 e mAP50-95 per epoca dei tre finetune di prova (adam, musgd, sgd), un grafico per metrica.
    .venv/bin/python experiments/ablation/finetune/lr_scheduler/plot_finetune.py
Legge i results.csv in outputs/sweep_yolo26p/<nome>_test/run/ e salva i PNG qui accanto.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUNS_DIR = Path(__file__).resolve().parents[4] / "outputs/sweep_yolo26p"
RUNS = {"Adam": "adam", "MuSGD": "musgd", "SGD": "sgd"}
COLORS = {"Adam": "#2a78d6", "MuSGD": "#eb6834", "SGD": "#1baf7a"}   # ordine categorico fisso
METRICS = {"metrics/mAP50(B)": ("mAP50", "finetune_map50.png"),
           "metrics/mAP50-95(B)": ("mAP50-95", "finetune_map50_95.png")}

data = {}
for label, name in RUNS.items():
    rows = [{k.strip(): v for k, v in r.items()}
            for r in csv.DictReader(open(RUNS_DIR / f"{name}_test/run/results.csv"))]
    data[label] = rows

for col, (title, fname) in METRICS.items():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ends = []
    for label, rows in data.items():
        x = [int(r["epoch"]) for r in rows]
        y = [float(r[col]) for r in rows]
        ax.plot(x, y, color=COLORS[label], linewidth=2, label=label)
        ends.append([y[-1], y[-1], label, x[-1]])
    # etichette finali distanziate di almeno il 4% dell'asse y, dall'alto verso il basso
    lo, hi = ax.get_ylim()
    gap = 0.04 * (hi - lo)
    ends.sort(reverse=True)
    for i in range(1, len(ends)):
        ends[i][1] = min(ends[i][1], ends[i - 1][1] - gap)
    for value, ypos, label, xend in ends:
        ax.annotate(f"{label} {value:.3f}", (xend, value), xytext=(xend + 1.5, ypos),
                    textcoords="data", va="center", fontsize=9, color="#333333")
    ax.axvline(75, color="#999999", linewidth=1, linestyle=":")       # close_mosaic: 25
    ax.text(75.5, ax.get_ylim()[0], " senza mosaic", fontsize=8, color="#666666", va="bottom")
    ax.set_title(f"{title} per epoca, finetune di yolo26p potato (search, epoca 58)", fontsize=11)
    ax.set_xlabel("epoca")
    ax.set_ylabel(title)
    ax.set_xlim(1, 112)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=150)
    plt.close(fig)
    print(HERE / fname)
