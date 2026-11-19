"""Grafico a barre dei canali tenuti da una search PIT, per layer potabile.

    .venv/bin/python tools/search_channels.py out.png n1=artifacts/search/<dir> n8=artifacts/search/<dir> ...
    (al posto di <dir> anche un .json con {"channels": righe}; i layer si accoppiano per nome alla prima serie)

Legge pit_meta['channels'] di <dir>/pruned.pt: (layer, canali prima, canali tenuti, potabile). Una barra per layer,
una serie per ogni search; in ascissa la frazione di canali tenuti, a destra "tenuti/prima" della prima serie.
"""
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def channels(search_dir: str) -> list:
    if search_dir.endswith(".json"):      # righe gia' estratte (ricerca in corso): {"channels": [...]}
        import json
        return json.loads(Path(search_dir).read_text())["channels"]
    d = torch.load(Path(search_dir) / "pruned.pt", weights_only=False, map_location="cpu")
    return d["pit_meta"]["channels"]


def main(out: str, runs: dict[str, str]) -> None:
    rows = {label: channels(p) for label, p in runs.items()}
    first = next(iter(rows.values()))
    idx = [i for i, r in enumerate(first) if r[3]]            # layer potabili
    names = [first[i][0] for i in idx]
    k = len(rows)
    h = 0.8 / k
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(idx) * k / 2 + 2))
    for j, (label, rs) in enumerate(rows.items()):
        by = {r[0]: r for r in rs}        # per nome: una serie puo' non avere (o non potare) un layer
        frac = [by[n][2] / by[n][1] if n in by and by[n][3] else 0 for n in names]
        ax.barh([y + (j - (k - 1) / 2) * h for y in range(len(idx))], frac, height=h, label=label)
    ax.set_yticks(range(len(idx)), [f"{n} ({first[i][1]})" for n, i in zip(names, idx)], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("frazione di canali tenuti")
    ax.set_title("Canali tenuti per layer potabile (tra parentesi i canali iniziali)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:]))
