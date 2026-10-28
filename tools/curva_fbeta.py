"""Curva F-beta in funzione della soglia di confidenza, per classe, come la BoxF1_curve di Ultralytics ma con beta a
scelta: beta 2 pesa il recall il doppio della precisione, 0.5 il contrario, 1 e' la F1.

    python tools/curva_fbeta.py PESI OUT_DIR                        # F2 sul val
    python tools/curva_fbeta.py PESI OUT_DIR --beta 1 --split test  # F1 sul test

Valida con la testa one-to-one (nms=False) e conf 0.001, come il val del training. Non lascia niente su disco oltre
alla figura: la cartella di Ultralytics e' temporanea e le cache delle label create qui vengono tolte.
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.plots import STYLE, WIDTH, color, save  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data/air-detection/all/dataset.yaml"


def _comma(x: float, fmt: str = "{:.2f}") -> str:
    return fmt.format(x).replace(".", ",")


def curve(pesi, split: str, beta: float, data=DATA):
    """(soglie, {classe: F-beta per soglia}) dalle curve P e R della validazione di Ultralytics."""
    from ultralytics import YOLO

    labels = Path(data).parent / "labels"
    caches = set(labels.glob("*.cache"))
    with tempfile.TemporaryDirectory() as tmp:
        r = YOLO(str(pesi)).val(data=str(data), split=split, nms=False, plots=False, verbose=False,
                                project=tmp, name="v", exist_ok=True)
    for c in set(labels.glob("*.cache")) - caches:
        c.unlink()
    b, b2 = r.box, beta ** 2
    fb = (1 + b2) * b.p_curve * b.r_curve / (b2 * b.p_curve + b.r_curve + 1e-16)
    return np.linspace(0, 1, fb.shape[1]), {r.names[c]: fb[i] for i, c in enumerate(b.ap_class_index)}


def figura(out, soglie, fb: dict, beta: float, split: str, nome: str = "fbeta"):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    lab = f"F{_comma(beta, '{:g}')}"
    for i, (cls, y) in enumerate(fb.items()):
        k = y.argmax()
        ax.plot(soglie, y, color=color(cls, i), lw=1.2,
                label=f"{cls}: {_comma(y[k])} a {_comma(soglie[k])}")
        ax.plot(soglie[k], y[k], "o", color=color(cls, i), ms=4)
    media = np.mean(list(fb.values()), axis=0)
    k = media.argmax()
    ax.plot(soglie, media, color="black", lw=2.2, label=f"media: {_comma(media[k])} a {_comma(soglie[k])}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Soglia di confidenza", ylabel=lab,
           title=f"{lab} per classe, {split}")
    ax.legend(loc="lower left", fontsize=7)
    return save(fig, Path(out) / f"{nome}_{lab.lower().replace(',', '')}_{split}", pdf=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pesi", help="best.pt da validare")
    ap.add_argument("out", nargs="?", default=".")
    ap.add_argument("--beta", type=float, default=2.0, help="2: recall pesato il doppio; 1: F1; 0.5: precisione")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    soglie, fb = curve(args.pesi, args.split, args.beta)
    for cls, y in fb.items():
        print(f"{cls:11s} {y.max():.3f} a soglia {soglie[y.argmax()]:.3f}")
    print(figura(args.out, soglie, fb, args.beta, args.split))
