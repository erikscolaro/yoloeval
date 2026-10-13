"""Stile e salvataggio comuni a tutti i grafici (report, esperimenti, notebook).

    import matplotlib.pyplot as plt
    from tools.plots import STYLE, WIDTH, color, save

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    ax.bar(qs, vals, color=[color(q) for q in qs])
    save(fig, out / "latency")          # latency.png + latency.pdf

Il colore segue l'entita', non la posizione: int8 e' arancio in ogni figura
della tesi, anche quando in un grafico mancano fp16 o fp32.
"""

from __future__ import annotations

from pathlib import Path

STYLE = Path(__file__).with_name("thesis.mplstyle")

#: \textwidth di una tesi A4 (15 cm); mezza larghezza per figure affiancate
WIDTH = 15 / 2.54
HALF = WIDTH / 2 - 0.1

# slot della palette in thesis.mplstyle
_P = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#9e9e9e"  # riferimenti, target, "altro"

COLORS = {
    # split
    "train": _P[0], "val": _P[1], "test": _P[2], "calib": _P[3],
    # precisione
    "fp32": _P[0], "fp16": _P[2], "int8": _P[1],
    # backend
    "onnxruntime": _P[0], "onnxruntime_py": _P[4], "openvino": _P[1],
    "openvino_qdq": _P[7], "tensorrt": _P[2],
    "executorch": _P[3], "axelera": _P[6],
    # board
    "wks4_rtx6000": _P[0], "jetson_orin": _P[2], "rpi5": _P[1], "rpi5_axelera": _P[6],
}


def color(key, i: int = 0) -> str:
    """Colore dell'entita' `key`; per chiavi nuove lo slot `i` della palette."""
    return COLORS.get(str(key), _P[i % len(_P)])


def save(fig, path: Path, pdf: bool = True) -> Path:
    """PNG (anteprima, report) e PDF (LaTeX) con lo stesso nome; chiude la figura."""
    import matplotlib.pyplot as plt

    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf") if pdf else ("png",):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return path.with_suffix(".png")
