"""Canali tenuti da una search PIT, raggruppati per famiglia di blocco YOLO26 (Conv, C3k2, SPPF).

    .venv/bin/python tools/canali_famiglie.py out.png artifacts/search/<dir> [artifacts/weights/<rete>/best.pt]

Il terzo argomento (facoltativo) sovrappone i canali di un'altra rete non potata (per esempio P-) come tacca nera.

Una riga per convoluzione: barra grigia = canali iniziali, colorata = tenuti (tratteggiata se non potabile).
I numeri dei layer seguono yolo26.yaml; C2PSA (10) e Detect (23) non hanno canali potabili e restano fuori.
"""
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# indice del layer in yolo26.yaml -> (famiglia, parte della rete)
BLOCCHI = {0: "Conv", 1: "Conv", 3: "Conv", 5: "Conv", 7: "Conv", 17: "Conv", 20: "Conv",
           2: "C3k2", 4: "C3k2", 6: "C3k2", 8: "C3k2", 13: "C3k2", 16: "C3k2", 19: "C3k2", 22: "C3k2", 9: "SPPF"}
FAMIGLIE = ["Conv", "C3k2", "SPPF"]
COLORI = {"Conv": "#4C72B0", "C3k2": "#DD8452", "SPPF": "#55A868"}


def canali_rete(pesi: str) -> dict:
    """Canali di uscita per nome yolopit (layers.X...); il cv1 dei C3k2 Ultralytics e' diviso in cv1a e cv1b."""
    m = torch.load(pesi, weights_only=False, map_location="cpu")["model"]
    c = {n.replace("model.", "layers.", 1): mod.out_channels for n, mod in m.named_modules()
         if isinstance(mod, torch.nn.Conv2d)}
    for n in [n for n in c if n.endswith(".cv1.conv") and n.count(".") == 3]:
        if n.replace(".cv1.", ".m.0.cv1.") in c or n.replace(".cv1.", ".m.0.0.cv1.") in c:   # C3k2, non SPPF
            c[n.replace(".cv1.", ".cv1a.")] = c[n.replace(".cv1.", ".cv1b.")] = c[n] // 2
    return c


def main(out: str, search_dir: str, ref: str | None = None) -> None:
    rows = torch.load(Path(search_dir) / "pruned.pt", weights_only=False, map_location="cpu")["pit_meta"]["channels"]
    rif = canali_rete(ref) if ref else {}
    if ref:
        manca = [r[0] for r in rows if r[0] not in rif]
        assert not manca, f"layer senza corrispondenza nella rete di riferimento: {manca}"
    per = {f: [] for f in FAMIGLIE}
    for nome, prima, tenuti, potabile in rows:
        i = int(nome.split(".")[1])
        assert i in BLOCCHI, f"layer {nome} senza famiglia"
        parte = "backbone" if i <= 10 else "neck"
        etichetta = f"{i} {parte} " + nome.split(".", 2)[2].removesuffix(".conv")
        per[BLOCCHI[i]].append((etichetta, prima, tenuti, potabile, rif.get(nome)))

    fig, axs = plt.subplots(len(FAMIGLIE), 1, figsize=(8, 0.22 * len(rows) + 2.5),
                            gridspec_kw={"height_ratios": [max(len(per[f]), 1) for f in FAMIGLIE]})
    for ax, f in zip(axs, FAMIGLIE):
        r = per[f]
        y = range(len(r))
        ax.barh(y, [p for _, p, _, _, _ in r], color="#D0D0D0", label="iniziali")
        ax.barh(y, [t for _, _, t, _, _ in r], color=COLORI[f], label="tenuti",
                hatch=None, edgecolor="none")
        for k, (_, p, t, pot, q) in enumerate(r):
            if q is not None:
                ax.plot([q, q], [k - 0.42, k + 0.42], color="black", lw=1.6, label="_")
            if not pot:
                ax.barh(k, t, color="none", edgecolor="black", hatch="//", linewidth=0)
            ax.text(p + 2, k, f"{t}/{p}", va="center", fontsize=6)
        ax.set_yticks(list(y), [e for e, *_ in r], fontsize=6)
        ax.set_ylim(len(r) - 0.5, -0.5)
        P = sum(p for _, p, _, pot, _ in r if pot)     # percentuale solo sui potabili: quelli fissi la gonfierebbero
        T = sum(t for _, _, t, pot, _ in r if pot)
        ax.set_title(f"{f}: {T}/{P} canali potabili tenuti ({100 * T / P:.0f}%)", fontsize=9, loc="left")
        ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    axs[-1].set_xlabel("canali di uscita")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg = [Patch(color="#D0D0D0", label="PN- iniziali"), Patch(color=COLORI["Conv"], label="PN- tenuti")]
    if ref:
        leg.append(Line2D([], [], color="black", lw=1.6, label=Path(ref).parent.name.split("_air")[0].replace("yolo26", "") + " (P-)"))
    axs[0].legend(handles=leg, fontsize=7, loc="lower right")
    fig.suptitle("Canali tenuti per famiglia di blocco (tratteggio: non potabile)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main(*sys.argv[1:4])
