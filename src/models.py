"""Architetture YOLO26 della tesi: larghezza e profondita' (NEW_MASTERPLAN §4).

Si parte dallo YAML ufficiale di YOLO26 e si cambiano solo le costanti di scala e, per D101, il
`c3k` dei C3k2. La scala si chiama `w`: con `m`, `l`, `x` Ultralytics forzerebbe `c3k=True` in
tutti i C3k2 (`parse_model`), e M cambierebbe profondita' oltre che larghezza.
"""

from __future__ import annotations

import copy

# profondita' -> (moltiplicatore di profondita', numero di conv della rete)
DEPTHS = {"D101": (0.5, 101), "D126": (0.5, 126), "D170": (1.0, 170)}
# scala discreta delle larghezze ammesse (§4): le altre non costruiscono o duplicano un'architettura
WIDTHS = (0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0)


def arch(width: float, depth: str, nc: int) -> dict:
    """Dizionario Ultralytics del modello (quello che si scrive in uno YAML)."""
    from ultralytics.nn.tasks import yaml_model_load

    if width not in WIDTHS or depth not in DEPTHS:
        raise ValueError(f"larghezza {width} o profondita' {depth} fuori dalla scala del piano "
                         f"({WIDTHS}, {tuple(DEPTHS)})")
    d = copy.deepcopy(yaml_model_load("yolo26n.yaml"))
    for k in ("yaml_file", "scale"):
        d.pop(k, None)
    # sopra 0.5 il tetto di canali della m ufficiale, altrimenti 0.75 avrebbe piu' parametri di M
    d.update(nc=int(nc), scales={"w": [DEPTHS[depth][0], width, 512 if width > 0.5 else 1024]})
    if depth == "D101":
        for layer in d["backbone"] + d["head"]:
            if layer[2] == "C3k2":
                layer[3][1] = False
    return d


def deploy_hw(cfg) -> list[int]:
    """[h, w] dell'ingresso di export e misure (gruppo `deploy`)."""
    return [int(x) for x in cfg.deploy.imgsz]


def check_convs(model, depth: str) -> None:
    """Il numero di conv definisce la profondita': un'altra versione di Ultralytics potrebbe
    cambiarlo senza errori."""
    import torch.nn as nn

    n = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    if n != DEPTHS[depth][1]:
        raise ValueError(f"{depth}: {n} conv invece di {DEPTHS[depth][1]}")
