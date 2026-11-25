"""Pezzi comuni della taratura: lettura delle detection salvate (detection.py), abbinamento detection-verita' e F1."""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.tracking import iou, xyxy2cxcywh  # noqa: E402

NAMES = ["drone", "airplane", "helicopter", "bird"]
FONTI = ("svanstrom", "anti_uav")
DRONE, AEREI = NAMES.index("drone"), (NAMES.index("airplane"), NAMES.index("helicopter"))


def carica(split):
    with open(HERE / "runs" / f"detection_{split}.pkl", "rb") as f:
        return pickle.load(f)


def abbina(det, gc, gb, classe=True, soglia=0.5):
    """Coppie (i_det, j_gt) con IoU >= soglia (e stessa classe), abbinamento goloso per confidenza."""
    if not len(det) or not len(gb):
        return []
    o = np.argsort(-det[:, 4])
    M, used, out = iou(det[o, :4], gb), np.zeros(len(gb), bool), []
    for k, i in enumerate(o):
        ok = ~used & (M[k] >= soglia)
        if classe:
            ok &= gc == int(det[i, 5])
        j = np.where(ok)[0]
        if len(j):
            j = j[M[k, j].argmax()]
            used[j] = True
            out.append((i, j))
    return out


def conta(det, gc, gb):
    """tp, fn, fp di un fotogramma: IoU >= 0,5 e stessa classe."""
    tp = len(abbina(det, gc, gb))
    return tp, len(gb) - tp, len(det) - tp


def f1(tp, fn, fp):
    return 2 * tp / max(2 * tp + fn + fp, 1)


def residui(det, gc, gb):
    """Per ogni detection abbinata: (classe, scarto cx, cy, w, h diviso l'altezza vera)."""
    out = []
    for i, j in abbina(det, gc, gb):
        z, g = xyxy2cxcywh(det[i, :4]), xyxy2cxcywh(gb[j])
        out.append((gc[j], (z - g) / max(g[3], 1.0)))
    return out


def conta_drone(out, gc, gb):
    """Droni: tp, fn, fp (track drone su drone vero, IoU >= 0,5); aerei ed elicotteri veri e quanti hanno un track
    drone sopra (IoU >= 0,5)."""
    o = out[out[:, 5] == DRONE] if len(out) else out
    dr = gc == DRONE
    tp = len(abbina(o, gc[dr], gb[dr], classe=False))
    ae = np.isin(gc, AEREI)
    colpiti = len(abbina(o, gc[ae], gb[ae], classe=False))
    return np.array([tp, dr.sum() - tp, len(o) - tp, ae.sum(), colpiti])


def sintesi(acc):
    tp, fn, fp, ae, col = acc
    return {"f2": 5 * tp / max(5 * tp + 4 * fn + fp, 1), "f1": 2 * tp / max(2 * tp + fn + fp, 1), "recall": tp / max(tp + fn, 1),
            "precisione": tp / max(tp + fp, 1), "aerei_come_drone": col / max(ae, 1), "aerei": int(ae)}
