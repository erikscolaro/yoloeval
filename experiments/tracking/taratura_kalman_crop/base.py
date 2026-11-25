"""Pezzi comuni della taratura con crop: dati di ../taratura_kalman (detection di P- salvate, R e Q), percorsi delle
immagini, fotogramma decodificato solo quando serve, metriche sui droni e vincolo su aerei ed elicotteri."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
KAL = HERE.parent / "taratura_kalman"
sys.path.insert(0, str(KAL))
from comune import AEREI, DRONE, FONTI, NAMES, abbina, carica, conta_drone as conta, sintesi  # noqa: E402,F401

_sp = importlib.util.spec_from_file_location("vtv", HERE.parent / "valuta_tracker_video" / "run.py")
V = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(V)
ROOT, SCAN, TRACK = V.ROOT, V.SCAN, V.TRACK


def dati(split):
    """Segmenti di detection_<split>.pkl con i percorsi delle immagini (stesso ordine di V.segmenti)."""
    d = carica(split)
    paths = [s for seq in sorted(p for p in (ROOT / "data/tracking" / split).iterdir() if (p / "seq.json").exists())
             for s in V.segmenti(seq)]
    assert len(paths) == len(d) and all(len(s["frames"]) == len(p) for s, p in zip(d, paths)), "rifare detection.py"
    for s, p in zip(d, paths):
        s["paths"] = p
    return d


def rq(split):
    return json.loads((KAL / "results" / f"rq_{split}.json").read_text())


class Pigro:
    """Un fotogramma che si decodifica solo al primo ritaglio: in SCAN le detection di P- sono gia' salvate."""
    def __init__(self, path, shape):
        self.path, self.shape, self.img = path, shape, None

    def __getitem__(self, k):
        if self.img is None:
            self.img = cv2.imread(str(self.path))
        return self.img[k]
