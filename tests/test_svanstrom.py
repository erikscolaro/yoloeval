"""tools/svanstrom.py: split per video e conversione dei box.

Lo split deve restare per video (mai frame dello stesso video in split
diversi: e' il leakage di AOD4) e avvicinarsi al 70/20/10 in ogni strato.
"""

from __future__ import annotations

import numpy as np

from tools.svanstrom import _yolo, assign_splits


def test_split_per_video_bilancia_classi_e_fasce():
    rng = np.random.default_rng(0)
    videos = [{"video": f"v{i}", "class": c,
               "sizes": list(rng.choice(["small", "medium", "large"], rng.integers(20, 120),
                                        p=[0.5, 0.4, 0.1]))}
              for c in ("drone", "bird") for i in range(40)]
    assign_splits(videos, seed=1)
    for c in ("drone", "bird"):
        g = [v for v in videos if v["class"] == c]
        for z in ("small", "medium", "large"):
            tot = sum(v["sizes"].count(z) for v in g)
            for s, target in (("train", 0.7), ("val", 0.2), ("test", 0.1)):
                got = sum(v["sizes"].count(z) for v in g if v["split"] == s)
                assert abs(got / tot - target) < 0.05, (c, z, s, got / tot)


def test_yolo_taglia_ai_bordi_e_scarta_box_fuori():
    boxes = np.array([[2, -10, 10, 30, 20],     # esce a sinistra: tagliato
                      [1, 700, 10, 30, 20]])    # tutto fuori: scartato
    lines = _yolo(boxes, 640, 512).splitlines()
    assert len(lines) == 1
    c, x, y, w, h = lines[0].split()
    assert c == "2" and float(w) * 640 == 20 and float(x) * 640 == 10
