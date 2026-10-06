"""Misura omogenea di piu' modelli adattati su Svanstrom: val e test, mAP a `conf` 0.001 e, per classe, F1 alla `conf`
che lo massimizza sul val (stessa soglia sul test).

    .venv/bin/python experiments/adapt/model_size/eval.py p=artifacts/adapt/<slug>/best.pt n=... [--imgsz 640]
"""
import sys

import numpy as np
from ultralytics import YOLO

args = [a for a in sys.argv[1:] if "=" in a]
for label, w in (a.split("=", 1) for a in args):
    m = YOLO(w)
    kw = dict(data="data/svanstrom/all/dataset.yaml", imgsz=640, conf=0.001, plots=False, workers=2, verbose=False,
              project="/tmp/val", name="eval", exist_ok=True)
    v, t = m.val(split="val", **kw), m.val(split="test", **kw)
    print(f"== {label}: val tutte mAP50 {v.box.map50:.3f} mAP50-95 {v.box.map:.3f} | test tutte {t.box.map50:.3f} / {t.box.map:.3f}")
    for pos, c in enumerate(v.box.ap_class_index):
        k = int(np.argmax(v.box.f1_curve[c]))
        f1 = lambda r: 2 * r.box.p_curve[c][k] * r.box.r_curve[c][k] / (r.box.p_curve[c][k] + r.box.r_curve[c][k] + 1e-16)
        print(f"| {v.names[c]} | conf* {v.box.px[k]:.3f} | val F1 {f1(v):.3f} mAP50 {v.box.class_result(pos)[2]:.3f} "
              f"mAP50-95 {v.box.class_result(pos)[3]:.3f} | test F1 {f1(t):.3f} mAP50 {t.box.class_result(pos)[2]:.3f} "
              f"mAP50-95 {t.box.class_result(pos)[3]:.3f}")
