"""Fase 1: R e scala di confidenza di S- (il modello dei crop), su ritagli intorno agli oggetti veri del val.
Ogni 5 fotogrammi, per ogni oggetto vero un ritaglio come quelli del tracker: lato = max(lato dell'oggetto / 0,6, 256)
x 1,2, al massimo 512, centro spostato a caso (deviazione 5% del lato, l'errore della predizione), portato a 256 con
CropTracker.cut. S- a conf >= 0,01; detection riportate nel fotogramma.
- R: come ../taratura_kalman/rq.py (scarto / altezza vera, deviazione robusta, conf >= 0,3, IoU >= 0,5, stessa classe).
- Confidenza: soglia con F1 massimo sui droni nei ritagli (oggetti interi nel ritaglio), da confrontare con P-.
Uscita: results/rs_<split>.json.

    python experiments/tracking/taratura_kalman_crop/rs.py [--split val]
"""
import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO

from base import DRONE, HERE, NAMES, TRACK, V, abbina, dati
from comune import residui

PASSO, BATCH = 5, 32


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    S, ct, rng = YOLO(str(TRACK)), V.CropTracker(None, 4), np.random.default_rng(0)
    casi = []   # (fonte, gc, gb nel ritaglio, det nel fotogramma)
    for s in dati(a.split):
        for i in range(0, len(s["frames"]), PASSO):
            gc, gb, _ = s["frames"][i]
            if not len(gb):
                continue
            fr = cv2.imread(str(s["paths"][i]))
            for b in gb:
                w, h = b[2] - b[0], b[3] - b[1]
                side = min(max(max(w, h) / 0.6, 256) * 1.2, 512)
                cx, cy = (b[0] + b[2]) / 2 + rng.normal(0, .05 * side), (b[1] + b[3]) / 2 + rng.normal(0, .05 * side)
                img, (x0, y0), sc = ct.cut(fr, cx, cy, side)
                dentro = (gb[:, 0] >= x0) & (gb[:, 1] >= y0) & (gb[:, 2] <= x0 + side) & (gb[:, 3] <= y0 + side)
                casi.append([s["fonte"], gc[dentro], gb[dentro], img, (x0, y0, sc)])
    for k in range(0, len(casi), BATCH):
        res = S.predict([c[3] for c in casi[k:k + BATCH]], imgsz=256, conf=0.01, verbose=False)
        for c, r in zip(casi[k:k + BATCH], res):
            d = r.boxes.data.cpu().numpy().copy()
            x0, y0, sc = c[4]
            d[:, [0, 2]] = d[:, [0, 2]] / sc + x0
            d[:, [1, 3]] = d[:, [1, 3]] / sc + y0
            c[3] = d
    out = {"ritagli": len(casi)}
    for fonte in ("svanstrom", "anti_uav", "tutte"):
        cc = [c for c in casi if fonte == "tutte" or c[0] == fonte]
        res = [r for c in cc for r in residui(c[3][c[3][:, 4] >= 0.3], c[1], c[2])]
        R = {}
        for nome, sel in [("tutte", res)] + [(n, [r for r in res if r[0] == i]) for i, n in enumerate(NAMES)]:
            if len(sel) >= 50:
                e = np.array([r[1] for r in sel])
                mad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))  # noqa: E731
                R[nome] = {"r_pos": mad(e[:, :2].ravel()), "r_size": mad(e[:, 2:].ravel()), "n": len(sel)}
        f1 = {}
        for conf in np.round(np.r_[0.01, 0.02, 0.03, 0.04, np.arange(0.05, 0.95, 0.05)], 2):
            tp = fn = fp = 0
            for _, gc, gb, d, _ in cc:
                o, dr = d[(d[:, 4] >= conf) & (d[:, 5] == DRONE)], gc == DRONE
                m = len(abbina(o, gc[dr], gb[dr], classe=False))
                tp, fn, fp = tp + m, fn + dr.sum() - m, fp + len(o) - m
            f1[float(conf)] = 2 * tp / max(2 * tp + fn + fp, 1)
        best = max(f1, key=f1.get)
        out[fonte] = {"R": R, "r_pos": R["tutte"]["r_pos"], "r_size": R["tutte"]["r_size"], "conf_f1_drone": best,
                      "f1_drone": f1[best], "curva_f1": f1}
        print(f"{fonte:10s} r_pos {R['tutte']['r_pos']:.4f} r_size {R['tutte']['r_size']:.4f} conf migliore {best} "
              f"(F1 droni {f1[best]:.3f})", flush=True)
    (HERE / "results" / f"rs_{a.split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
