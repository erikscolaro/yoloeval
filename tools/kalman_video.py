"""Classe per fotogramma sui video di predict/, detection grezze contro tracker di Kalman (src/tracking.py).

La classe giusta viene dal nome del video (drone_*.mp4, airplane_*, helicopter_*, bird_*): misura solo la
classificazione e la copertura, non le box, e non conta i falsi allarmi. Per ogni video: fotogrammi con almeno una
detection (grezze: conf >= --hi), classe giusta fra questi (quella della detection o del track piu' sicuro), e
fotogrammi giusti su tutti, che e' il numero da confrontare.

    python -m tools.kalman_video MODELLO.pt [--video predict] [--imgsz 640] [--hi 0.4 --lo 0.1 --vote 0.9 ...]
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.tracking import KalmanTracker, Params
from tools.predict_video import KF


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model"), p.add_argument("--video", default="predict"), p.add_argument("--imgsz", type=int, default=640)
    d = Params()
    for k in KF:
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(getattr(d, k)), default=getattr(d, k))
    a = p.parse_args()
    par = Params(**{k: getattr(a, k) for k in KF})
    m = YOLO(a.model)
    names = {v: k for k, v in m.names.items()}
    tot = {"grezze": [0, 0, 0], "kalman": [0, 0, 0]}
    print(f"{'video':28s} {'fotogr':>6s} | {'grezze: det':>11s} {'cls ok':>6s} {'giusti':>6s} | "
          f"{'kalman: det':>11s} {'cls ok':>6s} {'giusti':>6s}")
    for v in sorted(Path(a.video).glob("*.mp4")):
        if v.stem.split("_")[0] not in names:
            continue
        giusta = names[v.stem.split("_")[0]]
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        tr, n, rd, rc, kd, kc = KalmanTracker(len(m.names), par), 0, 0, 0, 0, 0
        while (f := cap.read()[1]) is not None:
            det = m.predict(f, imgsz=a.imgsz, conf=par.lo, verbose=False)[0].boxes.data.cpu().numpy()
            t, n = tr.step(det, 1 / fps), n + 1
            hi = det[det[:, 4] >= par.hi]
            if len(hi):
                rd, rc = rd + 1, rc + (int(hi[hi[:, 4].argmax(), 5]) == giusta)
            if t:
                kd, kc = kd + 1, kc + (max(t, key=lambda x: x.conf).cls == giusta)
        if not n:
            print(f"{v.stem:28s} non letto (codec?)")
            continue
        for k, x in (("grezze", (n, rd, rc)), ("kalman", (n, kd, kc))):
            tot[k] = [s + y for s, y in zip(tot[k], x)]
        print(f"{v.stem:28s} {n:6d} | {rd / n:11.0%} {rc / max(rd, 1):6.0%} {rc / n:6.0%} | "
              f"{kd / n:11.0%} {kc / max(kd, 1):6.0%} {kc / n:6.0%}")
    for k, (n, det, c) in tot.items():
        print(f"TOTALE {k}: fotogrammi con detection {det / max(n, 1):.1%}, classe giusta fra questi "
              f"{c / max(det, 1):.1%}, fotogrammi giusti su tutti {c / max(n, 1):.1%}")


if __name__ == "__main__":
    main()
