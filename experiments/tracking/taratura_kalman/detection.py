"""Passo 1 della taratura: P- (lo SCAN) su tutti i fotogrammi del val di ../video_val_test, una volta sola, come in
../valuta_tracker_video/run.py (fotogramma ridotto con INTER_AREA a 640 di lato lungo, conf >= 0,001). Salva detection e
verita' per segmento (le sequenze si spezzano nei buchi) in runs/detection_<split>.pkl, che le prove della taratura
leggono senza GPU.

    python experiments/tracking/taratura_kalman/detection.py [--split val] [--modello S-]

Con --modello S- fa lo stesso con S- (il modello del TRACK) sul fotogramma intero, in runs/detection_s_<split>.pkl:
serve solo come confronto (detector grande da solo), non alla taratura.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "valuta_tracker_video"))
import run as V  # noqa: E402

CONF = 0.001   # sotto la soglia piu' bassa che la taratura prova


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--modello", choices=["P-", "S-"], default="P-")
    a = ap.parse_args()
    P = YOLO(str(V.SCAN if a.modello == "P-" else V.TRACK))
    fit = V.CropTracker(None, 4).fit
    out = []
    for seq in sorted(p for p in (V.ROOT / "data/tracking" / a.split).iterdir() if (p / "seq.json").exists()):
        info = json.loads((seq / "seq.json").read_text())
        for seg in V.segmenti(seq):
            fr_out = []
            for p in seg:
                fr = cv2.imread(str(p))
                gc, gb = V.verita(p, fr.shape[1], fr.shape[0])
                small, r = fit(fr)
                d = P.predict(small, imgsz=640, conf=CONF, verbose=False)[0].boxes.data.cpu().numpy().copy()
                d[:, :4] /= r
                fr_out.append((gc, gb, d))
            out.append({"seq": seq.name, "fonte": info["fonte"], "fps": info["fps"], "frames": fr_out})
        print(f"{seq.name:32s} {info['fonte']}", flush=True)
    (HERE / "runs").mkdir(exist_ok=True)
    with open(HERE / "runs" / f"detection_{'' if a.modello == 'P-' else 's_'}{a.split}.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
