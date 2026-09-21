#!/usr/bin/env python3
"""Calcolo di mAP con Ultralytics, eseguito dove vive l'artefatto.

Gira sulla workstation per gli artefatti portabili (.pt, .onnx) e sulla board
per quelli hardware-specifici (engine TensorRT, modelli Axelera), che sulla
workstation non sono nemmeno caricabili.

Stampa una sola riga JSON su stdout: tutto il resto e' rumore di Ultralytics e
viene ignorato dal chiamante.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(args.model, task="detect")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,          # basso: la mAP integra su tutte le soglie
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        split=args.split,
        batch=1,
        plots=False,
        verbose=False,
    )
    payload = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "split": args.split,
        "wall_s": round(time.time() - t0, 3),
    }
    print("YOLOBENCH_JSON " + json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
