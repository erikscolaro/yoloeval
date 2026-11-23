"""Costo fuori dal modello dello SCAN per ogni cella camera x ingresso del modello: resize (se serve, senza cambiare le
proporzioni), bordo grigio 114 fino all'ingresso (multipli di 32), pre (uint8 HWC -> float32 NCHW / 255). Un core, OpenCV a un
thread. Una riga JSON per cella e interpolazione; il post (64 box) costa 0,009 ms (../modelli_budget_per_ingresso)."""
import json
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(1)
N, WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 300, 30
rng = np.random.default_rng(0)
CAM = {"640": (640, 360), "HD": (1280, 720), "FHD": (1920, 1080)}                 # larghezza, altezza
INP = {"640x384": (640, 384), "HD": (1280, 736), "FHD": (1920, 1088)}              # ingresso del modello


def cella(frame, interp, w, h):
    H, W = frame.shape[:2]
    r = min(w / W, h / H)
    nw, nh = round(W * r), round(H * r)
    img = frame if (nw, nh) == (W, H) else cv2.resize(frame, (nw, nh), interpolation=interp)
    tela = np.full((h, w, 3), 114, np.uint8)
    y0 = (h - nh) // 2
    tela[y0:y0 + nh, :nw] = img
    return np.ascontiguousarray(tela.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def tempi(f):
    for _ in range(WARM):
        f()
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    return {"median_ms": round(float(np.median(t)), 3), "p99_ms": round(float(np.percentile(t, 99)), 3)}


for cn, (cw, ch) in CAM.items():
    frame = rng.integers(0, 256, (ch, cw, 3), dtype=np.uint8)
    for inn, (w, h) in INP.items():
        if w > cw:                    # niente ingrandimenti
            continue
        for nome, ig in (("area", cv2.INTER_AREA), ("linear", cv2.INTER_LINEAR)):
            print(json.dumps({"camera": cn, "ingresso": inn, "interp": nome, **tempi(lambda: cella(frame, ig, w, h))}))
