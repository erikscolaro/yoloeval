"""Costo sul Pi del crop adattativo (src/crop_tracker.py): taglio + resize a 256 + pre-processing, contro il fit del
fotogramma intero a 640 (SCAN). Un core, OpenCV a un thread. Stampa una riga JSON per caso."""
import json
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(1)
N, WARM, CROP = int(sys.argv[1]) if len(sys.argv) > 1 else 500, 50, 256
rng = np.random.default_rng(0)
LATI = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (128, 256, 384, 512, 720)


def pre(img):
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def taglia(frame, side, interp_giu):
    """Come CropTracker.cut: quadrato centrato, grigio fuori dal fotogramma, resize a CROP."""
    H, W = frame.shape[:2]
    s, x0, y0 = side, W // 2 - side // 2, H // 2 - side // 2
    tela = np.full((s, s, 3), 114, np.uint8)
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    tela[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
    return tela if s == CROP else cv2.resize(tela, (CROP, CROP), interpolation=interp_giu if s > CROP else cv2.INTER_LINEAR)


def fit(frame, interp):
    """SCAN: fotogramma intero a lato lungo 640, multiplo di 32 (letterbox senza bordo, come Ultralytics rect)."""
    H, W = frame.shape[:2]
    r = 640 / max(H, W)
    return cv2.resize(frame, (640, -(-round(H * r) // 32) * 32), interpolation=interp)


def tempi(f):
    for _ in range(WARM):
        f()
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    return {"median_ms": round(float(np.median(t)), 3), "p90_ms": round(float(np.percentile(t, 90)), 3)}


for W, H in ((1280, 720), (1920, 1080)):
    frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    for nome, ig in (("area", cv2.INTER_AREA), ("linear", cv2.INTER_LINEAR)):
        print(json.dumps({"video": f"{W}x{H}", "caso": f"scan_fit640_{nome}", **tempi(lambda: pre(fit(frame, ig)))}))
        for side in LATI:
            print(json.dumps({"video": f"{W}x{H}", "caso": f"crop{side}_{nome}", "lato": side,
                              **tempi(lambda: pre(taglia(frame, side, ig)))}))
