"""Costo fuori dal modello di un fotogramma SCAN sul Pi: 720p -> 640x360 (proporzioni invariate) + bordo grigio fino a
640x384 + pre (uint8 HWC -> float32 NCHW / 255) + post (64x6 -> soglia). Un core, OpenCV a un thread.
Stampa una riga JSON per interpolazione."""
import json
import time

import cv2
import numpy as np

cv2.setNumThreads(1)
N, WARM = 1000, 100
frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
out = np.random.default_rng(1).random((1, 64, 6), dtype=np.float32)


def scan(interp):
    img = cv2.resize(frame, (640, 360), interpolation=interp)
    img = cv2.copyMakeBorder(img, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 360 -> 384, centrato
    assert img.shape == (384, 640, 3)
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x, out[0][out[0][:, 4] > 0.25]


for nome, ig in (("area", cv2.INTER_AREA), ("linear", cv2.INTER_LINEAR)):
    for _ in range(WARM):
        scan(ig)
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        scan(ig)
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    print(json.dumps({"interp": nome, "median_ms": round(float(np.median(t)), 3),
                      "p99_ms": round(float(np.percentile(t, 99)), 3),
                      "budget_ms": round(1000 / 30 - float(np.median(t)), 2)}))
