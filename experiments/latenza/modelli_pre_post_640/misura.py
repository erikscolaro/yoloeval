"""B1-2 sul Pi: pre-processing e post-processing fuori dal grafo, a 640x640, testa one-to-one (64 box).
pre:  fotogramma uint8 HWC gia' alla risoluzione di deploy -> float32 NCHW / 255 (l'ingresso degli export).
post: uscita (1, 64, 6) [x1, y1, x2, y2, score, classe] -> soglia di confidenza 0.25 e box in coordinate dell'immagine.
Decodifica e TopK sono nel grafo, quindi gia' nelle latenze del benchmark. Stampa una riga JSON."""
import json
import sys
import time

import numpy as np

H = W = int(sys.argv[1]) if len(sys.argv) > 1 else 640
N, WARM, CONF = int(sys.argv[2]) if len(sys.argv) > 2 else 2000, 200, 0.25
rng = np.random.default_rng(0)
frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
out = rng.random((1, 64, 6), dtype=np.float32) * np.array([W, H, W, H, 1, 4], np.float32)
scale = np.array([1, 1, 1, 1], np.float32)   # deploy = ingresso: nessun ridimensionamento


def pre():
    return np.ascontiguousarray(frame.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def post():
    d = out[0]
    d = d[d[:, 4] >= CONF]
    return d[:, :4] * scale, d[:, 4], d[:, 5].astype(np.int32)


def tempi(f):
    for _ in range(WARM):
        f()
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)), "p99_ms": float(np.percentile(t, 99))}


print(json.dumps({"imgsz": [H, W], "numpy": np.__version__, "pre": tempi(pre), "post": tempi(post)}))
