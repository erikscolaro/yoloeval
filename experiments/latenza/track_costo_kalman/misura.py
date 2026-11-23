"""Costo di un passo di KalmanTracker.step (src/tracking.py) sul Pi, per N oggetti che si muovono: per fotogramma N
detection forti, N/2 deboli vicine (doppioni o rumore) e 1 falso allarme debole lontano. Un core, numpy a un thread.
Stampa una riga JSON per N. tracking.py va copiato accanto a questo file."""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import KalmanTracker, Params  # noqa: E402

F, WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000, 100
rng = np.random.default_rng(0)


def scena(n, k):
    """Fotogramma k: n oggetti di 40x30 px a velocita' costante, piu' deboli e un falso allarme."""
    i = np.arange(n)
    cx, cy = 100 + 150 * (i % 8) + 3 * k, 100 + 150 * (i // 8) + k
    cx, cy = cx + rng.normal(0, 1, n), cy + rng.normal(0, 1, n)
    forti = np.stack([cx - 20, cy - 15, cx + 20, cy + 15, np.full(n, 0.8), i % 4], 1)
    deboli = forti[: n // 2].copy()
    deboli[:, :4] += rng.normal(0, 2, (len(deboli), 4))
    deboli[:, 4], deboli[:, 5] = 0.2, (deboli[:, 5] + 1) % 4
    falso = np.array([[1200, 650, 1215, 660, 0.15, 3]])
    return np.vstack([forti, deboli, falso])


for n in (0, 1, 2, 4, 8, 16):
    tr, frames = KalmanTracker(4, Params()), [scena(n, k) for k in range(WARM + F)]
    t = []
    for k, d in enumerate(frames):
        t0 = time.perf_counter()
        tr.step(d, 1 / 30)
        if k >= WARM:
            t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    print(json.dumps({"oggetti": n, "track": len(tr.tracks), "median_ms": round(float(np.median(t)), 3),
                      "p90_ms": round(float(np.percentile(t, 90)), 3), "p99_ms": round(float(np.percentile(t, 99)), 3)}))
