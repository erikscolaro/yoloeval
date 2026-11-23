"""Perche' INTER_AREA e' lento con alcuni fattori di riduzione? Solo cv2.resize (niente taglio ne' tela), sorgente
quadrata di lato S ridotta a 256. Varianti: area1 = AREA a 1 thread; area4 = AREA a 4 thread (il modello non gira
mentre si ritaglia, quindi i core sono liberi); lin1 = LINEAR; due1 = AREA 2x esatto poi LINEAR a 256 (solo S pari
e >= 512). Una riga JSON per S e variante, mediana in ms."""
import json
import sys
import time

import cv2
import numpy as np

N, WARM, DST = int(sys.argv[1]) if len(sys.argv) > 1 else 300, 30, 256
LATI = (300, 333, 384, 448, 512, 640, 720, 768, 1000, 1024, 1280, 1536, 2048)
print(json.dumps({"cv2": cv2.__version__, "ottimizzato": cv2.useOptimized(), "cpu": cv2.getNumberOfCPUs(),
                  "simd": [n for n in ("NEON", "SSE4_2", "AVX2") if hasattr(cv2, "CPU_" + n) and cv2.checkHardwareSupport(getattr(cv2, "CPU_" + n))]}))
rng = np.random.default_rng(0)


def tempo(f):
    for _ in range(WARM):
        f()
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return round(float(np.median(t)) * 1e3, 3)


for S in LATI:
    img = rng.integers(0, 256, (S, S, 3), dtype=np.uint8)
    v = {"area1": (1, lambda: cv2.resize(img, (DST, DST), interpolation=cv2.INTER_AREA)),
         "area4": (4, lambda: cv2.resize(img, (DST, DST), interpolation=cv2.INTER_AREA)),
         "lin1": (1, lambda: cv2.resize(img, (DST, DST), interpolation=cv2.INTER_LINEAR))}
    if S % 2 == 0 and S >= 512:
        v["due1"] = (1, lambda: cv2.resize(cv2.resize(img, (S // 2, S // 2), interpolation=cv2.INTER_AREA), (DST, DST),
                                           interpolation=cv2.INTER_LINEAR))
    for nome, (th, f) in v.items():
        cv2.setNumThreads(th)
        print(json.dumps({"S": S, "fattore": round(S / DST, 2), "variante": nome, "ms": tempo(f)}))
