"""Pre e post sul Pi a ogni ingresso di deploy (stessi pre e post di ../modelli_pre_post_640/misura.py, che misura solo 640x640).
pre:  fotogramma uint8 HWC gia' all'ingresso del modello -> float32 NCHW / 255.
post: uscita one-to-one (1, 64, 6) -> soglia 0.25 e box. Un core. Stampa una riga JSON per ingresso."""
import json
import time

import numpy as np

N, WARM, CONF = 1000, 100, 0.25
# [h, w] come conf/deploy e tools/figure_sintesi.RPIX, piu' 640x384 (720p senza cambiare le proporzioni)
INGRESSI = {"crop128": (128, 128), "crop256": (256, 256), "crop512": (512, 512), "640": (640, 640),
            "640x384": (384, 640), "480p": (480, 864), "720p": (736, 1280), "1080p": (1088, 1920)}
rng = np.random.default_rng(0)


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


for nome, (H, W) in INGRESSI.items():
    frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    out = rng.random((1, 64, 6), dtype=np.float32) * np.array([W, H, W, H, 1, 4], np.float32)
    pre = lambda: np.ascontiguousarray(frame.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    post = lambda: (lambda d: (d[:, :4], d[:, 4], d[:, 5].astype(np.int32)))(out[0][out[0][:, 4] >= CONF])
    assert pre().shape == (1, 3, H, W)
    print(json.dumps({"deploy": nome, "imgsz": [H, W], "pre": tempi(pre), "post": tempi(post)}), flush=True)
