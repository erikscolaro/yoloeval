"""Costo per fotogramma TRACK del crop tracker fuori dal modello, sul Pi, con N track (= N crop 256x256 nel batch).
Fotogramma 1920x1080 uint8. Il modello e' finto: per ogni crop restituisce la box vera dell'oggetto, cosi' i track
restano confermati e si misura il percorso TRACK vero (src/crop_tracker.py, src/tracking.py):
  cut:    ritaglio + resize dei N crop
  step:   tutto CropTracker.step tranne il modello (cut, Kalman, associazione, riproiezione)
  pre:    N crop uint8 HWC -> float32 NCHW / 255 (ingresso dell'export, batch N)
  post:   N uscite (64, 6) -> soglia di confidenza, come in ../modelli_pre_post_640/misura.py
Stampa una riga JSON per N."""
import json
import sys
import time

import numpy as np

from yoloeval_src.crop_tracker import CropTracker
from yoloeval_src.tracking import Params

H, W, CROP, CONF = 1080, 1920, 256, 0.25
ITER, WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000, 200
rng = np.random.default_rng(0)
frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def stats(t):
    t = np.array(t) * 1e3
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
            "p99_ms": float(np.percentile(t, 99))}


def misura(n):
    # n oggetti 60x40 px su una griglia, oscillano (+-60 px, periodo 3 s) per restare nel fotogramma
    c0 = np.array([[300 + 400 * (i % 4), 250 + 500 * (i // 4)] for i in range(n)], float)
    dt = 1 / 30
    k = {"f": 0}
    orig = []                           # (origine, scala) dei crop, nell'ordine in cui il tracker li taglia
    t_cut = []

    def boxes():
        a = 2 * np.pi * k["f"] * dt / 3
        c = c0 + 60 * np.array([np.sin(a), 0.5 * np.cos(a)])
        return np.hstack([c - [30, 20], c + [30, 20], np.full((n, 1), 0.9), np.zeros((n, 1))])

    def detect(images, imgsz, conf):
        b = boxes()
        if len(images) == 1 and imgsz != CROP:     # SCAN: fotogramma intero
            return [b]
        out = []
        for (x0, y0), sc in orig:                  # TRACK: la box vera che cade nel crop, in coordinate del crop
            c = (b[:, :2] + b[:, 2:4]) / 2
            j = int(np.argmin(np.hypot(*(c - [x0 + CROP / sc / 2, y0 + CROP / sc / 2]).T)))
            d = b[j].copy()
            d[[0, 2]] = (d[[0, 2]] - x0) * sc
            d[[1, 3]] = (d[[1, 3]] - y0) * sc
            out.append(d[None])
        orig.clear()
        return out

    tr = CropTracker(detect, nc=1, p=Params(fps=30), crop=CROP, rescan=0, max_crops=8)
    cut = tr.cut

    def cut_timed(*a):
        t0 = time.perf_counter()
        img, o, sc = cut(*a)
        t_cut[-1] += time.perf_counter() - t0
        orig.append((o, sc))
        return img, o, sc

    tr.cut = cut_timed
    crops = np.zeros((n, CROP, CROP, 3), np.uint8)
    out = rng.random((n, 64, 6), dtype=np.float32) * np.array([CROP, CROP, CROP, CROP, 1, 4], np.float32)
    t_step, t_pre, t_post, modes = [], [], [], []
    for i in range(WARM + ITER):
        t_cut.append(0.0)
        t0 = time.perf_counter()
        _, info = tr.step(frame, dt)
        t1 = time.perf_counter()
        x = np.ascontiguousarray(crops.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        t2 = time.perf_counter()
        for o in out:
            d = o[o[:, 4] >= CONF]
            d[:, :4], d[:, 4], d[:, 5].astype(np.int32)
        t3 = time.perf_counter()
        k["f"] += 1
        if i >= WARM:
            modes.append(info["mode"])
            t_step.append(t1 - t0), t_pre.append(t2 - t1), t_post.append(t3 - t2)
        elif i == WARM - 1:
            t_cut.clear()
    track = [m == "TRACK" for m in modes]
    assert sum(track) > 0.95 * len(track), f"n={n}: solo {sum(track)} fotogrammi TRACK su {len(track)}"
    sel = lambda t: [x for x, m in zip(t, track) if m]
    tot = [a + b + c for a, b, c in zip(sel(t_step), sel(t_pre), sel(t_post))]
    return {"n": n, "frame": [H, W], "crop": CROP, "track_frac": sum(track) / len(track),
            "cut": stats(sel(t_cut[-len(modes):])), "step": stats(sel(t_step)), "pre": stats(sel(t_pre)),
            "post": stats(sel(t_post)), "totale": stats(tot)}


for n in (1, 2, 4, 8):
    print(json.dumps(misura(n)), flush=True)
