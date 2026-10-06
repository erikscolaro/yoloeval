"""Gira sul Raspberry Pi (lanciato da bench.sh): tempi per frame di pre-processing, inferenza e
post-processing con OpenVINO (API Python) per ogni IR in models/, frame gia' 640x640 in memoria
(camera a 640). Giri alternati tra gli IR: la deriva termica si spalma.

    python pi_bench.py <dir immagini> <dir IR> <csv di uscita>
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

IMGSZ, CONF, ROUNDS, WARMUP, THREADS = 640, 0.25, 3, 50, 4
images_dir, ir_dir, out_csv = map(Path, sys.argv[1:4])
now = time.perf_counter
MODELS = {int(re.search(r"max_det_(\d+)_", d.name).group(1)): next(d.glob("*.xml"))
          for d in ir_dir.glob("max_det_*_openvino_model")}
MODELS = dict(sorted(MODELS.items()))


def temp_c() -> float:
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000


frames = [cv2.resize(cv2.imread(str(p)), (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
          for p in sorted(images_dir.glob("*.jpg"))]


def preprocess(img):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32) * np.float32(1 / 255)


def postprocess(out, shape):
    det = out[0]
    det = det[det[:, 4] > CONF]
    boxes = det[:, :4] * np.array([shape[1] / IMGSZ, shape[0] / IMGSZ] * 2, dtype=np.float32)
    np.clip(boxes, 0, [shape[1], shape[0]] * 2, out=boxes)
    return boxes


core = ov.Core()
requests = {md: core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": THREADS,
                                                     "NUM_STREAMS": 1}).create_infer_request()
            for md, xml in MODELS.items()}


def run(md, imgs, rows, rnd):
    req = requests[md]
    for i, img in enumerate(imgs):
        t0 = now()
        x = preprocess(img)
        t1 = now()
        req.infer({0: x})
        out = req.get_output_tensor(0).data
        t2 = now()
        boxes = postprocess(out, img.shape)
        t3 = now()
        rows.append(dict(max_det=md, round=rnd, image=i, pre_ms=(t1 - t0) * 1e3,
                         infer_ms=(t2 - t1) * 1e3, post_ms=(t3 - t2) * 1e3,
                         total_ms=(t3 - t0) * 1e3, ndet=len(boxes)))


env = {"openvino": ov.__version__, "images": len(frames), "max_det": list(MODELS),
       "temp_start_c": temp_c()}
for md in MODELS:                               # riscaldamento, scartato
    run(md, frames[:WARMUP], [], -1)
rows = []
for rnd in range(ROUNDS):
    order = list(MODELS) if rnd % 2 == 0 else list(MODELS)[::-1]
    for md in order:
        run(md, frames, rows, rnd)
    env[f"temp_round{rnd}_c"] = temp_c()
    print(f"giro {rnd + 1}/{ROUNDS}: {temp_c():.0f} C", flush=True)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
out_csv.with_suffix(".json").write_text(json.dumps(env, indent=1))
print(json.dumps(env))
