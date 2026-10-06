"""Gira sul Raspberry Pi (lanciato da bench.sh): tempi per frame di pre-processing, inferenza e
post-processing del yolo26p potato, con OpenVINO diretto e con model.predict() di Ultralytics.

Le immagini vengono lette e decodificate prima delle misure (come i frame di una camera, che
arrivano gia' in memoria); il tempo di imread viene registrato a parte.

    python pi_bench.py <dir immagini> <dir IR> <csv di uscita>
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

IMGSZ, CONF, ROUNDS, WARMUP, THREADS = 640, 0.25, 3, 50, 4
images_dir, ir_dir, out_csv = map(Path, sys.argv[1:4])
MODELS = {md: next((ir_dir / f"max_det_{md}_openvino_model").glob("*.xml")) for md in (300, 50)}
now = time.perf_counter


def temp_c() -> float:
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000


# --- frame in memoria, gia' ridotti a 640x640 (camera a 640) ---
frames, imread_ms = {640: []}, []
for p in sorted(images_dir.glob("*.jpg")):
    t = now()
    img = cv2.imread(str(p))
    imread_ms.append((now() - t) * 1e3)
    frames[640].append(cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR))


def preprocess(img):
    if img.shape[:2] != (IMGSZ, IMGSZ):
        img = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32) * np.float32(1 / 255)


def postprocess(out, shape):
    det = out[0]
    det = det[det[:, 4] > CONF]
    boxes = det[:, :4] * np.array([shape[1] / IMGSZ, shape[0] / IMGSZ] * 2, dtype=np.float32)
    np.clip(boxes, 0, [shape[1], shape[0]] * 2, out=boxes)
    return boxes, det[:, 4], det[:, 5].astype(int)


core = ov.Core()
compiled = {md: core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": THREADS,
                                                     "NUM_STREAMS": 1})
            for md, xml in MODELS.items()}
requests = {md: c.create_infer_request() for md, c in compiled.items()}


def run_openvino(md, size, rows, rnd):
    req = requests[md]
    for i, img in enumerate(frames[size]):
        t0 = now()
        x = preprocess(img)
        t1 = now()
        req.infer({0: x})
        out = req.get_output_tensor(0).data
        t2 = now()
        boxes, _, _ = postprocess(out, img.shape)
        t3 = now()
        rows.append(dict(pipeline="openvino", max_det=md, frame=size, round=rnd, image=i,
                         pre_ms=(t1 - t0) * 1e3, infer_ms=(t2 - t1) * 1e3,
                         post_ms=(t3 - t2) * 1e3, total_ms=(t3 - t0) * 1e3, ndet=len(boxes)))


from ultralytics import YOLO  # noqa: E402

ultra = {md: YOLO(str(MODELS[md].parent), task="detect") for md in MODELS}


def run_ultralytics(md, size, rows, rnd):
    model = ultra[md]
    for i, img in enumerate(frames[size]):
        t0 = now()
        r = model.predict(img, imgsz=IMGSZ, conf=CONF, max_det=md, device="cpu", verbose=False)[0]
        t1 = now()
        s = r.speed
        rows.append(dict(pipeline="ultralytics", max_det=md, frame=size, round=rnd, image=i,
                         pre_ms=s["preprocess"], infer_ms=s["inference"],
                         post_ms=s["postprocess"], total_ms=(t1 - t0) * 1e3, ndet=len(r.boxes)))


CONFIGS = [("openvino", md, size) for size in (640,) for md in (300, 50)] + \
          [("ultralytics", md, 640) for md in (300, 50)]
RUN = {"openvino": run_openvino, "ultralytics": run_ultralytics}

env = {"openvino": ov.__version__, "cv2": cv2.__version__, "images": len(frames[640]),
       "imread_ms_median": float(np.median(imread_ms)), "imread_ms_max": float(np.max(imread_ms)),
       "temp_start_c": temp_c(),
       "freq_khz": int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text())}
for pipe, md, size in CONFIGS:                  # riscaldamento, scartato
    warm = []
    frames_bak = frames[size]
    frames[size] = frames_bak[:WARMUP]
    RUN[pipe](md, size, warm, -1)
    frames[size] = frames_bak

rows = []
for rnd in range(ROUNDS):                       # giri alternati: la deriva termica si spalma
    for pipe, md, size in CONFIGS:
        RUN[pipe](md, size, rows, rnd)
    env[f"temp_round{rnd}_c"] = temp_c()
    print(f"giro {rnd + 1}/{ROUNDS}: {temp_c():.0f} C", flush=True)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
out_csv.with_suffix(".json").write_text(json.dumps(env, indent=1))
print(json.dumps(env))
