"""Caricamento e cambio di modello sul Pi 5 (OpenVINO fp16, 4 thread, un solo stream): SCAN = PN- a 640x384, TRACK = S- a
256x256 batch 1. Tre modi, una riga JSON per misura:
  carico  <xml> <HxW> [cache_dir]   read_model, compile_model, prima inferenza e inferenza a regime (processo nuovo)
  cambio  <xml scan> <HxW> <xml track> <HxW>   i due modelli compilati nello stesso processo: tempi di ogni chiamata
          da soli, alternati a ogni fotogramma e nel ciclo reale (1 SCAN ogni 30 fotogrammi, con e senza lavoro fra
          le chiamate che sporca la cache)."""
import json
import sys
import time

import numpy as np
import openvino as ov

CFG = {"INFERENCE_PRECISION_HINT": "f16", "INFERENCE_NUM_THREADS": "4", "NUM_STREAMS": "1"}


def rss_mb():
    for riga in open("/proc/self/status"):
        if riga.startswith("VmRSS"):
            return round(int(riga.split()[1]) / 1024, 1)


def stat(t):
    t = np.array(t) * 1e3
    return {"n": len(t), "median_ms": round(float(np.median(t)), 3), "p99_ms": round(float(np.percentile(t, 99)), 3),
            "max_ms": round(float(t.max()), 3)}


def carica(core, xml, cache=None):
    if cache:
        core.set_property({"CACHE_DIR": cache})
    t0 = time.perf_counter()
    m = core.read_model(xml)
    t1 = time.perf_counter()
    cm = core.compile_model(m, "CPU", CFG)
    t2 = time.perf_counter()
    return cm, t1 - t0, t2 - t1


class Rete:
    def __init__(self, cm, hw):
        self.req = cm.create_infer_request()
        self.x = np.random.default_rng(0).random((1, 3, *hw), dtype=np.float32)

    def __call__(self):
        t0 = time.perf_counter()
        self.req.infer({0: self.x})
        self.req.get_output_tensor(0).data.copy()
        return time.perf_counter() - t0


def hw(s):
    return tuple(int(v) for v in s.split("x"))


if sys.argv[1] == "carico":
    xml, size = sys.argv[2], hw(sys.argv[3])
    cache = sys.argv[4] if len(sys.argv) > 4 else None
    rss0 = rss_mb()
    cm, t_read, t_comp = carica(ov.Core(), xml, cache)
    r = Rete(cm, size)
    prima = r()
    seconda = r()
    for _ in range(30):
        r()
    reg = [r() for _ in range(100)]
    print(json.dumps({"modo": "carico", "xml": xml.split("/")[-2][:40], "cache": bool(cache), "read_ms": round(t_read * 1e3, 1),
                      "compile_ms": round(t_comp * 1e3, 1), "prima_inferenza_ms": round(prima * 1e3, 2),
                      "seconda_inferenza_ms": round(seconda * 1e3, 2), "regime": stat(reg),
                      "rss_prima_mb": rss0, "rss_dopo_mb": rss_mb()}))
else:
    xs, hs, xt, ht = sys.argv[2], hw(sys.argv[3]), sys.argv[4], hw(sys.argv[5])
    core = ov.Core()
    cs, _, _ = carica(core, xs)
    ct, _, _ = carica(core, xt)
    scan, track = Rete(cs, hs), Rete(ct, ht)
    sporco = np.zeros(4 * 1024 * 1024, np.float32)    # 16 MB: sporca la cache come il resto della pipeline

    def sporca():
        sporco.sum()

    for _ in range(50):
        scan(), track()
    out = {"modo": "cambio", "rss_due_modelli_mb": rss_mb()}
    N = 300
    out["scan_solo"] = stat([scan() for _ in range(N)])
    out["track_solo"] = stat([track() for _ in range(N)])
    a, b = [], []
    for _ in range(N):
        a.append(scan())
        b.append(track())
    out["alterna_scan"], out["alterna_track"] = stat(a), stat(b)
    for nome, gap in (("ciclo", None), ("ciclo_sporco", sporca)):
        sc, primo, resto = [], [], []
        for _ in range(10):
            for k in range(30):
                if gap:
                    gap()
                if k == 0:
                    sc.append(scan())
                else:
                    (primo if k == 1 else resto).append(track())
        out[nome + "_scan"], out[nome + "_track_dopo_scan"], out[nome + "_track_altri"] = stat(sc), stat(primo), stat(resto)
    print(json.dumps(out))
