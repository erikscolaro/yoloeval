"""Why does k1s1_160 48x64 come out either ~3.09 or ~2.75 ms (warmup.py, 06/11)?

  python bimodal.py measure    # LUT protocol (bench_dump, 200 iters): target x20, 4 neighbours x10, shuffled
  python bimodal.py compile    # on the Pi, 3 processes x 8 compiles per net: kernel chosen (primitiveType)
                               # and median of 200 inferences after 60 ms of warm-up, per compile
  python bimodal.py analyze    # results/bimodal.json

measure tells whether the regime is per run (all iterations shifted) or switches inside a run; compile tells
whether it follows the kernel OpenVINO picks, the compile, or the process.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))   # experiments/: _lib puts the repo root in sys.path
from _lib import ROOT  # noqa: E402,F401
from src.lut import pi as run  # noqa: E402  (the LUT procedure, once run.py here)

RES = OUT / "results"

TARGET = ("k1s1_160", "conv", 48, 64)
NEIGHBOURS = [("k1s1_160", "conv", 32, 32), ("k1s1_160", "conv", 48, 62), ("k3s1_160", "conv", 8, 16),
              ("k1s1_80", "conv", 58, 126)]
CSV = RES / "bimodal.csv"
COMPILE = RES / "bimodal_compile.json"   # renamed after each run: *_v1_hintlatency.json, *_v2_hintnone.json
PROCS, COMPILES = 3, 8

REMOTE = """
import json, sys, time, numpy as np, openvino as ov
core = ov.Core()
out = []
for p in sys.argv[2:]:
    for i in range(int(sys.argv[1])):
        # as benchmark_app -hint none -nstreams 1 -nthreads 4 -infer_precision f16 (v1 also had hint LATENCY)
        c = core.compile_model(p, "CPU", {"INFERENCE_PRECISION_HINT": "f16", "INFERENCE_NUM_THREADS": 4,
                                          "NUM_STREAMS": 1})
        prim = {}
        for o in c.get_runtime_model().get_ordered_ops():
            try:
                prim[o.get_friendly_name()] = o.get_rt_info()["primitiveType"].astype(str)
            except Exception:
                prim[o.get_friendly_name()] = "?"
        rq = c.create_infer_request()
        inp = c.input(0)        # fresh random input every compile, as benchmark_app fills its own
        x = ov.Tensor(np.random.default_rng(i).uniform(-1, 1, tuple(inp.get_shape())).astype(np.float16))
        rq.set_input_tensor(x)
        addr = {"in": x.data.ctypes.data % 4096, "out": rq.get_output_tensor().data.ctypes.data % 4096}
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.06:
            rq.infer()
        lat = []
        for _ in range(200):
            t = time.perf_counter(); rq.infer(); lat.append((time.perf_counter() - t) * 1000)
        out.append({"net": p, "compile": i, "median_ms": float(np.median(lat)), "primitive": prim,
                    "addr_mod4096": addr})
        del rq, c
print("RESULT " + json.dumps(out))
"""


def nets():
    return [TARGET, *NEIGHBOURS]


def measure():
    rng = random.Random(4)
    order = [(r, *TARGET) for r in range(20)] + [(r, *n) for n in NEIGHBOURS for r in range(10)]
    rng.shuffle(order)
    os.environ["ITERS"] = "200"                   # as the LUT (src.lut.pi.ITERS_LUT)
    CSV.unlink(missing_ok=True)
    run.run(order, CSV, run.make_cfg())


def compile_():
    from src.measure.board import board_controller, tuned
    from src.measure.compute import resolve_compute
    from src.remote.connection import connection
    from src.stages.benchmark import _effective_cores, _ensure_profile, _wrap_command
    cfg = run.make_cfg()
    d = f"{cfg.hardware.remote.workdir}/lut_pi/bimodal"
    res = []
    with connection(cfg) as conn:
        conn.run(f"mkdir -p {d}", hide=True)
        bc = board_controller(cfg)
        live = _ensure_profile(conn, cfg, bc)
        paths = []
        for f, kind, ci, co in nets():
            p = f"{d}/{f}_{ci}_{co}.xml"
            xml, bin_ = run.to_ir(run.build(f, kind, ci, co))
            live.put(io.BytesIO(xml), p)
            live.put(io.BytesIO(bin_), p[:-4] + ".bin")
            paths.append(p)
        live.put(io.StringIO(REMOTE), f"{d}/probe.py")
        with tuned(live, cfg, bc):
            ct = resolve_compute(cfg)
            n_cores = _effective_cores(live, cfg, bc, ct)
            cmd = _wrap_command(live, cfg, f"{cfg.hardware.remote.python} {d}/probe.py {COMPILES} "
                                f"{' '.join(paths)}", ct, n_cores) + run.STATE
            for proc in range(PROCS):
                while bc.read_temp(live) > run.TEMP_RESUME_C:
                    import time
                    time.sleep(10)
                r = live.run(cmd, hide=True)
                st = run.parse_state(r.stdout)
                rows = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT "))[7:])
                for x in rows:
                    x.update(proc=proc, net=Path(x["net"]).stem, temp_end_c=st["temp_c"], throttled=st["throttled"])
                res += rows
                print(proc, [(x["net"], round(x["median_ms"], 3)) for x in rows], st, flush=True)
    COMPILE.write_text(json.dumps(res, indent=1), encoding="utf-8")


def analyze():
    out = {}
    lat = {}
    for line in open(OUT / "runs" / "latencies_bimodal.txt"):
        rep, f, kind, ci, co, vs = line.rstrip("\n").split(",", 5)
        v = np.array(vs.split(), float)
        lat.setdefault(f"{f}_{ci}_{co}", []).append(
            {"rep": int(rep), "median": float(np.median(v)), "q10": float(np.percentile(v, 10)),
             "q90": float(np.percentile(v, 90)), "first_half": float(np.median(v[:100])),
             "second_half": float(np.median(v[100:]))})
    out["benchmark_app"] = lat
    if COMPILE.exists():
        comp = json.loads(COMPILE.read_text())
        by = {}
        for x in comp:
            conv = {k: v for k, v in x["primitive"].items() if k == "conv" or "conv" in k.lower()}
            by.setdefault(x["net"], []).append({"proc": x["proc"], "compile": x["compile"],
                                                "median_ms": round(x["median_ms"], 4), "conv_primitive": conv,
                                                "addr_mod4096": x.get("addr_mod4096"),
                                                "all_primitives": sorted(set(x["primitive"].values()))})
        out["compile"] = by
    (RES / "bimodal.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in lat.items():
        m = np.array([x["median"] for x in v])
        print(k, "benchmark_app medians:", np.round(np.sort(m), 3).tolist())
        print("   within run (2nd/1st half):", np.round([x["second_half"] / x["first_half"] for x in v], 3).tolist())
    for k, v in out.get("compile", {}).items():
        print(k, [(x["proc"], x["median_ms"], list(x["conv_primitive"].values())) for x in v])


if __name__ == "__main__":
    {"measure": measure, "compile": compile_, "analyze": analyze}[sys.argv[1]]()
