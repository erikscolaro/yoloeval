"""Is the SiLU fused into the conv in the graph OpenVINO compiles on the Pi (CPU, f16, 4 threads)?

  python check_fusion.py     -> results/fusion.json (runtime ops of a few nets, one per kernel/stride/map type,
                              and performance counters of the last of 20 inferences)
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import RES, build, make_cfg, to_ir  # noqa: E402

NETS = [("k1s1_20", "conv", 64, 64), ("k1s1_160", "conv", 32, 32), ("k3s1_20", "conv", 64, 128),
        ("k3s1_80", "conv", 16, 32), ("k3s2_40", "conv", 64, 64), ("k3s2_320", "conv", 3, 16),
        ("k3s1_20", "conv", 63, 128), ("k1s1_20", "conv", 1, 1), ("id160", "identity", 32, 32),
        ("id40", "identity", 128, 128)]
REMOTE = """
import json, sys, openvino as ov
core = ov.Core()
out = {}
for p in sys.argv[1:]:
    c = core.compile_model(p, "CPU", {"INFERENCE_PRECISION_HINT": "f16", "INFERENCE_NUM_THREADS": 4})
    out[p] = [dict(name=o.get_friendly_name(), type=o.get_rt_info()["layerType"].astype(str),
                   original=o.get_rt_info()["originalLayersNames"].astype(str),
                   precision=o.get_rt_info()["runtimePrecision"].astype(str))
              for o in c.get_runtime_model().get_ordered_ops()]
    # performance counters: which runtime nodes really run at every inference, and for how long
    c = core.compile_model(p, "CPU", {"INFERENCE_PRECISION_HINT": "f16", "INFERENCE_NUM_THREADS": 4,
                                      "PERF_COUNT": True})
    rq = c.create_infer_request()
    for _ in range(20):
        rq.infer()
    out[p + "#perf"] = [dict(name=i.node_name, type=i.node_type, status=str(i.status).split(".")[-1],
                             exec_type=i.exec_type, real_us=i.real_time.total_seconds() * 1e6)
                        for i in rq.profiling_info]
print(json.dumps(out))
"""


def main():
    from src.remote.connection import connection
    cfg = make_cfg()
    d = f"{cfg.hardware.remote.workdir}/lut_pi/fusion"
    with connection(cfg) as conn:
        conn.run(f"mkdir -p {d}", hide=True)
        paths = []
        for f, kind, ci, co in NETS:
            p = f"{d}/{f}_{ci}_{co}.xml"              # same IR as run.py: fp16, NHWC input/output
            xml, bin_ = to_ir(build(f, kind, ci, co), nhwc=kind == "conv")
            conn.put(io.BytesIO(xml), p)
            conn.put(io.BytesIO(bin_), p[:-4] + ".bin")
            paths.append(p)
        conn.put(io.StringIO(REMOTE), f"{d}/fusion.py")
        r = conn.run(f"{cfg.hardware.remote.python} {d}/fusion.py {' '.join(paths)}", hide=True)
    raw = json.loads(r.stdout.strip().splitlines()[-1])
    ops = {Path(k).stem: v for k, v in raw.items() if not k.endswith("#perf")}
    perf = {Path(k[:-5]).stem: v for k, v in raw.items() if k.endswith("#perf")}
    res = {}
    for net, lst in ops.items():
        compute = [o for o in lst if o["type"] not in ("Parameter", "Result", "Input", "Output", "Reorder")]
        # fused: one runtime node carries conv, sigmoid and silu among its original layers
        fused = any({"conv", "sigmoid", "silu"} <= set(o["original"].split(",")) for o in compute)
        converts = [o["type"] for o in lst if o["type"] == "Convert" or "Convert" in o["original"]]
        executed = [q["type"] for q in perf[net] if q["status"] == "EXECUTED"]
        res[net] = {"fused": fused, "converts": converts, "executed_per_infer": executed, "compute_ops": compute, "all_types": [o["type"] for o in lst]}
        res[net]["perf_last_infer"] = perf[net]
        print("   perf:", [(q["name"], q["type"], q["status"], round(q["real_us"], 1)) for q in perf[net]])
        print(net, "SiLU fusa" if fused else "NON fusa", f"Convert: {len(converts)}", "eseguiti:", executed, [(o["type"], o["original"], o["precision"]) for o in compute])
    (RES / "fusion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
