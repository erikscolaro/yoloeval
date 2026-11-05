"""Is the SiLU fused into the conv in the graph OpenVINO compiles on the Pi (CPU, f16, 4 threads)?

  python check_fusion.py     -> results/fusion.json (runtime ops of a few nets, one per kernel/stride/map type)
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import RES, build, make_cfg  # noqa: E402

NETS = [("k1s1_20", 64, 64), ("k1s1_160", 32, 32), ("k3s1_20", 64, 128), ("k3s1_80", 16, 32),
        ("k3s2_40", 64, 64), ("k3s2_320", 3, 16), ("k3s1_20", 63, 128), ("k1s1_20", 1, 1)]
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
print(json.dumps(out))
"""


def main():
    from src.remote.connection import connection
    cfg = make_cfg()
    d = f"{cfg.hardware.remote.workdir}/lut_pi/fusion"
    with connection(cfg) as conn:
        conn.run(f"mkdir -p {d}", hide=True)
        paths = []
        for f, ci, co in NETS:
            p = f"{d}/{f}_{ci}_{co}.onnx"
            conn.put(io.BytesIO(build(f, "conv", ci, co)), p)
            paths.append(p)
        conn.put(io.StringIO(REMOTE), f"{d}/fusion.py")
        r = conn.run(f"{cfg.hardware.remote.python} {d}/fusion.py {' '.join(paths)}", hide=True)
    ops = {Path(k).stem: v for k, v in json.loads(r.stdout.strip().splitlines()[-1]).items()}
    res = {}
    for net, lst in ops.items():
        compute = [o for o in lst if o["type"] not in ("Parameter", "Result", "Input", "Output", "Reorder")]
        # fused: one runtime node carries conv, sigmoid and silu among its original layers
        fused = any({"conv", "sigmoid", "silu"} <= set(o["original"].split(",")) for o in compute)
        res[net] = {"fused": fused, "compute_ops": compute, "all_types": [o["type"] for o in lst]}
        print(net, "SiLU fusa" if fused else "NON fusa", [(o["type"], o["original"], o["precision"]) for o in compute])
    (RES / "fusion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
