"""Does latency follow the tensor sizes (input, output, weights) instead of the channels?

  python sizes.py        -> results/sizes.json (reads results/points.csv from analyze.py)

Within one family input ~ c_in, output ~ c_out, weights ~ MACs ~ c_in*c_out: a power law in the three
sizes is the separable model again, so here: rank correlations, a linear roofline-like fit
t = t0 + a*in + b*out + c*w, and the jumps of the 1D scans with the sizes where they happen.
Sizes in KB, 2 bytes per value (fp16), no padding or OpenVINO blocked layout.
"""
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FAM = {"A": (1, 40), "B": (3, 20)}          # kernel, spatial size
JUMP = 0.12                                 # step between neighbouring channels counted as a jump
L2_KB = 512                                 # Cortex-A76 L2 per core


def sizes(f, ci, co):
    k, hw = FAM[f]
    return {"in": 2 * ci * hw * hw / 1024, "out": 2 * co * hw * hw / 1024, "w": 2 * ci * co * k * k / 1024}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    P = {(r["family"], int(r["c_in"]), int(r["c_out"])): float(r["median_ms"])
         for r in csv.DictReader(open(OUT / "results" / "points.csv")) if r["kind"] == "conv"}
    noise = {f: d["noise_rel"] for f, d in json.loads((OUT / "results" / "summary.json").read_text()).items()}
    out = {}
    for f in FAM:
        K = [q for q in P if q[0] == f]
        t = np.array([P[q] for q in K])
        S = {n: np.array([sizes(*q)[n] for q in K]) for n in ("in", "out", "w")}
        S["tot"] = S["in"] + S["out"] + S["w"]
        A = np.column_stack([np.ones_like(t), S["in"], S["out"], S["w"]]) / t[:, None]  # relative error
        b, *_ = np.linalg.lstsq(A, np.ones_like(t), rcond=None)
        e = np.abs(A @ b - 1)
        big = S["tot"] > L2_KB
        jumps = []
        for ref in (32, 64, 128):
            for ax in ("c_out", "c_in"):
                key = (lambda x: (f, ref, x)) if ax == "c_out" else (lambda x: (f, x, ref))
                for x in range(2, 129):
                    d = P[key(x)] / P[key(x - 1)] - 1
                    if abs(d) > JUMP:
                        jumps.append({"scan": ax, "other": ref, "at": x, "step": round(d, 3),
                                      **{n: round(v, 1) for n, v in sizes(*key(x)).items()}})
        out[f] = {"spearman_log_t": {n: round(spearman(v, t), 3) for n, v in S.items()},
                  "linear_fit": {"t0_ms": b[0], "ms_per_KB_in": b[1], "ms_per_KB_out": b[2], "ms_per_KB_w": b[3],
                                 "err_med": float(np.median(e)), "err_max": float(e.max()),
                                 "frac_within_2noise": float(np.mean(e <= 2 * noise[f]))},
                  "over_L2": {"n": int(big.sum()), "err_med_in": float(np.median(e[~big])),
                              "err_med_over": float(np.median(e[big])) if big.any() else None},
                  "jumps": jumps}
    (OUT / "results" / "sizes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for f, d in out.items():
        print(f, d["spearman_log_t"], {k: round(v, 3) for k, v in d["linear_fit"].items()}, d["over_L2"],
              f"{len(d['jumps'])} salti")


if __name__ == "__main__":
    main()
