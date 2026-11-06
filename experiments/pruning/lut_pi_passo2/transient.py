"""How long is the warm-up transient of benchmark_app, and how many iterations does a stable median need?

  python transient.py measure    # 3 nets per family (small, mid, max) + 3 identities, 2000 iterations, 2 repeats
  python transient.py analyze    # results/transient.json, figures/transient.png

Steady state of a measure = median of iterations 1000-1999. Transient = first iteration after which every
window of 20 iterations has its median within TOL of the steady state. Then, for each warm-up W discarded and
N iterations kept, the error of median(it[W:W+N]) against the steady state (median over nets of |error|).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import run  # noqa: E402

ITERS, REPEATS, TOL, WIN = 2000, 2, 0.02, 20
WARMUPS = (0, 10, 20, 50, 100, 200)
KEEPS = (30, 50, 100, 200, 300, 500)


def nets() -> list[tuple]:
    out = []
    for f, (_, _, _, layers) in run.FAMILIES.items():
        ci, co = max(layers, key=lambda l: l[0] * l[1])
        fixed = f in run.FIXED_CIN
        for a, b in ((2, 2), (ci // 2, co // 2), (ci, co)):
            out.append((f, "conv", ci if fixed else max(1, a - a % 2), max(1, b - b % 2) if b != co else co))
    out += [("id20", "identity", 2, 2), ("id40", "identity", 128, 128), ("id160", "identity", 32, 32)]
    return sorted(set(out))


def measure():
    os.environ["ITERS"] = str(ITERS)

    rng = random.Random(1)
    order = []
    for rep in range(REPEATS):
        p = nets()
        rng.shuffle(p)
        order += [(rep, *x) for x in p]
    csv_path = run.RES / "transient.csv"
    csv_path.unlink(missing_ok=True)
    run.run(order, csv_path, run.make_cfg())


def load() -> dict:
    lat = {}
    for line in open(OUT / "runs" / "latencies_transient.txt"):
        rep, f, kind, ci, co, v = line.rstrip("\n").split(",", 5)
        lat[(int(rep), f, kind, int(ci), int(co))] = np.array(v.split(), float)
    return lat


def transient_len(v: np.ndarray, steady: float) -> int:
    med = np.array([np.median(v[i:i + WIN]) for i in range(0, 1000 - WIN + 1)])
    bad = np.nonzero(np.abs(med / steady - 1) > TOL)[0]
    return 0 if not len(bad) else int(bad[-1] + 1)


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.plots import STYLE, color
    plt.style.use(STYLE)

    lat = load()
    rows, err = [], {(w, n): [] for w in WARMUPS for n in KEEPS}
    for k, v in sorted(lat.items()):
        steady = float(np.median(v[1000:2000]))
        rows.append({"rep": k[0], "family": k[1], "kind": k[2], "c_in": k[3], "c_out": k[4],
                     "steady_ms": steady, "transient_iters": transient_len(v, steady),
                     "first10_over_steady": float(v[:10].mean() / steady),
                     "median200_over_steady": float(np.median(v[:200]) / steady),
                     "spread_steady": float((np.percentile(v[1000:], 90) - np.percentile(v[1000:], 10)) / steady)})
        for w in WARMUPS:
            for n in KEEPS:
                err[(w, n)].append(abs(np.median(v[w:w + n]) / steady - 1))
    tr = np.array([r["transient_iters"] for r in rows])
    rep = {}                       # same net, two repeats: steady-state difference (between-run noise)
    for r in rows:
        rep.setdefault((r["family"], r["c_in"], r["c_out"]), []).append(r["steady_ms"])
    between = [abs(a / b - 1) for a, b in (v for v in rep.values() if len(v) == 2)]
    summary = {
        "iters": ITERS, "repeats": REPEATS, "tol": TOL, "window": WIN, "measures": len(rows),
        "transient_iters": {"median": float(np.median(tr)), "p90": float(np.percentile(tr, 90)), "max": int(tr.max())},
        "median200_over_steady": {"median": float(np.median([r["median200_over_steady"] for r in rows])),
                                  "max": float(max(r["median200_over_steady"] for r in rows))},
        "between_repeats_steady_diff": {"median": float(np.median(between)), "max": float(max(between))},
        "median_abs_err": {f"W{w}_N{n}": float(np.median(e)) for (w, n), e in err.items()},
        "p90_abs_err": {f"W{w}_N{n}": float(np.percentile(e, 90)) for (w, n), e in err.items()},
        "measures_detail": rows,
    }
    (run.RES / "transient.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), constrained_layout=True)
    for i, (k, v) in enumerate(sorted(lat.items())):
        if k[0] != 0:
            continue
        sm = np.convolve(v, np.ones(5) / 5, mode="valid") / np.median(v[1000:])
        axes[0].plot(np.arange(len(sm)) + 1, sm, lw=0.5, alpha=0.5,
                     color=color("x", 0 if k[2] == "identity" else 1 + (k[1] in ("k3s2_320", "k1s1_160"))))
    axes[0].set_xscale("log")
    axes[0].axhline(1 + TOL, color="#9e9e9e", lw=0.5)
    axes[0].axhline(1 - TOL, color="#9e9e9e", lw=0.5)
    axes[0].set_ylim(0.9, 3)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("latency / steady state")
    axes[0].set_title("warm-up transient (moving average of 5)", fontsize=8)
    for j, w in enumerate(WARMUPS):
        axes[1].plot(KEEPS, [100 * summary["p90_abs_err"][f"W{w}_N{n}"] for n in KEEPS], marker="o", ms=3,
                     lw=0.9, color=color("x", j), label=f"warm-up {w}")
    axes[1].set_xlabel("iterations kept N")
    axes[1].set_ylabel("p90 |median error| [%]")
    axes[1].set_yscale("log")
    axes[1].set_title("median of it[W:W+N] vs steady state", fontsize=8)
    axes[1].legend(fontsize=7)
    run.RES.parent.joinpath("figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "transient.png", dpi=200)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("measures_detail", "p90_abs_err")}, indent=1))
    print("p90:", {k: round(100 * v, 2) for k, v in summary["p90_abs_err"].items()})


if __name__ == "__main__":
    {"measure": measure, "analyze": analyze}[sys.argv[1]]()
