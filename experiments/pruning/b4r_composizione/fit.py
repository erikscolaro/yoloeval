"""B4-R: fit of the composition latency = C + sum_f a_f * sum_{layers in f} LUT_f(c_in, c_out), and its check.

  python fit.py     # needs results/raw.csv (measure.py), results/layers.csv (generate.py) and the LUT
                    # (../lut_pi_passo2/results/lut_<family>.csv) -> results/fit.json, results/predictions.csv

- y_j = median of the repeats of net j; x_jf = sum of the LUT medians of its layers in family f.
- Ridge towards 1 with a_f >= 0 (NEW_MASTERPLAN.md, 4a, "Dalla LUT alla latenza della rete"): min |y - C - X a|^2 + lam |a - 1|^2,
  solved as a bounded least-squares problem (scipy lsq_linear); lam by 5-fold cross-validation on the
  training nets. Columns that always move together (correlation > MERGE_R on the training nets) share one a.
- Test: HOLDOUT of the random nets, never used in the fit, plus the unpruned nano ("full").
- Baselines on the same split: "lut_sum" (a_f = 1, only C fitted) and "macs" (C + beta * MACs of yolopit).
C contains everything the LUT does not see: C2PSA, PSABlock, Detect, non-conv nodes, and the per-call fixed
cost counted 74 times by the LUT sum (DECISIONE_IDENTITA.md).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
LUT_RES = OUT.parent / "lut_pi_passo2" / "results"
RES = OUT / "results"
HOLDOUT, SEED, MERGE_R, FOLDS = 0.2, 0, 0.98, 5
LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def solve(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """C, a minimising |y - C - X a|^2 + lam * scale * |a - 1|^2 with a >= 0 (C free). The penalty is
    scaled by mean(X^2) so lam does not depend on the units of the LUT."""
    from scipy.optimize import lsq_linear
    n, k = X.shape
    s = np.sqrt(lam * max(float(np.mean(X ** 2)), 1e-12))
    A = np.vstack([np.hstack([np.ones((n, 1)), X]), np.hstack([np.zeros((k, 1)), s * np.eye(k)])])
    b = np.concatenate([y, s * np.ones(k)])
    r = lsq_linear(A, b, bounds=(np.r_[-np.inf, np.zeros(k)], np.full(k + 1, np.inf)))
    return float(r.x[0]), r.x[1:]


def merge_groups(X: np.ndarray, names: list[str]) -> list[list[int]]:
    """Greedy: a column joins the first group whose first column it correlates with above MERGE_R."""
    groups: list[list[int]] = []
    for j in range(X.shape[1]):
        for g in groups:
            if np.std(X[:, j]) > 0 and np.std(X[:, g[0]]) > 0 and np.corrcoef(X[:, j], X[:, g[0]])[0, 1] > MERGE_R:
                g.append(j)
                break
        else:
            groups.append([j])
    return groups


def cv_lambda(X: np.ndarray, y: np.ndarray, rng) -> tuple[float, dict]:
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, FOLDS)
    score = {}
    for lam in LAMBDAS:
        err = []
        for k in range(FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[i] for i in range(FOLDS) if i != k])
            c, a = solve(X[tr], y[tr], lam)
            err += list(np.abs((c + X[te] @ a) / y[te] - 1))
        score[lam] = float(np.mean(err))
    return min(score, key=score.get), score


def errors(pred: np.ndarray, y: np.ndarray) -> dict:
    e = np.abs(pred / y - 1)
    return {"mape": float(e.mean()), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "max": float(e.max()), "n": int(len(y))}


def load():
    meas = {}
    for r in csv.DictReader(open(RES / "raw.csv")):
        meas.setdefault(r["net"], []).append(float(r["ms"]))
    nets = {r["net"]: r for r in csv.DictReader(open(RES / "nets.csv"))}
    layers = list(csv.DictReader(open(RES / "layers.csv")))
    fams = sorted({r["family"] for r in layers})
    luts = {}
    for f in fams:
        luts[f] = {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"])
                   for r in csv.DictReader(open(LUT_RES / f"lut_{f}.csv"))}
    names = sorted(n for n in nets if n in meas)
    X = np.zeros((len(names), len(fams)))
    pos = {n: i for i, n in enumerate(names)}
    for r in layers:
        if r["net"] in pos:
            X[pos[r["net"]], fams.index(r["family"])] += luts[r["family"]][(int(r["c_in"]), int(r["c_out"]))]
    y = np.array([np.median(meas[n]) for n in names])
    macs = np.array([float(nets[n]["cost_ops"]) for n in names])
    spread = {n: (max(v) - min(v)) / float(np.median(v)) for n, v in meas.items()}
    return names, fams, X, y, macs, spread


def main():
    names, fams, X, y, macs, spread = load()
    rng = np.random.default_rng(SEED)
    rand = [i for i, n in enumerate(names) if n != "full"]
    test = sorted(rng.choice(rand, int(round(HOLDOUT * len(rand))), replace=False).tolist())
    test += [names.index("full")] if "full" in names else []
    train = [i for i in range(len(names)) if i not in test]
    groups = merge_groups(X[train], fams)
    G = np.stack([X[:, g].sum(1) for g in groups], 1)
    lam, cv = cv_lambda(G[train], y[train], rng)
    c, a = solve(G[train], y[train], lam)
    pred = c + G @ a
    c1 = float(np.mean(y[train] - X[train].sum(1)))                  # baseline: a_f = 1
    pred_sum = c1 + X.sum(1)
    beta = np.polyfit(macs[train], y[train], 1)                      # baseline: C + beta * MACs
    pred_mac = np.polyval(beta, macs)
    tr, te = np.array(train), np.array(test)
    out = {"nets": len(names), "train": len(train), "test": len(test), "lambda": lam, "cv_mape": cv,
           "C_ms": c, "a": {"+".join(fams[j] for j in g): float(v) for g, v in zip(groups, a)},
           "merged": [[fams[j] for j in g] for g in groups if len(g) > 1],
           "errors": {m: {"train": errors(p[tr], y[tr]), "test": errors(p[te], y[te])}
                      for m, p in (("composition", pred), ("lut_sum", pred_sum), ("macs", pred_mac))},
           "baselines": {"lut_sum_C_ms": c1, "macs_beta_ms_per_gmac": float(beta[0] * 1e9),
                         "macs_C_ms": float(beta[1])},
           "repeat_spread": {"median": float(np.median(list(spread.values()))),
                             "max": float(max(spread.values()))}}
    if "full" in names:
        k = names.index("full")
        out["full"] = {"measured_ms": float(y[k]), "composition_ms": float(pred[k]),
                       "lut_sum_ms": float(pred_sum[k]), "macs_ms": float(pred_mac[k])}
    (RES / "fit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(RES / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "split", "measured_ms", "composition_ms", "lut_sum_ms", "macs_ms", "lut_total_ms",
                    "gmacs", "repeat_spread"])
        for i, n in enumerate(names):
            w.writerow([n, "test" if i in test else "train", y[i], pred[i], pred_sum[i], pred_mac[i],
                        X[i].sum(), macs[i] / 1e9, round(spread[n], 5)])
    print(json.dumps({k: v for k, v in out.items() if k != "cv_mape"}, indent=1))


if __name__ == "__main__":
    main()
