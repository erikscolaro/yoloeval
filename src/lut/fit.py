"""B4-R: fit of the composition latency = C_arch + sum_f a_f * sum_{layers in f} LUT_f(c_in, c_out), and its check.

  python -m src.lut.fit <b4r_dir> <lut_res>
      # needs <b4r_dir>/results/raw.csv (src.lut.pi b4r), nets.csv, layers.csv (src.lut.b4r) and the LUT
      # (<lut_res>/lut_<family>.csv) -> <b4r_dir>/results/fit.json, predictions.csv

- y_j = median of the repeats of net j; x_jf = sum of the LUT medians of its layers in family f.
- One C per architecture (width x depth): C2PSA, PSABlock, Detect and the non-conv nodes are not pruned, so what the
  LUT does not see changes with the architecture, not with the pruning. The a_f are shared by every architecture.
- Ridge towards 1 with a_f >= 0 (NEW_MASTERPLAN.md, 4a): min |y - C_arch - X a|^2 + lam |a - 1|^2, bounded least
  squares (scipy lsq_linear); lam by cross-validation grouped by architecture (random folds reward overfitting:
  nets of the same architecture are alike).
- Check, leave-one-architecture-out (nested: lam is chosen again inside each fold): fit on the other architectures, calibrate C of the held-out one on its unpruned
  net (C = y_full - x_full . a), predict its pruned nets. This is the deploy use: a new architecture costs one
  measure of the whole net, then the estimator prices every pruning of it.
- Baselines with the same split and calibration: "lut_sum" (a_f = 1) and "macs" (C_arch + beta * MACs of yolopit).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SEED, MERGE_R, FOLDS = 0, 0.98, 5
LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def solve(X: np.ndarray, y: np.ndarray, lam: float, D: np.ndarray | None = None):
    """C, a minimising |y - D C - X a|^2 + lam * scale * |a - 1|^2 with a >= 0 (C free). D is the one-hot of the
    architecture (one C each); without D a single C, returned as a float. The penalty is scaled by mean(X^2) so lam
    does not depend on the units of the LUT."""
    from scipy.optimize import lsq_linear
    n, k = X.shape
    D1 = np.ones((n, 1)) if D is None else D
    m = D1.shape[1]
    s = np.sqrt(lam * max(float(np.mean(X ** 2)), 1e-12))
    A = np.vstack([np.hstack([D1, X]), np.hstack([np.zeros((k, m)), s * np.eye(k)])])
    b = np.concatenate([y, s * np.ones(k)])
    r = lsq_linear(A, b, bounds=(np.r_[np.full(m, -np.inf), np.zeros(k)], np.full(m + k, np.inf)))
    return (float(r.x[0]) if D is None else r.x[:m]), r.x[m:]


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


def onehot(arch: list[str], levels: list[str]) -> np.ndarray:
    return np.array([[a == l for l in levels] for a in arch], float)


def cv_lambda(X: np.ndarray, y: np.ndarray, arch: list[str]) -> tuple[float, dict]:
    """lam by cross-validation with folds of whole architectures; a held-out C is set to its mean residual."""
    levels = sorted(set(arch))
    folds = np.array_split(np.random.default_rng(SEED).permutation(levels), min(FOLDS, len(levels)))
    score = {}
    for lam in LAMBDAS:
        err = []
        for held in folds:
            te = np.array([a in held for a in arch])
            tr_arch = [a for a, h in zip(arch, te) if not h]
            _, a = solve(X[~te], y[~te], lam, onehot(tr_arch, sorted(set(tr_arch))))
            for h in held:
                s = np.array([x == h for x in arch])
                c = np.mean(y[s] - X[s] @ a)
                err += list(np.abs((c + X[s] @ a) / y[s] - 1))
        score[lam] = float(np.mean(err))
    return min(score, key=score.get), score


def errors(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred / y - 1
    return {"median": float(np.median(abs(e))), "p95": float(np.percentile(abs(e), 95)),
            "p99": float(np.percentile(abs(e), 99)), "max": float(abs(e).max()),
            "bias": float(np.mean(e)), "n": int(len(y))}


def load(res: Path, lut_res: Path):
    meas = {}
    for r in csv.DictReader(open(res / "raw.csv")):
        meas.setdefault(r["net"], []).append(float(r["ms"]))
    nets = {r["net"]: r for r in csv.DictReader(open(res / "nets.csv"))}
    layers = list(csv.DictReader(open(res / "layers.csv")))
    fams = sorted({r["family"] for r in layers})
    luts = {f: {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"])
                for r in csv.DictReader(open(lut_res / f"lut_{f}.csv"))} for f in fams}
    names = sorted(n for n in nets if n in meas)
    X = np.zeros((len(names), len(fams)))
    pos = {n: i for i, n in enumerate(names)}
    for r in layers:
        if r["net"] in pos:
            X[pos[r["net"]], fams.index(r["family"])] += luts[r["family"]][(int(r["c_in"]), int(r["c_out"]))]
    y = np.array([np.median(meas[n]) for n in names])
    macs = np.array([float(nets[n]["cost_ops"]) for n in names]) / 1e9
    arch = [nets[n]["arch"] for n in names]
    spread = {n: (max(v) - min(v)) / float(np.median(v)) for n, v in meas.items()}
    return names, fams, X, y, macs, arch, spread


def predict_held_out(G, X, y, macs, arch, full, h):
    """Fit on every architecture but h (lam chosen on those only: nested); calibrate C_h on the unpruned net of h;
    predict the pruned nets of h."""
    tr = np.array([a != h for a in arch])
    tr_arch = [a for a in arch if a != h]
    lam, _ = cv_lambda(G[tr], y[tr], tr_arch)
    D = onehot(tr_arch, sorted(set(tr_arch)))
    _, a = solve(G[tr], y[tr], lam, D)
    k = full[h]
    _, beta = solve(macs[tr, None], y[tr], 0.0, D)                  # baseline C_arch + beta * MACs, beta >= 0
    return {"composition": y[k] - G[k] @ a + G @ a, "lut_sum": y[k] - X[k].sum() + X.sum(1),
            "macs": y[k] - beta[0] * macs[k] + beta[0] * macs}


def main(res: Path, lut_res: Path):
    names, fams, X, y, macs, arch, spread = load(res, lut_res)
    levels = sorted(set(arch))
    full = {a: names.index(f"{a}_full") for a in levels if f"{a}_full" in names}
    groups = merge_groups(X, fams)
    G = np.stack([X[:, g].sum(1) for g in groups], 1)
    lam, cv = cv_lambda(G, y, arch)
    C, a = solve(G, y, lam, onehot(arch, levels))                   # final model: every net
    pred_train = onehot(arch, levels) @ C + G @ a
    test = {m: np.full(len(y), np.nan) for m in ("composition", "lut_sum", "macs")}
    for h in full:                                                   # leave one architecture out
        p = predict_held_out(G, X, y, macs, arch, full, h)
        s = np.array([x == h for x in arch]) & ~np.isin(np.arange(len(y)), list(full.values()))
        for m in test:
            test[m][s] = p[m][s]
    ok = ~np.isnan(test["composition"])
    out = {"nets": len(names), "architectures": len(levels), "held_out_architectures": len(full),
           "test_nets": int(ok.sum()), "lambda": lam, "cv_mape": {str(k): v for k, v in cv.items()},
           "C_ms": dict(zip(levels, map(float, C))),
           "a": {"+".join(fams[j] for j in g): float(v) for g, v in zip(groups, a)},
           "merged": [[fams[j] for j in g] for g in groups if len(g) > 1],
           "errors_test": {m: errors(p[ok], y[ok]) for m, p in test.items()},
           "errors_train": {"composition": errors(pred_train, y)},
           "errors_test_by_arch": {h: {m: errors(test[m][s], y[s])["median"] for m in test}
                                   for h in full for s in [np.array([x == h for x in arch]) & ok] if s.any()},
           "repeat_spread": {"median": float(np.median(list(spread.values()))),
                             "max": float(max(spread.values()))}}
    (res / "fit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(res / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "arch", "split", "measured_ms", "composition_ms", "lut_sum_ms", "macs_ms", "lut_total_ms",
                    "gmacs", "repeat_spread"])
        for i, n in enumerate(names):
            w.writerow([n, arch[i], "train", y[i], pred_train[i], "", "", X[i].sum(), macs[i], round(spread[n], 5)])
            if ok[i]:
                w.writerow([n, arch[i], "test", y[i], test["composition"][i], test["lut_sum"][i], test["macs"][i],
                            X[i].sum(), macs[i], round(spread[n], 5)])
    print(json.dumps({k: v for k, v in out.items() if k not in ("cv_mape", "C_ms")}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) / "results", Path(sys.argv[2]))
