"""Fit B4-R + reti PIT (train) contro il vecchio fit (solo B4-R): reti PIT di test, livello 0, B4-R leave-one-architecture-out.

  python run.py fit    -> results/confronto.csv (per rete), results/valutazione.json, results/fit_pit.json (+ tabelle a schermo)

Riusa src.lut.fit (load, solve, cv_lambda, predict_held_out, errors). Il vecchio fit e' artifacts/lut/rpi5/fit_b4r.json
(o fit.json prima della sostituzione). Candidati: `nuovo` = ridge su B4-R + PIT train (C per architettura ai minimi quadrati);
`nuovo xW` = le reti PIT di train contate W volte; `C su PIT` = a vecchi, C del nano tarata sulle reti PIT di train.
"""
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.lut import fit as F  # noqa: E402

D = Path(__file__).resolve().parent
RES = D / os.environ.get("RES_DIR", "results")
B4R = REPO / "experiments/pruning/b4r_composizione/results"
L0 = REPO / "experiments/pruning/pendenze/livello0/results"
LUT = REPO / "artifacts/lut/rpi5"
OLD = LUT / "fit_b4r.json" if (LUT / "fit_b4r.json").exists() else LUT / "fit.json"
NANO, WEIGHTS, BEST = "w0.25_D126", (3, 10), "nuovo x10"    # x10: le PIT di train pesano quanto B4-R (76 x 10 = 760 reti contro 750)


def l0_data(fams):
    luts = {f: {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"]) for r in csv.DictReader(open(LUT / f"lut_{f}.csv"))}
            for f in fams}
    meas = {r["net"]: float(r["meas_ms"]) for r in csv.DictReader(open(L0 / "confronto.csv"))}
    chk = D / "l0_remisura/results/check.csv"                     # rimisura nella stessa sessione (python run.py l0): se c'e', vale questa
    if chk.exists():
        v = {}
        for r in csv.DictReader(open(chk)):
            v.setdefault(r["net"], []).append(float(r["ms"]))
        meas = {n: statistics.median(x) for n, x in v.items()}
    X, names = {}, []
    for r in csv.DictReader(open(L0 / "layers.csv")):
        X.setdefault(r["net"], np.zeros(len(fams)))[fams.index(r["family"])] += luts[r["family"]][(int(r["c_in"]), int(r["c_out"]))]
    names = sorted(X)
    return names, np.array([X[n] for n in names]), np.array([meas[n] for n in names])


def stats(pred, y):
    e = errors = F.errors(pred, y)
    return {"bias_pct": 100 * e["bias"], "bias_ms": float(np.mean(y - pred)), "p99_pct": 100 * e["p99"],
            "median_pct": 100 * e["median"], "n": e["n"]}


def main():
    nb, fams, Xb, yb, mb, ab, sb = F.load(B4R, LUT)
    npit, fams2, Xp, yp, mp, ap, sp = F.load(RES, LUT)
    assert fams == fams2
    info = {r["net"]: r for r in csv.DictReader(open(RES / "nets.csv"))}
    test = np.array([info[n]["split"] == "test" for n in npit])
    tr = ~test
    old = json.loads(OLD.read_text())
    a_old = np.array([old["a"][f] for f in fams])            # famiglie non unite (merged = [])
    assert not old["merged"]
    C_old = old["C_ms"]
    levels = sorted(set(ab) | set(ap))
    n_b = len(nb)

    cvs = {}

    def fit_new(w):
        """B4-R + PIT train (ripetute w volte); lam per cross-validation su architetture intere."""
        X = np.vstack([Xb] + [Xp[tr]] * w)
        y = np.concatenate([yb] + [yp[tr]] * w)
        ar = ab + list(np.array(ap)[tr]) * w
        lam, cv = F.cv_lambda(X, y, ar)
        C, a = F.solve(X, y, lam, F.onehot(ar, levels))
        cvs[lam] = cv
        return lam, dict(zip(levels, C)), a

    cands = {"vecchio": (None, C_old, a_old)}
    for w in (1, *WEIGHTS):
        cands["nuovo" if w == 1 else f"nuovo x{w}"] = fit_new(w)
    nano_tr = [i for i in range(len(ap)) if tr[i] and ap[i] == NANO]
    C_pit = dict(C_old)
    C_pit[NANO] = float(np.mean(yp[nano_tr] - Xp[nano_tr] @ a_old))      # C del nano dalle PIT di train (la rete intera B4-R no)
    cands["C su PIT"] = (None, C_pit, a_old)

    def pred(c, X, arch):
        return np.array([c[1][ar] for ar in arch]) + X @ c[2]

    out = {"pit_test": {}, "livello0": {}, "pit_train": {}, "loao": {}, "lambda": {k: v[0] for k, v in cands.items()}}
    ap_t, names_t = list(np.array(ap)[test]), [n for n, t in zip(npit, test) if t]
    n0, X0, y0 = l0_data(fams)
    rows = [{"net": n, "set": "pit_test", "meas": y, **{k: round(float(p), 3) for k, p in
            ((k, pred(c, Xp[test][i:i + 1], [ap_t[i]])[0]) for k, c in cands.items())}}
            for i, (n, y) in enumerate(zip(names_t, yp[test]))]
    rows += [{"net": n, "set": "livello0", "meas": y, **{k: round(float(pred(c, X0[i:i + 1], [NANO])[0]), 3)
             for k, c in cands.items()}} for i, (n, y) in enumerate(zip(n0, y0))]
    with open(RES / "confronto.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for k, c in cands.items():
        out["pit_test"][k] = stats(pred(c, Xp[test], ap_t), yp[test])
        out["livello0"][k] = stats(pred(c, X0, [NANO] * len(y0)), y0)
        out["pit_train"][k] = stats(pred(c, Xp[tr], list(np.array(ap)[tr])), yp[tr])
    # B4-R leave-one-architecture-out (C della rete esclusa dalla sola rete intera, come nei report); le PIT dell'arch esclusa non entrano
    loao_err = {}
    full = {a: nb.index(f"{a}_full") for a in sorted(set(ab))}
    skip = set(full.values())
    for tag, w in (("vecchio", 0), ("nuovo", 1), *((f"nuovo x{k}", k) for k in WEIGHTS)):
        X = np.vstack([Xb] + [Xp[tr]] * w)
        y, arch = np.concatenate([yb] + [yp[tr]] * w), ab + list(np.array(ap)[tr]) * w
        mac = np.concatenate([mb] + [mp[tr]] * w)
        P = np.full(n_b, np.nan)
        for h in full:
            s = np.array([a == h for a in ab]) & ~np.isin(np.arange(n_b), list(skip))
            P[s] = F.predict_held_out(X, X, y, mac, arch, full, h)["composition"][:n_b][s]
        ok = ~np.isnan(P)
        grp = {"totale": ok, "20-35 ms": ok & (yb >= 20) & (yb < 35), "nano": ok & (np.array(ab) == NANO)}
        out["loao"][tag] = {g: stats(P[m], yb[m]) for g, m in grp.items()}
        loao_err[tag] = F.errors(P[ok], yb[ok])
    res = json.dumps(out, indent=1)
    (RES / "valutazione.json").write_text(res)
    lam, C, a = cands[BEST]
    X_all, y_all = np.vstack([Xb, Xp[tr]]), np.concatenate([yb, yp[tr]])
    fit = {
           "nets": len(y_all), "architectures": len(levels), "pit_nets": int(tr.sum()), "pit_weight": int(BEST.split("x")[1]),
           "held_out_architectures": old["held_out_architectures"], "test_nets": old["test_nets"], "lambda": lam,
           "cv_mape": {str(k): v for k, v in cvs[lam].items()}, "C_ms": {k: float(v) for k, v in C.items()},
           "a": {f: float(v) for f, v in zip(fams, a)}, "merged": [],
           "errors_test": {**old["errors_test"], "composition": loao_err[BEST]},
           "errors_train": {"composition": F.errors(pred(cands[BEST], X_all, ab + list(np.array(ap)[tr])), y_all)},
           "errors_pit_test": F.errors(pred(cands[BEST], Xp[test], ap_t), yp[test]),
           "errors_by_arch_B4R_in_sample": {}, "repeat_spread": old["repeat_spread"]}
    fit.pop("errors_by_arch_B4R_in_sample")
    (RES / "fit_pit.json").write_text(json.dumps(fit, indent=1))
    for sec in ("pit_test", "livello0", "pit_train"):
        print(sec)
        for k, v in out[sec].items():
            print(f"  {k:10s} n={v['n']:3d} bias {v['bias_pct']:+.2f} %  {v['bias_ms']:+.3f} ms  p99 {v['p99_pct']:.2f} %")
    print("loao")
    for k, g in out["loao"].items():
        print(f"  {k:10s} " + "  ".join(f"{n}: {v['bias_pct']:+.2f}/{v['p99_pct']:.2f}/{v['bias_ms']:+.3f}(n{v['n']})" for n, v in g.items()))
