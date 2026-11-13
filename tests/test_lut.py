"""Procedura della LUT del Pi (src/lut): griglia, punti instabili, guardia termica della misura, fit."""
import csv
from types import SimpleNamespace

import numpy as np

from src.lut import fit, pi


def test_griglia_e_reti_b4r():
    assert len(pi.points()) == pi.N_CONV == 71328
    assert len(pi.ARCHS) * (pi.N_NETS + 1) == 765
    assert pi.axis(8, {3}) == [1, 2, 3, 4, 6, 8]


def test_unstable(tmp_path):
    rows = [(1, r, ms) for r, ms in enumerate((1.0, 1.01, 1.02))]          # 2% spread: stable
    rows += [(2, r, ms) for r, ms in enumerate((1.0, 1.0, 1.2))]           # 20%: unstable
    rows += [(2, 3, 5.0), (4, 0, 1.0)]       # extra repeat ignored; point with 1 of 3 repeats skipped
    p = tmp_path / "raw.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["family", "kind", "c_in", "c_out", "repeat", "ms"])
        w.writeheader()
        for c, r, ms in rows:
            w.writerow(dict(family="k1s1_20", kind="conv", c_in=c, c_out=c, repeat=r, ms=ms))
    assert pi.unstable(p) == [("k1s1_20", "conv", 2, 2)]


def test_measure_rifa_la_misura_calda(monkeypatch):
    """Sopra TEMP_MAX_C o in throttling la misura si scarta, si aspetta TEMP_RESUME_C e si rifà."""
    out = "LATENCIES_MS 1 2\nSTATE {} throttled={} 2400000 performance\n"
    runs = iter([out.format(80000, "0x0"), out.format(60000, "0x4"), out.format(60000, "0x0")])
    waits = []
    monkeypatch.setattr("src.measure.thermal.wait_thermal", lambda *a, **k: waits.append(a[2]))
    s = SimpleNamespace(live=SimpleNamespace(run=lambda *a, **k: SimpleNamespace(failed=False, stdout=next(runs))),
                        bc=None, backend=SimpleNamespace(parse=lambda o: SimpleNamespace(iters=2)))
    lat, iters, st, redone = pi.measure(s, "cmd", "net")
    assert (iters, st["temp_c"], st["throttled"], redone) == (["1", "2"], 60.0, "0x0", 2)
    assert waits == [pi.TEMP_RESUME_C] * 2


def test_solve_recovers_c_and_a():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 5, (120, 4))
    a, c = np.array([0.8, 1.2, 1.0, 0.5]), 7.0
    y = c + X @ a + rng.normal(0, 0.01, 120)
    c_hat, a_hat = fit.solve(X, y, 0.0)
    assert abs(c_hat - c) < 0.05 and np.allclose(a_hat, a, atol=0.02)


def test_solve_keeps_a_nonnegative_and_ridge_pulls_to_one():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.5, 5, (60, 2))
    y = 3.0 + X @ np.array([-0.5, 1.0]) + rng.normal(0, 0.01, 60)    # true a_0 < 0: clipped at 0
    _, a = fit.solve(X, y, 0.0)
    assert a[0] >= 0
    _, a_big = fit.solve(X, y, 1e6)
    assert np.allclose(a_big, 1.0, atol=0.01)


def test_merge_groups_joins_collinear_columns():
    rng = np.random.default_rng(2)
    base = rng.uniform(1, 2, 50)
    X = np.stack([base, 2 * base + 1e-4 * rng.normal(size=50), rng.uniform(1, 2, 50)], 1)
    assert fit.merge_groups(X, ["a", "b", "c"]) == [[0, 1], [2]]


def test_solve_one_c_per_architecture():
    rng = np.random.default_rng(3)
    X = rng.uniform(0.5, 5, (90, 3))
    arch = ["p", "q", "r"] * 30
    C = {"p": 2.0, "q": 5.0, "r": 9.0}
    y = np.array([C[k] for k in arch]) + X @ np.array([0.9, 1.1, 0.7]) + rng.normal(0, 0.01, 90)
    c, a = fit.solve(X, y, 0.0, fit.onehot(arch, ["p", "q", "r"]))
    assert np.allclose(c, [2, 5, 9], atol=0.05) and np.allclose(a, [0.9, 1.1, 0.7], atol=0.02)
