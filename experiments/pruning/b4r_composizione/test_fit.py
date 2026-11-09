"""python -m pytest experiments/pruning/b4r_composizione/test_fit.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit  # noqa: E402


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
