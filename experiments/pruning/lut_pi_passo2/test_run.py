"""python -m pytest experiments/pruning/lut_pi_passo2/test_run.py"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402


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
    assert run.unstable(p) == [("k1s1_20", "conv", 2, 2)]
