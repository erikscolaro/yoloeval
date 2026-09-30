"""Roofline: complessita' dal grafo ONNX, tetti misurati, confronto con la baseline."""

from __future__ import annotations

import json

import pytest

from src.measure import complexity
from src.probe import dummy
from tests.conftest import make_cfg

pytestmark = pytest.mark.usefixtures("no_dataset_hash")


def test_complessita_di_una_rete_nota(tmp_path):
    s = dummy.Shape(kernel=3, hw=8, layers=2)
    cx = complexity.onnx_complexity(dummy.artifact(s, 12, "fp32", tmp_path))
    assert cx["compute_layers"] == 2
    assert cx["macs"] == 2 * 12 * 12 * 9 * 64            # senza il MAC del bias
    assert cx["weight_elems"] == 2 * (12 * 12 * 9 + 12)
    assert cx["act_elems"] == 2 * 2 * 12 * 64             # input + output di ogni conv
    fp32, int8 = (complexity.with_precision(cx, p) for p in ("fp32", "int8"))
    assert fp32["bytes"] == 4 * int8["bytes"]
    assert int8["intensity_mac_per_byte"] == pytest.approx(4 * fp32["intensity_mac_per_byte"])
    lin = dummy.Shape(kind="linear", hw=10, layers=1)
    assert complexity.onnx_complexity(dummy.artifact(lin, 12, "fp32", tmp_path))["macs"] == \
        10 * 12 * 12


def _cell(tmp_path, name, strategy, n, ms, macs, map_):
    rec = {"status": "ok", "cell_id": name,
           "axes": {"board": "b", "compute_target": "cpu_1", "quantization": "fp32",
                    "backend": "onnxruntime", "model": "yolo26n", "strategy": strategy},
           "latency": {"median_ms": ms},
           "complexity": complexity.with_precision(
               {"macs": macs, "weight_elems": 1e6, "act_elems": 4e6, "params": 0,
                "compute_layers": 1}, "fp32"),
           "model_info": {"strategy": strategy, "pit_n": n},
           "accuracy": {"map50_95": map_}}
    (tmp_path / f"{name}.json").write_text(json.dumps(rec))


def test_confronto_con_la_baseline_e_tetti(tmp_path):
    from tools.roofline import add_baseline_comparison, load_roofs, main, rows_from_results

    (tmp_path / "roofline").mkdir()
    (tmp_path / "roofline" / "r.json").write_text(json.dumps({
        "board": "b", "compute_target": "cpu_1", "quantization": "fp32",
        "backend": "onnxruntime_py", "peak_gmacs": 100.0, "bandwidth_gbs": 10.0,
        "ridge_mac_per_byte": 10.0, "created_at": "2026", "datasheet": None}))
    _cell(tmp_path, "base", "baseline", None, 10.0, 1e9, 0.50)
    _cell(tmp_path, "n16", "pit_duccio", 16, 5.0, 4e8, 0.48)
    rows = rows_from_results(tmp_path, load_roofs(tmp_path))
    add_baseline_comparison(rows)
    r = {x["label"]: x for x in rows}
    assert set(r) == {"baseline", "pit_duccio N=16"}
    p = r["pit_duccio N=16"]
    assert p["speedup_vs_baseline"] == pytest.approx(2.0)
    assert p["mac_ratio_vs_baseline"] == pytest.approx(0.4)
    assert p["map_delta_vs_baseline"] == pytest.approx(-0.02)
    assert p["roof_backend"] == "onnxruntime_py"          # tetto di un altro backend
    # intensita' 4e8 / (5e6 * 4 byte) = 20 MAC/byte > ridge 10: compute-bound
    assert p["bound"] == "compute" and p["attainable_gmacs"] == pytest.approx(100.0)
    assert p["efficiency"] == pytest.approx(80 / 100)      # 4e8 MAC in 5 ms = 80 GMAC/s
    assert main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "roofline.csv").exists()
    assert (tmp_path / "out" / "b_cpu_1_fp32_onnxruntime.png").exists()


def test_stage_roofline_locale(tmp_path):
    from src.stages.roofline import run_roofline

    cfg = make_cfg(
        "stage=roofline", "backend=onnxruntime_py", "compute_target=cpu_1",
        f"results_dir={tmp_path / 'results'}", f"artifacts_dir={tmp_path / 'artifacts'}",
        "stage.iters=3", "stage.thermal.timeout_s=0", "stage.bandwidth.elements=1000000",
        "stage.compute_shapes=[{kind: conv, kernel: 3, hw: 8, layers: 2, c: 32}]",
    )
    (path,) = run_roofline(cfg)
    r = json.loads(path.read_text())
    assert r["peak_gmacs"] > 0 and r["bandwidth_gbs"] > 0
    assert r["ridge_mac_per_byte"] == pytest.approx(r["peak_gmacs"] / r["bandwidth_gbs"])
