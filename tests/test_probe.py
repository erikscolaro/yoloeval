"""Probe: scelta di N dalla curva latenza(C), reti dummy, `pit.n: auto`."""

from __future__ import annotations

import json

import pytest

from src.probe import analysis, dummy
from tests.conftest import make_cfg

pytestmark = pytest.mark.usefixtures("no_dataset_hash")


def staircase(step, c_min=32, c_max=128, misaligned_penalty=0.0):
    """Latenza a gradini di `step` canali; i C non multipli possono costare di piu'."""
    lat = {}
    for c in range(c_min, c_max + 1):
        lat[c] = 1.0 + (-(-c // step)) * 0.1
        if c % step:
            lat[c] *= 1 + misaligned_penalty
    return lat


@pytest.mark.parametrize("step", [4, 8, 16, 32])
def test_n_e_la_larghezza_del_gradino(step):
    res = analysis.best_n(staircase(step), tol=0.03)
    assert res["n_opt"] == step
    assert res["tested"][step]["optimal"] and not res["tested"][2 * step]["pass"]


def test_canali_non_allineati_piu_lenti_non_cambiano_n():
    """Il caso visto su ONNX Runtime: fuori dai multipli di 16 il modello e' PIU' lento."""
    assert analysis.best_n(staircase(16, misaligned_penalty=0.3), tol=0.03)["n_opt"] == 16


def test_latenza_lineare_da_n_1():
    lat = {c: 1.0 + 0.01 * c for c in range(32, 129)}
    assert analysis.best_n(lat, tol=0.005)["n_opt"] == 1


def test_curva_piatta_da_at_least():
    res = analysis.best_n({c: 1.0 for c in range(32, 129)}, tol=0.03)
    assert res["at_least"] and res["n_opt"] == max(int(k) for k in res["tested"])


def test_rumore_sotto_tolleranza_ignorato():
    lat = staircase(16)
    lat[40] *= 0.99                                   # 1% "piu' veloce": rumore
    assert analysis.best_n(lat, tol=0.03)["n_opt"] == 16


def test_n_globale_e_il_minimo():
    assert analysis.global_n({"a": {"n_opt": 16}, "b": {"n_opt": 8}}) == 8


def test_rete_dummy_e_costi(tmp_path):
    import onnx
    import onnxruntime as ort
    import numpy as np

    s = dummy.Shape(kernel=3, hw=8, layers=2)
    m = dummy.build(s, 12)
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    assert len(convs) == 2
    assert dummy.costs(s, 12) == {"macs": 2 * 12 * (12 * 9 + 1) * 64,
                                  "params": 2 * 12 * (12 * 9 + 1)}
    for precision in ("fp32", "fp16", "int8"):
        p = dummy.artifact(s, 12, precision, tmp_path)
        assert p.exists() and dummy.artifact(s, 12, precision, tmp_path) == p   # cache
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": np.zeros((1, 12, 8, 8), np.float32)})[0]
        assert out.shape == (1, 12, 8, 8)
    dw = dummy.Shape(kernel=3, hw=8, layers=1, depthwise=True)
    assert onnx.load(str(dummy.artifact(dw, 12, "fp32", tmp_path))).graph.node[0] \
        .attribute[0].i in (12, 1)


def _summary(tmp_path, n, created="2026-01-01T00:00:00", **over):
    d = tmp_path / "probe"
    d.mkdir(exist_ok=True)
    s = {"board": "wks4_rtx6000", "compute_target": "cpu_1", "quantization": "fp32",
         "backend": "onnxruntime", "n_opt": n, "created_at": created, **over}
    (d / f"x{n}_{created[:4]}.summary.json").write_text(json.dumps(s))


def test_n_auto_usa_il_probe_giusto(tmp_path):
    from src.cache import search_key
    from src.stages.probe import find_n

    cfg = make_cfg("strategy=pit_auto", f"results_dir={tmp_path}")
    with pytest.raises(FileNotFoundError, match="stage=probe"):
        find_n(cfg, "cpu_1")
    _summary(tmp_path, 8, "2026-01-01T00:00:00")
    _summary(tmp_path, 32, "2026-02-01T00:00:00", quantization="int8")   # altra precisione
    assert find_n(cfg, "cpu_1")[0] == 8
    k8 = search_key(cfg)
    _summary(tmp_path, 16, "2026-03-01T00:00:00")                        # probe piu' recente
    assert find_n(cfg, "cpu_1")[0] == 16
    assert search_key(cfg) != k8                  # nella chiave entra il valore di N


def test_probe_locale_end_to_end(tmp_path):
    """Probe vero sulla CPU locale con il timer Python (onnxruntime_perf_test non serve)."""
    from src.stages.probe import run_probe

    cfg = make_cfg(
        "stage=probe", "backend=onnxruntime_py", "compute_target=cpu_1",
        f"results_dir={tmp_path / 'results'}", f"artifacts_dir={tmp_path / 'artifacts'}",
        "stage.sweep.c_min=8", "stage.sweep.c_max=24", "stage.plot=true",
        "stage.sweep.shapes=[{kernel: 1, hw: 8, depthwise: false, layers: 2}]",
        "backend.benchmark.iters=20", "backend.benchmark.warmup_iters=5",
        "stage.thermal.timeout_s=0",
    )
    written = run_probe(cfg)
    assert len(written) == 1
    s = json.loads(written[0].read_text())
    assert s["n_opt"] >= 1 and "conv1x1@8x2" in s["per_shape"]
    lines = (tmp_path / "results" / "probe" / f"{s['probe_id']}.jsonl").read_text().splitlines()
    assert len(lines) == 17 + 2                   # 17 valori di C + la misura ripetuta
    assert (tmp_path / "results" / "probe" / f"{s['probe_id']}.png").exists()
