"""Probe: scelta di N dalla curva latenza(C), reti dummy, `pit.n: auto`."""

from __future__ import annotations

import json

import pytest

from src.probe import analysis, dummy
from tests.conftest import make_cfg

pytestmark = pytest.mark.usefixtures("no_dataset_hash")


def padded(step, c_min=8, c_max=96, extra=0.0):
    """Latenza di un kernel che lavora a blocchi di `step` canali: il lavoro e' quello dei
    canali arrotondati al multiplo successivo (piu' un costo fisso per i non allineati)."""
    lat, macs = {}, {}
    for c in range(c_min, c_max + 1):
        pad = -(-c // step) * step
        macs[c] = c * c
        lat[c] = pad * c * (1 + (extra if c % step else 0))
    return lat, macs


@pytest.mark.parametrize("step", [4, 8, 16, 32])
def test_n_e_il_blocco_dell_hardware(step):
    # tolleranza stretta: con C grandi il padding di pochi canali costa poco davvero (su C=90
    # arrotondare a 92 e' il 2%), e con tol=0.05 N=2 passerebbe per step=4
    lat, macs = padded(step, c_max=max(96, 3 * step))
    res = analysis.best_n(lat, macs, tol=0.02)
    assert res["n_opt"] == step
    assert res["tested"][step]["optimal"]
    assert not res["tested"][step // 2]["pass"] if step > 1 else True


def test_non_allineati_piu_lenti_non_cambiano_n():
    lat, macs = padded(16, extra=0.5)
    assert analysis.best_n(lat, macs, tol=0.05)["n_opt"] == 16


def test_hardware_senza_preferenze_da_n_1():
    macs = {c: c * c for c in range(8, 97)}
    lat = {c: 0.5 + c * c * 1e-3 for c in macs}          # efficienza liscia in C
    assert analysis.best_n(lat, macs, tol=0.05)["n_opt"] == 1


def test_misura_sporca_isolata_non_boccia_n():
    lat, macs = padded(16)
    lat[48] *= 1.5                                        # un multiplo di 16 misurato male
    assert analysis.best_n(lat, macs, tol=0.05, max_violations=0.2)["n_opt"] == 16


def test_nessun_n_se_tutto_rumore():
    import random
    rnd = random.Random(0)
    macs = {c: c * c for c in range(8, 97)}
    lat = {c: macs[c] * rnd.uniform(0.3, 1.7) for c in macs}
    res = analysis.best_n(lat, macs, tol=0.05, max_violations=0.0)
    assert res["n_opt"] is None or res["n_opt"] >= 16


def test_n_globale_e_il_massimo_delle_forme_concluse():
    assert analysis.global_n({"a": {"n_opt": 16}, "b": {"n_opt": 8},
                              "c": {"n_opt": None}}) == 16
    assert analysis.global_n({"a": {"n_opt": None}}) is None


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
    _summary(tmp_path, None, "2025-12-01T00:00:00")
    with pytest.raises(ValueError, match="non e' concluso"):
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
        "stage.sweep.repeats=2",
        "stage.sweep.shapes=[{kernel: 1, hw: 8, depthwise: false, layers: 2}]",
        "backend.benchmark.iters=20", "backend.benchmark.warmup_iters=5",
        "stage.thermal.timeout_s=0",
    )
    written = run_probe(cfg)
    assert len(written) == 1
    s = json.loads(written[0].read_text())
    assert "conv1x1@8x2" in s["per_shape"] and "noise_rel" in s
    lines = (tmp_path / "results" / "probe" / f"{s['probe_id']}.jsonl").read_text().splitlines()
    assert len(lines) == 17 * 2                   # 17 valori di C x 2 passate
    assert (tmp_path / "results" / "probe" / f"{s['probe_id']}.png").exists()
