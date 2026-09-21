"""Parser dei tool di misura.

Un parser che sbaglia non produce un errore: produce un numero. Per questo i
test partono da output reali salvati in `fixtures/` e verificano anche il caso
opposto — che un output troncato o di formato diverso sollevi `ParseError`
invece di restituire zeri.
"""

from __future__ import annotations

import pytest

from src.backends import get_backend
from src.errors import ParseError
from src.measure.power import parse_tegrastats
from src.measure.rpi5 import parse_throttled
from src.validation.graph import parse_trtexec_bindings
from tests.conftest import FIXTURES


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_trtexec():
    lat = get_backend("tensorrt").parse(read("trtexec_int8.txt"))
    assert lat.mean_ms == pytest.approx(3.73579)
    assert lat.median_ms == pytest.approx(3.70337)
    assert lat.p90_ms == pytest.approx(3.77661)
    assert lat.p95_ms == pytest.approx(3.81616)
    assert lat.p99_ms == pytest.approx(4.14771)
    assert lat.throughput_qps == pytest.approx(267.291)
    # il tempo di sola GPU non sostituisce la latenza: e' un campo a parte
    assert lat.gpu_compute_ms == pytest.approx(3.38561)
    assert lat.gpu_compute_ms < lat.mean_ms
    assert lat.scope == "end_to_end_with_transfers"


def test_trtexec_troncato_solleva():
    with pytest.raises(ParseError):
        get_backend("tensorrt").parse(read("trtexec_truncated.txt"))


def test_trtexec_vuoto_solleva():
    with pytest.raises(ParseError):
        get_backend("tensorrt").parse("")


def test_ort_perf_test():
    lat = get_backend("onnxruntime").parse(read("ort_perf_test.txt"))
    assert lat.mean_ms == pytest.approx(5.17251)
    # i percentili di perf_test sono in secondi: vanno convertiti
    assert lat.median_ms == pytest.approx(5.12)
    assert lat.p99_ms == pytest.approx(6.59)
    assert lat.throughput_qps == pytest.approx(193.296)
    assert lat.iters == 200


def test_ort_senza_percentili_solleva():
    """Senza `-I` perf_test non stampa i percentili: meglio fallire."""
    with pytest.raises(ParseError):
        get_backend("onnxruntime").parse(read("ort_perf_test_no_percentiles.txt"))


def test_benchmark_app():
    lat = get_backend("openvino").parse(read("benchmark_app.txt"))
    assert lat.median_ms == pytest.approx(29.55)
    assert lat.mean_ms == pytest.approx(29.85)
    assert lat.min_ms == pytest.approx(27.77)
    assert lat.throughput_qps == pytest.approx(33.85)
    assert lat.iters == 2000
    # benchmark_app non riporta i percentili alti: restano None, non zero
    assert lat.p99_ms is None


def test_ultralytics_predict():
    lat = get_backend("axelera").parse(read("ultralytics_predict.txt"))
    assert lat.iters == 5
    assert lat.min_ms == pytest.approx(13.6)
    assert lat.max_ms == pytest.approx(15.1)
    assert lat.mean_ms == pytest.approx(14.14, abs=0.01)


def test_ultralytics_solo_riepilogo():
    """Senza righe per immagine resta la media: i percentili sono None."""
    lat = get_backend("axelera").parse(read("ultralytics_speed_only.txt"))
    assert lat.mean_ms == pytest.approx(14.1)
    assert lat.p99_ms is None


def test_ultralytics_output_estraneo_solleva():
    with pytest.raises(ParseError):
        get_backend("axelera").parse("Traceback (most recent call last): ...")


def test_tegrastats():
    power = parse_tegrastats(read("tegrastats.txt"))
    assert power["rail"] == "VDD_IN"
    assert power["n_samples"] == 3
    assert power["mean_power_w"] == pytest.approx(8.389, abs=0.01)


def test_tegrastats_non_riconoscibile():
    assert parse_tegrastats("qualcosa di completamente diverso") is None


def test_vcgencmd_throttled():
    assert parse_throttled("throttled=0x0") == 0
    assert parse_throttled("throttled=0x50005") == 0x50005
    assert parse_throttled("output inatteso") is None


def test_trtexec_bindings():
    bindings = parse_trtexec_bindings(read("trtexec_int8.txt"))
    assert bindings["inputs"][0]["shape"] == (1, 3, 640, 640)
    assert bindings["outputs"][0]["name"] == "output0"
    assert bindings["outputs"][0]["shape"] == (1, 300, 6)
