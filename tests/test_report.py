"""tools.report: nomi delle reti, fronte di Pareto, fallite gia' rifatte."""
import pandas as pd

from tools import report


def test_nomi():
    assert report.arch("yolo26w0.125_D101") == (0.125, "D101")
    assert report.arch("yolo26n_D126") == (0.25, "D126")
    assert report.arch("altro") is None
    assert report.sigla("yolo26w0.1875_D101") == "PN-" and report.sigla("yolo26n_D126") == "N"


def test_fronte():
    pts = pd.DataFrame({"latenza_ms": [10, 20, 15, 30], "mAP50_95": [0.5, 0.6, 0.45, 0.55]})
    assert list(report.front(pts)["latenza_ms"]) == [10, 20]


def test_fallite_rifatte_non_compaiono():
    base = dict(model="yolo26w0.125_D101", deploy="640", quantization="fp16", backend="openvino",
                strategy="baseline", error="MissingExport: x")
    df = pd.DataFrame([{**base, "cell_id": "a", "status": "failed", "timestamp": "2026-11-13T10:00"},
                       {**base, "cell_id": "b", "status": "ok", "timestamp": "2026-11-14T10:00"},
                       {**base, "cell_id": "c", "status": "failed", "timestamp": "2026-11-15T10:00"}])
    assert list(report.failed(df)["cell_id"]) == ["c"]
