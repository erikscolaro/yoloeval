"""Rilevamento della testa e fallback end-to-end.

YOLO26 esporta la testa one-to-one (senza NMS) oppure quella one-to-many
(con NMS) a seconda di runtime e quantizzazione, e lo fa in silenzio. Se si
assume quella richiesta, la differenza di latenza del post-processing finisce
attribuita alla quantizzazione.
"""

from __future__ import annotations

from src.validation.graph import classify_head, e2e_fallback_reason
from tests.conftest import make_cfg


def test_testa_one_to_one_e_end_to_end():
    head, e2e = classify_head([(1, 300, 6)], nms_in_graph=False, nc=4)
    assert head == "one_to_one"
    assert e2e is True


def test_testa_one_to_many_non_e_end_to_end():
    head, e2e = classify_head([(1, 8, 8400)], nms_in_graph=False, nc=4)
    assert head == "one_to_many"
    assert e2e is False


def test_nms_nel_grafo_non_e_il_percorso_e2e_di_yolo26():
    head, e2e = classify_head([(1, 300, 6)], nms_in_graph=True, nc=4)
    assert head == "one_to_many+nms"
    assert e2e is False


def test_forma_inattesa_resta_sconosciuta():
    """Non si indovina: si dichiara di non sapere."""
    head, e2e = classify_head([(1, 17)], nms_in_graph=False, nc=4)
    assert head == "unknown"
    assert e2e is False


def test_fallback_noto_trt103_int8_jetpack6():
    cfg = make_cfg("backend=tensorrt", "quantization=int8",
                   "hardware=jetson_orin")
    reason = e2e_fallback_reason(cfg, {"tensorrt": "10.3.0", "jetpack": "6.2"})
    assert reason == "trt 10.3 + int8 + jetpack6"


def test_nessun_fallback_per_fp16_su_trt103():
    cfg = make_cfg("backend=tensorrt", "quantization=fp16",
                   "hardware=jetson_orin")
    assert e2e_fallback_reason(cfg, {"tensorrt": "10.3.0", "jetpack": "6.2"}) is None


def test_fallback_per_trt_vecchio():
    cfg = make_cfg("backend=tensorrt", "quantization=fp32",
                   "hardware=jetson_orin")
    assert e2e_fallback_reason(cfg, {"tensorrt": "8.4.1"}) == "trt < 8.5.0"


def test_env_incompleto_non_esplode():
    cfg = make_cfg("backend=tensorrt", "quantization=int8")
    assert e2e_fallback_reason(cfg, {}) is None
