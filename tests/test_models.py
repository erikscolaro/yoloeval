"""Le 27 architetture della scala discreta (NEW_MASTERPLAN §4) si costruiscono, girano e hanno
il numero di conv della loro profondita'."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from src.models import DEPTHS, WIDTHS, arch, check_convs  # noqa: E402


def test_scala_discreta():
    from ultralytics.nn.tasks import DetectionModel

    for w in WIDTHS:
        for d in DEPTHS:
            m = DetectionModel(arch(w, d, 4), verbose=False).eval()
            check_convs(m, d)
            m(torch.zeros(1, 3, 64, 96))


def test_fuori_scala():
    with pytest.raises(ValueError):
        arch(0.4, "D126", 4)
