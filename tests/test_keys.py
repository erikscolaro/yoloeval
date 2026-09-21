"""Chiavi di cache: il test piu' importante della suite.

Se `training_key` dipendesse dal backend o dall'hardware, lo stesso modello
verrebbe riallenato per ogni cella della matrice — e la cosa non si noterebbe
subito, si noterebbe dopo una notte di GPU sprecata.
"""

from __future__ import annotations

import pytest

from src.cache import cell_key, export_key, training_key
from tests.conftest import make_cfg

pytestmark = pytest.mark.usefixtures("no_dataset_hash")


def test_training_key_stabile():
    assert training_key(make_cfg()) == training_key(make_cfg())


@pytest.mark.parametrize("override", [
    "model=yolo26s",
    "train.epochs=42",
    "train.seed=1",
    "dataset.name=altro",
])
def test_training_key_cambia_con_model_train_dataset(override):
    assert training_key(make_cfg()) != training_key(make_cfg(override))


@pytest.mark.parametrize("override", [
    "backend=tensorrt",
    "quantization=int8",
    "hardware=jetson_orin",
    "freq_target=w15",
    "compute_target=cpu_1",
    "eval.bench_conf=0.5",
])
def test_training_key_non_cambia_con_gli_altri_assi(override):
    """Backend, quantizzazione, hardware e profilo NON entrano nel training."""
    base = make_cfg("hardware=jetson_orin")
    other = make_cfg("hardware=jetson_orin", override)
    assert training_key(base) == training_key(other)


@pytest.mark.parametrize("override", [
    "model=yolo26s",
    "quantization=int8",
    "backend=tensorrt",
    "hardware=jetson_orin",
    "freq_target=w15",
    "compute_target=cpu_1",
    "eval.map_iou=0.5",
    "train.epochs=7",
])
def test_cell_key_cambia_per_ogni_asse(override):
    base = make_cfg("hardware=wks4_rtx6000", "compute_target=gpu")
    other = make_cfg("hardware=wks4_rtx6000", "compute_target=gpu", override)
    assert cell_key(base) != cell_key(other)


def test_export_key_leggibile_e_senza_hardware_specifico():
    cfg = make_cfg("model=yolo26s", "quantization=int8", "backend=tensorrt",
                   "hardware=jetson_orin")
    key = export_key(cfg)
    assert key.startswith("yolo26s_")
    assert key.endswith("_int8_tensorrt_aarch64")
    # la board non entra: lo stesso engine vale per ogni Jetson della stessa arch
    assert "jetson_orin_nx" not in key


def test_export_key_axelera_include_la_versione_sdk():
    """Un aggiornamento dell'SDK invalida i .axm compilati in precedenza."""
    cfg = make_cfg("backend=axelera", "hardware=rpi5", "quantization=int8")
    assert export_key(cfg).endswith("_sdk1.8.0")


def test_export_key_cambia_con_arch():
    x86 = export_key(make_cfg("backend=onnxruntime", "hardware=wks4_rtx6000"))
    arm = export_key(make_cfg("backend=onnxruntime", "hardware=jetson_orin"))
    assert x86 != arm
