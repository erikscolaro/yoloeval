"""Celle valide e combinazioni non rappresentabili.

Due comportamenti diversi, ed e' la distinzione che tiene pulita l'analisi:

* una combinazione **non supportata** (TensorRT sul Pi) e' una cella
  `skipped`, con il motivo scritto nel risultato;
* una combinazione **inesistente** (`cpu_8` sul Pi) e' un errore immediato,
  perche' e' un refuso, non un dato.
"""

from __future__ import annotations

import pytest
from omegaconf import open_dict

from src.measure.compute import compute_targets, resolve_compute, resolve_freq
from src.validation.compat import is_valid
from tests.conftest import make_cfg


def test_tensorrt_su_pi_e_saltata():
    cfg = make_cfg("hardware=rpi5", "backend=tensorrt", "freq_target=max",
                   "compute_target=cpu_4")
    valid, reason = is_valid(cfg)
    assert not valid
    assert "tensorrt" in reason


def test_axelera_su_jetson_e_saltata():
    cfg = make_cfg("hardware=jetson_orin", "backend=axelera",
                   "freq_target=maxn", "compute_target=cpu_4")
    valid, reason = is_valid(cfg)
    assert not valid
    assert "Axelera" in reason or "axelera" in reason


def test_cpu8_in_w15_e_saltata_per_requires_freq():
    """Su Jetson il profilo 15W porta online 4 core su 8."""
    cfg = make_cfg("hardware=jetson_orin", "backend=onnxruntime",
                   "freq_target=w15", "compute_target=cpu_8")
    valid, reason = is_valid(cfg)
    assert not valid
    assert "cpu_8" in reason and "w15" in reason


def test_cpu8_in_maxn_e_valida():
    cfg = make_cfg("hardware=jetson_orin", "backend=onnxruntime",
                   "freq_target=maxn", "compute_target=cpu_8")
    assert is_valid(cfg)[0]


def test_tensorrt_su_cpu_e_saltata():
    cfg = make_cfg("hardware=jetson_orin", "backend=tensorrt",
                   "freq_target=maxn", "compute_target=cpu_4")
    valid, reason = is_valid(cfg)
    assert not valid
    assert "CPU" in reason


def test_axelera_fp32_e_saltata():
    """Il Voyager SDK quantizza sempre a INT8: una cella fp32 sarebbe una
    misura di un modello INT8 con l'etichetta sbagliata."""
    cfg = make_cfg("hardware=rpi5", "backend=axelera", "quantization=fp32",
                   "freq_target=max", "compute_target=axelera")
    valid, reason = is_valid(cfg)
    assert not valid
    assert "int8" in reason


def test_axelera_int8_sull_aipu_e_valida():
    cfg = make_cfg("hardware=rpi5", "backend=axelera", "quantization=int8",
                   "freq_target=max", "compute_target=axelera")
    assert is_valid(cfg)[0]


def test_axelera_sul_pi_passa_grazie_al_container():
    """Raspberry Pi OS non e' supportato da Axelera, ma il Voyager SDK gira in
    un container Ubuntu 22.04: la cella e' valida per quello."""
    cfg = make_cfg("hardware=rpi5", "backend=axelera", "quantization=int8",
                   "freq_target=max", "compute_target=axelera")
    assert cfg.hardware.os == "raspbian12"
    assert cfg.backend.requires.os == ["ubuntu22", "ubuntu24"]
    assert is_valid(cfg)[0]


def test_backend_con_os_incompatibile_e_senza_container_e_saltato():
    cfg = make_cfg("hardware=rpi5", "backend=axelera", "quantization=int8",
                   "freq_target=max", "compute_target=axelera")
    with open_dict(cfg):
        cfg.hardware.provision.container = False
        cfg.backend.build.container = False
    valid, reason = is_valid(cfg)
    assert not valid
    assert "raspbian12" in reason


def test_gpu_su_pi_non_e_rappresentabile():
    """Il Pi non dichiara `gpu`: la chiave non esiste, quindi KeyError."""
    cfg = make_cfg("hardware=rpi5", "freq_target=max", "compute_target=gpu")
    with pytest.raises(KeyError):
        resolve_compute(cfg)


def test_cpu8_su_pi_non_e_rappresentabile():
    cfg = make_cfg("hardware=rpi5", "freq_target=max", "compute_target=cpu_8")
    with pytest.raises(KeyError):
        resolve_compute(cfg)


def test_profilo_jetson_su_pi_non_e_rappresentabile():
    cfg = make_cfg("hardware=rpi5", "freq_target=w15")
    with pytest.raises(KeyError):
        resolve_freq(cfg)


def test_compute_target_nullo_significa_tutti():
    cfg = make_cfg("hardware=rpi5", "freq_target=max")
    assert cfg.compute_target is None
    assert compute_targets(cfg) == ["cpu_1", "cpu_2", "cpu_4", "axelera"]


def test_compute_target_esplicito_resta_uno():
    cfg = make_cfg("hardware=rpi5", "freq_target=max", "compute_target=cpu_2")
    assert compute_targets(cfg) == ["cpu_2"]


def test_validazione_dinamica_sui_core_online():
    """Con meno core online del richiesto, la cella e' saltata, non falsata."""
    class FakeConn:
        is_local = True
        host = "fake"

        def run(self, cmd, **kwargs):
            from src.remote.connection import Result

            if "cpu/online" in cmd:
                return Result("0-1", "", 0)
            return Result("", "", 1)

    cfg = make_cfg("hardware=jetson_orin", "backend=onnxruntime",
                   "freq_target=maxn", "compute_target=cpu_8")
    valid, reason = is_valid(cfg, FakeConn())
    assert not valid
    assert "online 2" in reason
