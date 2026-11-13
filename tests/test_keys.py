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
    "pretrain_dataset.name=altro",
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
    "dataset.name=altro",
])
def test_training_key_non_cambia_con_gli_altri_assi(override):
    """Backend, quantizzazione, hardware, profilo e dataset di destinazione NON entrano nel
    training: il training su AOD4 si fa una volta sola."""
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
    "dataset.name=altro",
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
    assert "jetson_agx_orin" not in key


def test_export_key_axelera_include_la_versione_sdk():
    """Un aggiornamento dell'SDK invalida i .axm compilati in precedenza."""
    cfg = make_cfg("backend=axelera", "hardware=rpi5", "quantization=int8")
    assert export_key(cfg).endswith("_sdk1.8.0")


def test_export_key_cambia_con_arch():
    x86 = export_key(make_cfg("backend=onnxruntime", "hardware=wks4_rtx6000"))
    arm = export_key(make_cfg("backend=onnxruntime", "hardware=jetson_orin"))
    assert x86 != arm


# --- strategie di compressione (yolopit) -----------------------------------------------

from src.cache import (finetune_key, is_baseline, model_label, search_key,  # noqa: E402
                       weights_key)


def test_baseline_default_e_chiavi_invariate():
    """Con strategy=baseline i pesi che vanno all'export sono quelli del training, e il
    gruppo finetune (che la baseline non usa) non entra in nessuna chiave."""
    base = make_cfg()
    assert is_baseline(base)
    assert weights_key(base) == training_key(base)
    other = make_cfg("finetune.epochs=7")
    assert cell_key(base) == cell_key(other)
    assert export_key(base) == export_key(other)
    with pytest.raises(ValueError):
        search_key(base)


@pytest.mark.parametrize("override", [
    "backend=tensorrt",
    "quantization=int8",
    "hardware=jetson_orin",
    "freq_target=w15",
    "compute_target=cpu_1",
    "finetune.epochs=7",
])
def test_search_key_non_cambia_con_hardware_export_e_finetune(override):
    """La stessa ricerca vale per ogni board, backend e fine-tuning."""
    base = make_cfg("strategy=pit_duccio", "hardware=jetson_orin")
    other = make_cfg("strategy=pit_duccio", "hardware=jetson_orin", override)
    assert search_key(base) == search_key(other)


@pytest.mark.parametrize("override", [
    "strategy=pit_standard",
    "strategy.search.regularizer.target.ops=30%",
    "strategy.search.pit.n=8",
    "train.epochs=42",
    "model=yolo26s",
    "deploy=480p",
    "pretrain_dataset.name=altro",
])
def test_search_key_cambia_con_strategia_training_e_risoluzione(override):
    assert search_key(make_cfg("strategy=pit_duccio")) != \
        search_key(make_cfg("strategy=pit_duccio", override))


def test_search_key_senza_destinazione():
    """La ricerca parte dal training e gira su AOD4: il dataset di destinazione non ci entra."""
    assert search_key(make_cfg("strategy=pit_duccio")) == \
        search_key(make_cfg("strategy=pit_duccio", "dataset.name=altro"))


def test_finetune_key_e_weights_key():
    base = make_cfg("strategy=pit_duccio")
    assert finetune_key(base) != finetune_key(make_cfg("strategy=pit_duccio",
                                                       "finetune.epochs=7"))
    assert finetune_key(base) != finetune_key(make_cfg(
        "strategy=pit_duccio", "strategy.search.regularizer.target.ops=30%"))
    assert weights_key(base) == finetune_key(base) != training_key(base)


def test_export_e_cella_distinguono_strategia_e_baseline():
    base = make_cfg()
    pit = make_cfg("strategy=pit_duccio")
    assert export_key(pit).startswith("yolo26n_D126-pit_duccio_")
    assert export_key(pit) != export_key(base)
    assert cell_key(pit) != cell_key(base)
    assert model_label("yolo26n", "baseline") == "yolo26n"


def test_pretrain_dataset_si_risolve_nel_suo_gruppo():
    """I campi del dataset si risolvono nel loro gruppo: con `${dataset.local_path}` il training
    di `pretrain_dataset` finirebbe sul dataset di destinazione."""
    cfg = make_cfg("dataset.subset=s16")
    assert cfg.pretrain_dataset.yaml.endswith("data/air-detection/all/dataset.yaml")
    assert cfg.dataset.yaml.endswith("data/air-detection/s16/dataset.yaml")


# --- pesi finali -------------------------------------------------------------------------

from src.cache import final_weights  # noqa: E402


def _meta(root, kind, name, **keys):
    import json

    d = root / kind / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(keys))
    return d / "best.pt"


@pytest.mark.parametrize("strategy", ["baseline", "pit_duccio"])
def test_final_weights(tmp_path, strategy):
    """All'export vanno i pesi del training (baseline) o del fine-tuning (strategie)."""
    cfg = make_cfg(f"strategy={strategy}", f"artifacts_dir={tmp_path}")
    assert final_weights(cfg) == (None, weights_key(cfg))
    kind, field = (("weights", "training_key") if strategy == "baseline"
                   else ("finetune", "finetune_key"))
    pesi = _meta(tmp_path, kind, "a", **{field: weights_key(cfg)})
    assert final_weights(cfg) == (pesi, weights_key(cfg))


# --- strategia pit_latency: LUT e fit del Pi ---------------------------------------------

from src.cache import latency_conf  # noqa: E402
from src.stages.search import search_args  # noqa: E402


def _lut(d, fit="{}"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "lut_k1s1_20.csv").write_text("c_in,c_out,median_ms\n1,1,0.01\n")
    (d / "fit.json").write_text(fit)
    return d


def test_latency_path_relativi_alla_root_e_arch_dal_modello(tmp_path):
    lat = latency_conf(make_cfg("strategy=pit_latency", "target=rpi5_640", "model=yolo26", "model.width=0.09375",
                                "model.depth=D101", project_root=str(tmp_path)))
    assert lat == {"arch": "w0.09375_D101", "estimator": "smooth",
                   "lut_dir": str(tmp_path / "artifacts/lut/rpi5"),
                   "fit": str(tmp_path / "artifacts/lut/rpi5/fit.json")}
    cfg = make_cfg("strategy=pit_latency", "target=rpi5_640", "strategy.search.pit.latency.arch=w0.25_D170",
                   "strategy.search.pit.latency.lut_dir=/abs/lut", project_root=str(tmp_path))
    assert latency_conf(cfg)["arch"] == "w0.25_D170" and latency_conf(cfg)["lut_dir"] == "/abs/lut"
    assert search_args(cfg)["pit"]["latency"] == latency_conf(cfg)
    assert latency_conf(make_cfg("strategy=pit_duccio")) is None


def test_search_key_cambia_con_il_contenuto_di_lut_e_fit(tmp_path):
    d = _lut(tmp_path / "artifacts/lut/rpi5")
    cfg = make_cfg("strategy=pit_latency", "target=rpi5_640", project_root=str(tmp_path))
    k = search_key(cfg)
    assert search_key(cfg) == k
    (d / "fit.json").write_text('{"C_ms": {}}')
    assert search_key(cfg) != k
    (d / "fit.json").unlink()
    with pytest.raises(FileNotFoundError):
        search_key(cfg)
