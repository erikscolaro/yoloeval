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
    "adapt.epochs=7",
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
    "adapt.epochs=7",
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
    assert "jetson_orin_nx" not in key


def test_export_key_axelera_include_la_versione_sdk():
    """Un aggiornamento dell'SDK invalida i .axm compilati in precedenza."""
    cfg = make_cfg("backend=axelera", "hardware=rpi5", "quantization=int8")
    assert export_key(cfg).endswith("_sdk1.8.0")


def test_export_key_cambia_con_arch():
    x86 = export_key(make_cfg("backend=onnxruntime", "hardware=wks4_rtx6000"))
    arm = export_key(make_cfg("backend=onnxruntime", "hardware=jetson_orin"))
    assert x86 != arm


# --- strategie di compressione (yolopit) -----------------------------------------------

from src.cache import (adapt_key, finetune_key, is_baseline, model_label,  # noqa: E402
                       search_key, weights_key)


@pytest.mark.parametrize("override", [
    "adapt.epochs=7",
    "dataset.name=altro",
    "train.epochs=42",
    "pretrain_dataset.name=altro",
    "model=yolo26s",
])
def test_adapt_key_cambia_con_adapt_destinazione_e_training(override):
    assert adapt_key(make_cfg()) != adapt_key(make_cfg(override))


@pytest.mark.parametrize("override", [
    "backend=tensorrt",
    "quantization=int8",
    "hardware=jetson_orin",
    "finetune.epochs=7",
])
def test_adapt_key_non_cambia_con_export_e_hardware(override):
    """Un solo adattamento per modello: vale per ogni board e backend; la baseline ignora il
    gruppo finetune."""
    assert adapt_key(make_cfg()) == adapt_key(make_cfg(override))


def test_adapt_key_baseline_invariata():
    """Lo stesso payload di prima dello spostamento dell'adapt: le cache in artifacts/adapt/
    della baseline restano valide."""
    from src.cache import _group, _sha

    cfg = make_cfg()
    assert adapt_key(cfg) == _sha({
        "training_key": training_key(cfg),
        "adapt": _group(cfg, "adapt"),
        "dataset": _group(cfg, "dataset"),
        "dataset_hash": None,
    })[:8]


@pytest.mark.parametrize("override", [
    "finetune.epochs=7",
    "strategy.search.regularizer.target.ops=30%",
    "adapt.epochs=7",
    "dataset.name=altro",
])
def test_adapt_key_pit_a_valle_del_finetune(override):
    """Per le strategie l'adapt parte dal potato finetunato: cambia con ricerca e finetune."""
    base = make_cfg("strategy=pit_duccio")
    assert adapt_key(base) != adapt_key(make_cfg("strategy=pit_duccio", override))
    assert adapt_key(base) != adapt_key(make_cfg())


def test_baseline_default_e_chiavi_invariate():
    """Con strategy=baseline i pesi che vanno all'export sono quelli dell'adattamento, e il
    gruppo finetune (che la baseline non usa) non entra in nessuna chiave."""
    base = make_cfg()
    assert is_baseline(base)
    assert weights_key(base) == adapt_key(base)
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
    "model.imgsz=320",
    "pretrain_dataset.name=altro",
])
def test_search_key_cambia_con_strategia_training_e_risoluzione(override):
    assert search_key(make_cfg("strategy=pit_duccio")) != \
        search_key(make_cfg("strategy=pit_duccio", override))


@pytest.mark.parametrize("override", ["adapt.epochs=7", "dataset.name=altro", "adapt=none"])
def test_search_key_senza_adapt_ne_destinazione(override):
    """La ricerca parte dal training e gira su AOD4: adapt e dataset di destinazione non ci
    entrano."""
    assert search_key(make_cfg("strategy=pit_duccio")) == \
        search_key(make_cfg("strategy=pit_duccio", override))


def test_finetune_key_e_weights_key():
    base = make_cfg("strategy=pit_duccio")
    assert finetune_key(base) != finetune_key(make_cfg("strategy=pit_duccio",
                                                       "finetune.epochs=7"))
    assert finetune_key(base) != finetune_key(make_cfg(
        "strategy=pit_duccio", "strategy.search.regularizer.target.ops=30%"))
    assert weights_key(base) == adapt_key(base)
    assert len({weights_key(base), finetune_key(base), training_key(base)}) == 3


def test_export_e_cella_distinguono_strategia_e_baseline():
    base = make_cfg()
    pit = make_cfg("strategy=pit_duccio")
    assert export_key(pit).startswith("yolo26n-pit_duccio_")
    assert export_key(pit) != export_key(base)
    assert cell_key(pit) != cell_key(base)
    assert model_label("yolo26n", "baseline") == "yolo26n"


def test_pretrain_dataset_punta_ad_aod4():
    """I campi del dataset si risolvono nel loro gruppo: con `${dataset.local_path}` il training
    di `pretrain_dataset` finirebbe sul dataset di destinazione."""
    cfg = make_cfg()
    assert cfg.pretrain_dataset.yaml.endswith("data/aod4/dataset.yaml")
    assert cfg.dataset.yaml.endswith("data/svanstrom/all/dataset.yaml")


# --- adapt facoltativo -------------------------------------------------------------------

from src.cache import adapt_enabled, final_weights  # noqa: E402


@pytest.mark.parametrize("strategy, upstream", [("baseline", training_key),
                                                ("pit_duccio", finetune_key)])
def test_adapt_none_manda_all_export_i_pesi_dello_stadio_prima(strategy, upstream):
    on = make_cfg(f"strategy={strategy}")
    off = make_cfg(f"strategy={strategy}", "adapt=none")
    assert adapt_enabled(on) and not adapt_enabled(off)
    assert weights_key(off) == upstream(off) == upstream(on)
    assert weights_key(off) != weights_key(on)
    assert export_key(off) != export_key(on)
    assert cell_key(off) != cell_key(on)


def _meta(root, kind, name, **keys):
    import json

    d = root / kind / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(keys))
    return d / "best.pt"


@pytest.mark.parametrize("strategy", ["baseline", "pit_duccio"])
def test_final_weights(tmp_path, strategy):
    on = make_cfg(f"strategy={strategy}", f"artifacts_dir={tmp_path}")
    off = make_cfg(f"strategy={strategy}", f"artifacts_dir={tmp_path}", "adapt=none")
    assert final_weights(on) == (None, adapt_key(on))
    kind, field = (("weights", "training_key") if strategy == "baseline"
                   else ("finetune", "finetune_key"))
    prima = _meta(tmp_path, kind, "a", **{field: weights_key(off)})
    adattati = _meta(tmp_path, "adapt", "b", adapt_key=adapt_key(on))
    assert final_weights(off) == (prima, weights_key(off))
    assert final_weights(on) == (adattati, adapt_key(on))
