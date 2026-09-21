"""Esecuzione di una cella: cosa resta su disco quando va storto.

Una cella che esplode deve lasciare un `results.json` con lo stato, l'errore e
il timing. Senza, verrebbe ritentata alla cieca a ogni rilancio e il tempo che
ha occupato la macchina sparirebbe dalle machine hours.
"""

from __future__ import annotations

import json

import pytest

from src.stages.benchmark import run_cell
from tests.conftest import make_cfg


@pytest.fixture
def cfg_local(tmp_path, monkeypatch):
    monkeypatch.setattr("src.stages.benchmark.collect_versions",
                        lambda *a, **k: {"git_sha": "test"})
    return make_cfg(
        "hardware=wks4_rtx6000",
        "backend=onnxruntime",
        "compute_target=cpu_1",
        f"results_dir={tmp_path}/results",
        f"artifacts_dir={tmp_path}/artifacts",
        # niente attese ne' modifiche di sistema in un test
        "stage.thermal.threshold_c=999",
        "stage.scheduling.rt_priority=null",
        "stage.scheduling.drop_caches=false",
        "stage.scheduling.swap_off=false",
    )


def test_export_mancante_lascia_una_cella_failed(cfg_local, tmp_path):
    dest = tmp_path / "results" / "cafe12345678.json"
    run_cell(cfg_local, "cafe12345678", dest)

    rec = json.loads(dest.read_text())
    assert rec["status"] == "failed"
    assert "MissingExport" in rec["error"]
    assert "Traceback" in rec["trace"]
    # il timing c'e' anche qui: la cella ha comunque occupato la macchina
    assert rec["timing"]["wall_s"] >= 0
    assert rec["timing"]["device"] == "wks4_rtx6000"
    assert rec["axes"]["backend"] == "onnxruntime"
    assert rec["config"]["model"]["name"] == "yolo26n"


def test_cella_non_valida_lascia_una_cella_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("src.stages.benchmark.collect_versions",
                        lambda *a, **k: {})
    cfg = make_cfg(
        "hardware=wks4_rtx6000", "backend=axelera", "compute_target=cpu_1",
        f"results_dir={tmp_path}", f"artifacts_dir={tmp_path}",
    )
    dest = tmp_path / "beef12345678.json"
    run_cell(cfg, "beef12345678", dest)

    rec = json.loads(dest.read_text())
    assert rec["status"] == "skipped"
    assert "Axelera" in rec["reason"]
    # una combinazione non supportata non e' un crash, e in analisi si distingue
    assert "error" not in rec
