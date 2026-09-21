"""Resume: cosa si salta e cosa si rifa'.

Interrompere e rilanciare deve riprendere dal punto in cui si era fermati.
Il rischio non e' rifare una cella: e' saltarne una che non era finita.
"""

from __future__ import annotations

import json

import pytest

from src.stages.benchmark import _should_skip
from tests.conftest import make_cfg


@pytest.fixture
def cfg_res(tmp_path):
    return make_cfg(f"results_dir={tmp_path}", "hardware=wks4_rtx6000",
                    "compute_target=cpu_1")


def write(dest, status, **extra):
    dest.write_text(json.dumps({"status": status, "cell_id": "x", **extra}))
    return dest


def test_cella_mancante_si_esegue(cfg_res, tmp_path):
    assert not _should_skip(cfg_res, "x", tmp_path / "x.json")


def test_cella_ok_si_salta(cfg_res, tmp_path):
    dest = write(tmp_path / "x.json", "ok")
    assert _should_skip(cfg_res, "x", dest)


def test_cella_skipped_si_salta(cfg_res, tmp_path):
    dest = write(tmp_path / "x.json", "skipped", reason="non supportata")
    assert _should_skip(cfg_res, "x", dest)


def test_cella_failed_si_salta_senza_flag(cfg_res, tmp_path):
    dest = write(tmp_path / "x.json", "failed", error="boom")
    assert _should_skip(cfg_res, "x", dest)


def test_cella_failed_si_ritenta_con_flag(cfg_res, tmp_path):
    dest = write(tmp_path / "x.json", "failed", error="boom")
    cfg = make_cfg(f"results_dir={tmp_path}", "retry_failed=true")
    assert not _should_skip(cfg, "x", dest)


def test_force_rifa_anche_le_celle_ok(cfg_res, tmp_path):
    dest = write(tmp_path / "x.json", "ok")
    cfg = make_cfg(f"results_dir={tmp_path}", "force=true")
    assert not _should_skip(cfg, "x", dest)


def test_json_troncato_non_viene_scambiato_per_completato(cfg_res, tmp_path):
    """Un file troncato da un'interruzione va rifatto, non creduto."""
    dest = tmp_path / "x.json"
    dest.write_text('{"status": "ok", "latency": {"mean_ms": 12.')
    assert not _should_skip(cfg_res, "x", dest)
