"""Record di risultato: forma e campi obbligatori.

Le celle fallite e saltate sono risultati a tutti gli effetti. Una cella che
crasha dopo venti minuti di build TensorRT ha occupato la macchina, e il suo
`timing` fa parte delle machine hours esattamente come quello di una riuscita.
"""

from __future__ import annotations

from src.backends.base import LatencyResult
from src.schema import (
    SCHEMA_VERSION,
    base_record,
    failed,
    latency_block,
    ok,
    skipped,
    validate_record,
)
from src.timing import PhaseTimer
from tests.conftest import make_cfg


def _rec(**kw):
    cfg = make_cfg("hardware=jetson_orin", "freq_target=w15",
                   "compute_target=cpu_2", **kw)
    return base_record(cfg, "a91c3e7f0b22", training_key="a3f2c891"), cfg


def test_assi_appiattiti():
    rec, _ = _rec()
    assert rec["axes"]["board"] == "jetson_orin_nx"
    assert rec["axes"]["freq_target"] == "w15"
    assert rec["axes"]["compute_target"] == "cpu_2"
    assert rec["axes"]["training_key"] == "a3f2c891"
    assert rec["schema_version"] == SCHEMA_VERSION


def test_record_ok_conforme():
    rec, _ = _rec()
    lat = LatencyResult(mean_ms=12.4, median_ms=12.1, p90_ms=13.8, p95_ms=14.2,
                        p99_ms=18.9, throughput_qps=80.6, iters=200,
                        raw_stdout="...")
    ok(rec, {
        "latency": latency_block(lat, 200, {"warmup_ms": 2000},
                                 "end_to_end_with_transfers"),
        "runtime_state": {"cores_online": 4},
    })
    rec["timing"] = PhaseTimer("jetson_orin_nx").block()
    assert validate_record(rec) == []
    # il raw stdout non finisce nel JSON: sta nel file .raw.txt accanto
    assert "raw_stdout" not in rec["latency"]


def test_record_skipped_ha_il_motivo():
    rec, _ = _rec()
    skipped(rec, "tensorrt non disponibile su questa board")
    assert rec["status"] == "skipped"
    assert validate_record(rec) == []


def test_record_failed_porta_errore_traccia_e_timing():
    rec, _ = _rec()
    try:
        raise ValueError("engine non caricabile")
    except ValueError as exc:
        failed(rec, exc)
    rec["timing"] = PhaseTimer("jetson_orin_nx").block()
    assert "ValueError" in rec["error"]
    assert "Traceback" in rec["trace"]
    assert validate_record(rec) == []


def test_record_ok_senza_latenza_non_e_conforme():
    rec, _ = _rec()
    ok(rec, {"runtime_state": {}})
    problems = validate_record(rec)
    assert any("latency" in p for p in problems)


def test_scope_non_valido_viene_segnalato():
    rec, _ = _rec()
    lat = LatencyResult(mean_ms=1.0, median_ms=1.0, p99_ms=1.2)
    ok(rec, {
        "latency": latency_block(lat, 200, {}, "quello_che_capita"),
        "runtime_state": {"cores_online": 4},
    })
    rec["timing"] = PhaseTimer("x").block()
    assert any("scope" in p for p in validate_record(rec))


def test_timing_separa_attese_e_calcolo():
    timer = PhaseTimer("wks4_rtx6000")
    timer.add("compute", 10.0)
    timer.add("setup", 5.0)       # attesa: non entra in compute_s
    block = timer.block()
    assert block["compute_s"] == 10.0
    assert block["phase_s"]["setup_s"] == 5.0
    assert block["device"] == "wks4_rtx6000"
