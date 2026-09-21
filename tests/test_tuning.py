"""Tuning temporaneo e profili di potenza.

Le due garanzie che contano: lo stato viene ripristinato **anche** quando il
corpo solleva, e un cambio di profilo che richiede il riavvio non riavvia
niente se non e' stato autorizzato.
"""

from __future__ import annotations

import pytest

from src.errors import ProfileChangeRequiresReboot, ProfileMismatch
from src.measure.board import tuned
from src.measure.jetson import JetsonController, needs_reboot, parse_active_profile
from src.measure.x86 import X86Controller
from src.remote.connection import Result
from tests.conftest import make_cfg


class FakeConn:
    """Connessione finta che registra i comandi e risponde su copione."""

    is_local = True
    host = "fake"

    def __init__(self, replies=None):
        self.commands = []
        self.replies = replies or {}

    def run(self, cmd, **kwargs):
        self.commands.append(cmd)
        for needle, reply in self.replies.items():
            if needle in cmd:
                return reply if isinstance(reply, Result) else Result(reply, "", 0)
        if "scaling_governor" in cmd and cmd.startswith("cat"):
            return Result("schedutil", "", 0)
        if "scaling_cur_freq" in cmd:
            return Result("2400000", "", 0)
        if "cpu/online" in cmd:
            return Result("0-7", "", 0)
        if "swapon" in cmd:
            return Result("", "", 0)
        if "thermal" in cmd:
            return Result("40000", "", 0)
        return Result("", "", 0)

    def close(self):
        pass


def test_tuned_ripristina_anche_su_eccezione():
    cfg = make_cfg("hardware=wks4_rtx6000", "compute_target=cpu_1")
    conn = FakeConn()
    bc = X86Controller(cfg)

    with pytest.raises(RuntimeError):
        with tuned(conn, cfg, bc):
            raise RuntimeError("la misura esplode")

    applicazioni = [c for c in conn.commands if "tee" in c and "governor" in c]
    # una per applicare `performance`, una per rimettere quello di prima
    assert len(applicazioni) >= 2
    assert "schedutil" in applicazioni[-1]


def test_tuned_ripristina_lo_swap_se_era_acceso():
    cfg = make_cfg("hardware=wks4_rtx6000", "compute_target=cpu_1")
    conn = FakeConn({"swapon --show": Result("/swapfile file 2G 0B -2", "", 0)})
    with tuned(conn, cfg, X86Controller(cfg)):
        pass
    assert any("swapoff -a" in c for c in conn.commands)
    assert any("swapon -a" in c for c in conn.commands)


def test_nvpmodel_parsing():
    out = "NV Power Mode: MODE_15W\n2\n"
    assert parse_active_profile(out) == 2
    assert parse_active_profile("") is None


def test_riconoscimento_del_riavvio_necessario():
    assert needs_reboot("This mode requires reboot to take effect")
    assert needs_reboot("tpc_pg_mask change needs a reboot")
    assert not needs_reboot("NV Power Mode: MODE_15W")


def test_profilo_che_richiede_riavvio_si_ferma_senza_autorizzazione():
    cfg = make_cfg("hardware=jetson_orin", "freq_target=w15",
                   "compute_target=cpu_2")
    conn = FakeConn({
        "nvpmodel -m": Result("This mode requires reboot to take effect", "", 0),
    })
    with pytest.raises(ProfileChangeRequiresReboot):
        JetsonController(cfg).set_profile(conn, cfg, "w15")
    # e soprattutto: nessun riavvio partito
    assert not any("shutdown" in c for c in conn.commands)


def test_profilo_attivo_diverso_da_quello_richiesto_si_ferma(monkeypatch):
    """Meglio fermarsi che misurare l'intera matrice sul profilo sbagliato."""
    import src.measure.jetson as jetson

    monkeypatch.setattr(jetson, "SETTLE_S", 0)
    cfg = make_cfg("hardware=jetson_orin", "freq_target=w15",
                   "compute_target=cpu_2")
    conn = FakeConn({"nvpmodel -q": Result("NV Power Mode: MAXN\n0\n", "", 0)})
    with pytest.raises(ProfileMismatch):
        JetsonController(cfg).set_profile(conn, cfg, "w15")


def test_core_online_letti_a_runtime():
    cfg = make_cfg("hardware=jetson_orin", "freq_target=w15")
    conn = FakeConn({"cpu/online": Result("0-3", "", 0)})
    assert JetsonController(cfg).online_cores(conn) == 4
