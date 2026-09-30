"""Modelli potati sulle board: yolopit.runtime copiato e messo nel PYTHONPATH di chi carica
il .pt (export Axelera nel container, confronto numerico), senza installarlo con pip."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.remote import sync
from tests.conftest import make_cfg
from tests.test_tuning import FakeConn

pytestmark = pytest.mark.usefixtures("no_dataset_hash")


class RemoteConn(FakeConn):
    is_local = False


@pytest.fixture
def rsync_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(sync, "_rsync", lambda src, dst, **kw: calls.append((src, dst)))
    return calls


def test_niente_per_baseline_o_in_locale(rsync_calls):
    assert sync.ensure_yolopit(RemoteConn(), make_cfg("hardware=rpi5")) is None
    assert sync.ensure_yolopit(FakeConn(), make_cfg("hardware=rpi5",
                                                    "strategy=pit_duccio")) is None
    assert rsync_calls == []


def test_copia_il_pacchetto_installato(rsync_calls):
    import yolopit

    cfg = make_cfg("hardware=rpi5", "strategy=pit_duccio")
    path = sync.ensure_yolopit(RemoteConn(), cfg)
    assert path.startswith("/home/bench/bench/python/yolopit-")
    (src, dst), = rsync_calls
    assert Path(src.rstrip("/")) == Path(yolopit.__file__).resolve().parent
    assert dst.endswith(f"{path}/yolopit/")


def test_export_axelera_con_pythonpath_nel_container(rsync_calls):
    from src.backends.axelera import AxeleraBackend

    cfg = make_cfg("hardware=rpi5", "backend=axelera", "quantization=int8",
                   "strategy=pit_duccio")
    conn = RemoteConn()
    prefix = AxeleraBackend(cfg)._ensure_yolopit(conn, cfg)
    assert prefix.startswith("PYTHONPATH=/bench/python/yolopit-")
    check = conn.commands[-1]
    assert check.startswith("docker exec") and "import yolopit.runtime" in check


def test_import_fallito_nel_container_e_un_errore_chiaro(rsync_calls):
    from src.backends.axelera import AxeleraBackend
    from src.errors import ExportFailed
    from src.remote.connection import Result

    cfg = make_cfg("hardware=rpi5", "backend=axelera", "quantization=int8",
                   "strategy=pit_duccio")
    conn = RemoteConn({"import yolopit.runtime": Result("", "No module named torch", 1)})
    with pytest.raises(ExportFailed, match="yolopit.runtime non si importa"):
        AxeleraBackend(cfg)._ensure_yolopit(conn, cfg)
    assert AxeleraBackend(cfg)._ensure_yolopit(
        RemoteConn(), make_cfg("hardware=rpi5", "backend=axelera")) == ""


def test_la_copia_nel_pythonpath_vince_su_quella_installata(tmp_path):
    """Il meccanismo vero: con il prefisso, Python importa yolopit dalla copia."""
    import yolopit

    shutil.copytree(Path(yolopit.__file__).resolve().parent, tmp_path / "yolopit")
    cmd = (sync.pythonpath_prefix(str(tmp_path)) + f"{sys.executable} -c "
           "'import yolopit.runtime as r; print(r.__file__)'")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)
    assert out.stdout.strip().startswith(str(tmp_path))
