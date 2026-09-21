"""Regressioni: bug trovati in review, ognuno con il suo test.

Sono tutti della stessa famiglia — il tool girava lo stesso e restituiva
qualcosa. Senza un test che li inchiodi, tornano alla prima rifattorizzazione.
"""

from __future__ import annotations

import pytest

from src.cache import cell_key, compose_export_key, read_dataset_hash
from src.errors import BenchmarkFailed
from src.measure.board import _differs
from src.measure.jetson import JetsonController
from src.remote.connection import Result
from src.timing import PhaseTimer
from tests.conftest import make_cfg
from tests.test_tuning import FakeConn

# --- l'hash del dataset non deve congelarsi --------------------------------

def test_hash_del_dataset_segue_i_dati(tmp_path):
    """Era in cache nel file sentinella: aggiungere immagini non cambiava piu'
    il training_key, e il tool riusava pesi allenati su un altro dataset."""
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    first = read_dataset_hash(tmp_path)
    assert (tmp_path / ".dataset_hash").exists()

    (tmp_path / "b.jpg").write_bytes(b"y" * 20)
    second = read_dataset_hash(tmp_path)

    assert first != second
    assert (tmp_path / ".dataset_hash").read_text().strip() == second


def test_hash_stabile_se_i_dati_non_cambiano(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    assert read_dataset_hash(tmp_path) == read_dataset_hash(tmp_path)


def test_dataset_assente_non_solleva(tmp_path):
    assert read_dataset_hash(tmp_path / "non-esiste") is None


# --- il cell_key deve accorgersi di un riallenamento ------------------------

def test_cell_key_cambia_se_cambia_il_dataset(monkeypatch):
    """Il gruppo `dataset` contiene solo path e nomi: due dataset diversi allo
    stesso path davano la stessa chiave, e la cella veniva saltata al rilancio
    come gia' completata pur misurando un modello diverso."""
    from src import cache

    cfg = make_cfg("hardware=wks4_rtx6000", "compute_target=cpu_1")
    monkeypatch.setattr(cache, "read_dataset_hash", lambda _: "hash-vecchio")
    prima = cell_key(cfg)
    monkeypatch.setattr(cache, "read_dataset_hash", lambda _: "hash-nuovo")
    assert cell_key(cfg) != prima


def test_export_key_composto_in_un_solo_posto():
    """tools.artifacts prune ricostruisce la chiave: se le formule divergono,
    cancella engine che costano ore di compilazione."""
    cfg = make_cfg("model=yolo26s", "quantization=int8", "backend=axelera",
                   "hardware=rpi5")
    from src.cache import export_key, training_key

    atteso = compose_export_key("yolo26s", training_key(cfg), "int8",
                                "axelera", "aarch64", "1.8.0")
    assert export_key(cfg) == atteso


# --- profilo di potenza -----------------------------------------------------

class SequenceConn(FakeConn):
    """Come FakeConn, ma `nvpmodel -q` risponde diversamente ogni volta."""

    def __init__(self, profiles):
        super().__init__()
        self.profiles = list(profiles)

    def run(self, cmd, **kwargs):
        self.commands.append(cmd)
        if "nvpmodel -q" in cmd:
            value = self.profiles.pop(0) if self.profiles else self.profiles
            return Result(f"NV Power Mode: X\n{value}\n", "", 0)
        if "nvpmodel -m" in cmd:
            return Result("This mode requires reboot to take effect", "", 0)
        return super().run(cmd, **kwargs)


def test_profilo_gia_attivo_non_viene_riapplicato(monkeypatch):
    """Riapplicarlo costava SETTLE_S di sleep per ogni cella dello sweep."""
    import src.measure.jetson as jetson

    monkeypatch.setattr(jetson, "SETTLE_S", 0)
    cfg = make_cfg("hardware=jetson_orin", "freq_target=w15",
                   "compute_target=cpu_2")
    conn = SequenceConn([2])            # w15 = nvpmodel 2, gia' attivo
    JetsonController(cfg).set_profile(conn, cfg, "w15")
    assert not any("nvpmodel -m" in c for c in conn.commands)


def test_post_reboot_non_scrive_sul_config(monkeypatch):
    """Scriverlo sollevava ConfigAttributeError (il config di Hydra e' in
    struct mode) subito dopo aver pagato il costo del riavvio."""
    import src.measure.jetson as jetson
    import src.remote.provision as provision

    monkeypatch.setattr(jetson, "SETTLE_S", 0)
    nuova = SequenceConn([2])
    monkeypatch.setattr(provision, "reboot_and_wait", lambda c, cfg, **k: nuova)
    monkeypatch.setattr(provision, "ensure_env", lambda c, cfg, **k: False)

    cfg = make_cfg("hardware=jetson_orin", "freq_target=w15",
                   "compute_target=cpu_2", "allow_reboot=true")
    conn = SequenceConn([0])            # attivo MAXN, serve il cambio
    bc = JetsonController(cfg)
    restituita = bc.set_profile(conn, cfg, "w15")

    assert bc.post_reboot is True
    assert "_post_reboot" not in cfg
    # la connessione restituita e' quella nuova, non quella chiusa dal riavvio
    assert restituita is nuova


# --- avviso sullo stato non ripristinato ------------------------------------

def test_la_frequenza_istantanea_non_fa_scattare_l_avviso():
    """Dopo 200 iterazioni la CPU e' ancora in alto: confrontarla faceva
    apparire l'avviso a ogni cella, seppellendo quello vero."""
    prima = {"governor": "schedutil", "freq_khz": 1000000, "swap_off": False}
    dopo = {"governor": "schedutil", "freq_khz": 2400000, "swap_off": False}
    assert not _differs(prima, dopo)


def test_governor_non_ripristinato_fa_scattare_l_avviso():
    prima = {"governor": "schedutil", "freq_khz": 1000000}
    dopo = {"governor": "performance", "freq_khz": 1000000}
    assert _differs(prima, dopo)


# --- machine hours ----------------------------------------------------------

def test_validazione_e_map_contano_come_calcolo():
    """Una passata di mAP su CPU ARM dura piu' della misura che accompagna."""
    timer = PhaseTimer("jetson_orin_nx")
    timer.add("compute", 10.0)
    timer.add("accuracy", 3600.0)
    timer.add("validation", 60.0)
    timer.add("setup", 30.0)           # attesa: resta fuori
    assert timer.block()["compute_s"] == 3670.0


# --- riga di comando di onnxruntime_perf_test -------------------------------

def test_comando_perf_test():
    """Il modello e' posizionale; senza -s non ci sono percentili e il parser
    solleva; -x e' il flag dei thread intra-op."""
    from src.backends import get_backend

    cfg = make_cfg("hardware=jetson_orin", "backend=onnxruntime",
                   "freq_target=maxn", "compute_target=cpu_4")
    cmd = get_backend("onnxruntime", cfg).build_cmd(cfg, "/b/m.onnx")

    assert cmd.endswith("/b/m.onnx")
    assert " -m times " in cmd and " -r 200 " in cmd
    assert " -x 4 " in cmd and " -s " in cmd and " -e cpu " in cmd
    assert "intra_op_num_threads" not in cmd


# --- flag di sweep che prima erano decorativi -------------------------------

def test_skip_invalid_false_ferma_lo_sweep(tmp_path, monkeypatch):
    from src.stages.benchmark import _CellSkipped, run_cell

    monkeypatch.setattr("src.stages.benchmark.collect_versions",
                        lambda *a, **k: {})
    cfg = make_cfg("hardware=wks4_rtx6000", "backend=axelera",
                   "compute_target=cpu_1", "skip_invalid=false",
                   f"results_dir={tmp_path}", f"artifacts_dir={tmp_path}")
    with pytest.raises(_CellSkipped):
        run_cell(cfg, "dead00000000", tmp_path / "dead00000000.json")
    # il record c'e' lo stesso: fermarsi non significa perdere la cella
    assert (tmp_path / "dead00000000.json").exists()


def test_continue_on_error_false_ferma_lo_sweep(tmp_path, monkeypatch):
    import json

    from src.stages.benchmark import run_cell

    monkeypatch.setattr("src.stages.benchmark.collect_versions",
                        lambda *a, **k: {})
    cfg = make_cfg("hardware=wks4_rtx6000", "backend=onnxruntime",
                   "compute_target=cpu_1", "continue_on_error=false",
                   "stage.thermal.threshold_c=999",
                   "stage.scheduling.rt_priority=null",
                   "stage.scheduling.drop_caches=false",
                   "stage.scheduling.swap_off=false",
                   f"results_dir={tmp_path}", f"artifacts_dir={tmp_path}")
    dest = tmp_path / "beef00000000.json"
    with pytest.raises(BenchmarkFailed):
        run_cell(cfg, "beef00000000", dest)
    assert json.loads(dest.read_text())["status"] == "failed"
