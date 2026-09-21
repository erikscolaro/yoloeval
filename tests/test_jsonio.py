"""Scrittura atomica.

Un'interruzione a meta' scrittura lascerebbe un JSON troncato che al rilancio
verrebbe letto come cella completata: la misura non verrebbe piu' rifatta e il
buco resterebbe invisibile fino all'analisi.
"""

from __future__ import annotations

import json

import pytest

from src.jsonio import atomic_write_json, read_json


def test_scrive_e_rilegge(tmp_path):
    dest = tmp_path / "a" / "b.json"
    atomic_write_json(dest, {"x": 1})
    assert json.loads(dest.read_text())["x"] == 1


def test_errore_in_serializzazione_non_lascia_file(tmp_path):
    class Exploding:
        def __str__(self):
            raise RuntimeError("boom")

    dest = tmp_path / "b.json"
    with pytest.raises(RuntimeError):
        atomic_write_json(dest, {"x": Exploding()})
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_sovrascrittura_non_lascia_temporanei(tmp_path):
    dest = tmp_path / "c.json"
    atomic_write_json(dest, {"n": 1})
    atomic_write_json(dest, {"n": 2})
    assert json.loads(dest.read_text())["n"] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_read_json_su_file_corrotto_ritorna_default(tmp_path):
    dest = tmp_path / "d.json"
    dest.write_text("{non json")
    assert read_json(dest, default={"fallback": True}) == {"fallback": True}
