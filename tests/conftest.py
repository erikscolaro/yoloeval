"""Fixture comuni: config reali, composte da `conf/` con l'API di Hydra.

I test girano sui file di configurazione veri, non su dizionari inventati: un
test che passa su un config finto non dice nulla su quello che verra' usato.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


def make_cfg(*overrides: str, project_root: str | None = None) -> DictConfig:
    """Compone la config come farebbe `run.py`, senza avviare un job."""
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"project_root={project_root or ROOT}", *overrides],
        )
    return cfg


@pytest.fixture
def cfg() -> DictConfig:
    return make_cfg()


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def no_dataset_hash(monkeypatch):
    """Il dataset non esiste in CI: l'hash e' None, stabile fra chiamate."""
    from src import cache

    monkeypatch.setattr(cache, "read_dataset_hash", lambda _: None)
    return None


def set_key(cfg: DictConfig, key: str, value) -> DictConfig:
    with open_dict(cfg):
        cfg[key] = value
    return cfg
