"""Chiavi, naming degli artefatti e indice.

Tre chiavi distinte, con dipendenze deliberatamente diverse:

* `training_key` dipende solo da model, train, dataset (piu' l'hash del
  dataset). Se dipendesse anche da backend o hardware, lo stesso modello
  verrebbe riallenato per ogni cella della matrice.
* `export_key` aggiunge quantizzazione, backend e architettura — piu' la
  versione dell'SDK per Axelera, perche' un aggiornamento invalida i modelli
  gia' compilati.
* `cell_key` identifica la singola misura e include tutti gli assi.

Questo modulo non conosce i backend: riceve il config e produce stringhe.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .jsonio import atomic_write_json, read_json

log = logging.getLogger(__name__)

#: nome del file che porta l'hash del dataset, accanto ai dati
DATASET_HASH_FILE = ".dataset_hash"


def _sha(payload: Any) -> str:
    """SHA256 di una struttura, serializzata in modo deterministico."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def compute_dataset_hash(root: str | Path) -> str:
    """Hash della lista ordinata dei file con le loro dimensioni.

    Non del contenuto: su decine di migliaia di immagini il costo non sarebbe
    sostenibile a ogni lancio. Nome + dimensione intercetta l'aggiunta, la
    rimozione e la sostituzione di immagini, che e' il caso da cui proteggersi:
    il riuso silenzioso di pesi allenati su un dataset diverso.
    """
    root = Path(root)
    entries = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != DATASET_HASH_FILE:
            entries.append((str(p.relative_to(root)), p.stat().st_size))
    return _sha(entries)[:16]


def read_dataset_hash(root: str | Path) -> str | None:
    """Legge `.dataset_hash`, calcolandolo e salvandolo se manca.

    Se il dataset non e' presente su questa macchina ritorna None: gli stadi
    che ne hanno bisogno davvero (il training) falliranno esplicitamente, gli
    altri (benchmark su board remota) non devono averlo.
    """
    root = Path(root)
    if not root.exists():
        log.debug("dataset non presente in %s, hash non calcolabile", root)
        return None
    sentinel = root / DATASET_HASH_FILE
    if sentinel.exists():
        return sentinel.read_text(encoding="utf-8").strip()
    value = compute_dataset_hash(root)
    try:
        sentinel.write_text(value + "\n", encoding="utf-8")
    except OSError:
        log.warning("dataset in sola lettura, %s non scritto", sentinel)
    return value


def _group(cfg: DictConfig, key: str) -> Any:
    value = cfg[key]
    if isinstance(value, (DictConfig, list)) or OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def training_key(cfg: DictConfig) -> str:
    """Dipende SOLO da model, train, dataset e dall'hash del dataset."""
    payload = {k: _group(cfg, k) for k in cfg.training_key_fields}
    payload["dataset_hash"] = read_dataset_hash(cfg.dataset.local_path)
    return _sha(payload)[:8]


def export_key(cfg: DictConfig) -> str:
    key = (
        f"{cfg.model.name}_{training_key(cfg)}"
        f"_{cfg.quantization.name}_{cfg.backend.name}_{cfg.hardware.arch}"
    )
    if cfg.backend.name == "axelera":
        # Un aggiornamento dell'SDK invalida i .axm compilati in precedenza,
        # che vengono rifiutati al caricamento.
        key += f"_sdk{cfg.backend.build.sdk_version}"
    return key


def cell_key(cfg: DictConfig) -> str:
    payload = {k: _group(cfg, k) for k in cfg.cell_key_fields}
    return _sha(payload)[:12]


def training_slug(cfg: DictConfig) -> str:
    """Slug leggibile davanti, hash corto come suffisso disambiguante."""
    return f"{cfg.model.name}_{cfg.dataset.name}_e{cfg.train.epochs}"


def weights_dir(cfg: DictConfig) -> Path:
    return (
        Path(cfg.artifacts_dir)
        / "weights"
        / f"{training_slug(cfg)}_{training_key(cfg)}"
    )


def export_dir(cfg: DictConfig) -> Path:
    return Path(cfg.artifacts_dir) / "exports" / export_key(cfg)


def find_weights(artifacts_dir: str | Path, tkey: str) -> Path | None:
    """Cerca la directory il cui meta.json porta quel training_key.

    Non assume il nome della cartella: lo slug puo' cambiare (per esempio se
    cambia il numero di epoche dichiarato) senza che cambi la chiave.
    """
    root = Path(artifacts_dir) / "weights"
    if not root.is_dir():
        return None
    for d in sorted(root.iterdir()):
        meta = read_json(d / "meta.json")
        if meta and meta.get("training_key") == tkey:
            return d
    return None


def find_export(artifacts_dir: str | Path, ekey: str) -> Path | None:
    root = Path(artifacts_dir) / "exports" / ekey
    if (root / "meta.json").exists():
        return root
    # fallback: scansione, nel caso la cartella sia stata rinominata
    parent = Path(artifacts_dir) / "exports"
    if not parent.is_dir():
        return None
    for d in sorted(parent.iterdir()):
        meta = read_json(d / "meta.json")
        if meta and meta.get("export_key") == ekey:
            return d
    return None


#: estensioni attese per ciascun formato di export
ARTIFACT_SUFFIX = {
    "onnx": [".onnx"],
    "engine": [".engine", ".plan"],
    "axelera": [".axm", ".json"],
    "openvino": [".xml"],
    "pte": [".pte"],
    "torch": [".pt"],
}


def locate_artifact(export_dir_: str | Path, export_format: str) -> Path:
    """Trova il file dell'artefatto dentro la directory dell'export."""
    from .errors import MissingArtifact

    d = Path(export_dir_)
    for suffix in ARTIFACT_SUFFIX.get(export_format, [f".{export_format}"]):
        matches = sorted(d.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    # OpenVINO esporta una directory
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        if sub.name.endswith("_openvino_model") and export_format == "openvino":
            return sub
    raise MissingArtifact(
        f"nessun artefatto {export_format} in {d} "
        f"(attesi: {ARTIFACT_SUFFIX.get(export_format)})"
    )


def update_index(index_path: str | Path, key: str, meta: dict) -> None:
    """Indice leggibile hash -> campi che distinguono le run.

    Serve a `tools.artifacts ls`: senza, per sapere cosa c'e' in cache bisogna
    aprire un meta.json per volta.
    """
    index_path = Path(index_path)
    index = read_json(index_path, default={}) or {}
    cfg = meta.get("config", {})
    index[key] = {
        "model": cfg.get("model", {}).get("name"),
        "dataset": cfg.get("dataset", {}).get("name"),
        "epochs": cfg.get("train", {}).get("epochs"),
        "epochs_completed": meta.get("epochs_completed"),
        "seed": cfg.get("train", {}).get("seed"),
        "imgsz": cfg.get("train", {}).get("imgsz"),
        "metrics": meta.get("metrics"),
        "dataset_hash": meta.get("dataset_hash"),
        "timing": meta.get("timing"),
        "git_sha": (meta.get("env") or {}).get("git_sha"),
        "created_at": (meta.get("timing") or {}).get("started_at"),
    }
    atomic_write_json(index_path, index)
