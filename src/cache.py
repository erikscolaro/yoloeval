"""Chiavi, naming degli artefatti e indice.

La catena dei pesi e' training (AOD4) -> [search -> finetune (AOD4)] -> [adapt (Svanstrom)] -> export:

* `training_key` dipende solo da model, train, pretrain_dataset (piu' l'hash del
  dataset). Se dipendesse anche da backend o hardware, lo stesso modello
  verrebbe riallenato per ogni cella della matrice.
* `adapt_key` = chiave a monte (training_key per la baseline, finetune_key per le
  strategie) + gruppo `adapt` + `dataset` (piu' il suo hash): il fine-tuning sul dataset
  di destinazione, ultimo passo prima dell'export. Facoltativo: `adapt=none` lo salta.
* `export_key` aggiunge quantizzazione, backend e architettura — piu' la
  versione dell'SDK per Axelera, perche' un aggiornamento invalida i modelli
  gia' compilati.
* `cell_key` identifica la singola misura e include tutti gli assi.

Con una strategia di compressione (`strategy.method: pit`) la catena si allunga, sul
dataset di pretraining:

* `search_key` = training_key + gruppo `strategy` + versione di yolopit + risoluzione dei
  costi. Niente hardware: la stessa ricerca vale per ogni board.
* `finetune_key` = search_key + gruppo `finetune`.
* `weights_key` e' la chiave dei pesi che arrivano all'export: l'adapt_key, oppure con
  `adapt=none` il training_key per la baseline e il finetune_key per le strategie.

Questo modulo non conosce i backend: riceve il config e produce stringhe.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .jsonio import atomic_write_json, read_json

log = logging.getLogger(__name__)

#: nome del file che porta l'hash del dataset, accanto ai dati
DATASET_HASH_FILE = ".dataset_hash"

#: split che non entrano nel training: cambiarli non deve invalidare i pesi
NON_TRAINING_SPLITS = frozenset({"calib"})


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

    Restano fuori lo split di calibrazione, che non tocca il training, i
    `*.cache` che Ultralytics riscrive accanto alle label e i `*.npy` che scrive
    accanto alle immagini con `cache: disk`: dentro l'hash cambierebbero il
    `training_key` al primo training.
    """
    root = Path(root)
    entries = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if (not p.is_file() or p.name == DATASET_HASH_FILE
                or p.suffix in (".cache", ".npy")
                or NON_TRAINING_SPLITS & set(rel.parts)):
            continue
        entries.append((str(rel), p.stat().st_size))
    return _sha(entries)[:16]


def read_dataset_hash(root: str | Path) -> str | None:
    """Hash corrente del dataset, ricalcolato a ogni chiamata.

    Il file `.dataset_hash` accanto ai dati e' un promemoria leggibile, **non**
    una cache: se lo si rileggesse invece di ricalcolare, aggiungere immagini
    al dataset non cambierebbe piu' il `training_key` e il tool riuserebbe in
    silenzio pesi allenati su un dataset diverso — esattamente cio' da cui
    questo meccanismo deve proteggere.

    Se il dataset non e' presente su questa macchina ritorna None: gli stadi
    che ne hanno bisogno davvero (il training) falliranno esplicitamente, gli
    altri (benchmark su board remota) non devono averlo.
    """
    root = Path(root)
    try:
        present = root.exists()
    except PermissionError:
        # un genitore non attraversabile (es. /data di un altro utente)
        present = False
    if not present:
        log.debug("dataset non presente in %s, hash non calcolabile", root)
        return None
    value = compute_dataset_hash(root)
    sentinel = root / DATASET_HASH_FILE
    try:
        if not sentinel.exists() or sentinel.read_text().strip() != value:
            sentinel.write_text(value + "\n", encoding="utf-8")
    except OSError:
        log.debug("dataset in sola lettura, %s non aggiornato", sentinel)
    return value


# Argomenti di model.train() che cambiano solo come si caricano i dati, non il risultato:
# fuori dalle chiavi, cosi' cambiarli non rifa' training, ricerche e celle gia' in cache.
RUNTIME_ONLY_ARGS = ("workers", "cache")


def _strip_runtime(args: Any) -> None:
    if isinstance(args, dict):
        for k in RUNTIME_ONLY_ARGS:
            args.pop(k, None)


def _group(cfg: DictConfig, key: str) -> Any:
    value = cfg[key]
    if isinstance(value, (DictConfig, list)) or OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if key in ("train", "adapt", "finetune"):
        _strip_runtime(value)
    elif key == "strategy" and isinstance(value, dict):
        _strip_runtime(value.get("search"))
    return value


def training_key(cfg: DictConfig) -> str:
    """Dipende SOLO da model, train, pretrain_dataset e dall'hash del dataset."""
    payload = {k: _group(cfg, k) for k in cfg.training_key_fields}
    payload["dataset_hash"] = read_dataset_hash(cfg.pretrain_dataset.local_path)
    return _sha(payload)[:8]


def adapt_enabled(cfg: DictConfig) -> bool:
    """`adapt=none` (gruppo vuoto) salta l'adattamento."""
    return bool(cfg.get("adapt"))


def pre_adapt(cfg: DictConfig) -> tuple[str, str, str]:
    """(cartella di artifacts/, campo del meta.json, chiave) dei pesi a monte dell'adapt:
    il training per la baseline, il fine-tuning del potato per le strategie."""
    if is_baseline(cfg):
        return "weights", "training_key", training_key(cfg)
    return "finetune", "finetune_key", finetune_key(cfg)


def adapt_key(cfg: DictConfig) -> str:
    """Chiave a monte + gruppo `adapt` + dataset di destinazione e il suo hash. Per la
    baseline il payload e' quello di sempre (training_key in testa): le cache restano."""
    _, field, key = pre_adapt(cfg)
    return _sha({
        field: key,
        "adapt": _group(cfg, "adapt"),
        "dataset": _group(cfg, "dataset"),
        "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
    })[:8]


def weights_chain(cfg: DictConfig) -> list[str]:
    """Gli stadi che producono i pesi dell'export, nell'ordine."""
    stages = ["train"] if is_baseline(cfg) else ["train", "search", "finetune"]
    return stages + (["adapt"] if adapt_enabled(cfg) else [])


def is_baseline(cfg: DictConfig) -> bool:
    """Senza gruppo `strategy` (config vecchie) o con method != pit: solo Ultralytics."""
    strategy = cfg.get("strategy")
    return strategy is None or strategy.get("method", "ultralytics") != "pit"


def strategy_name(cfg: DictConfig) -> str:
    strategy = cfg.get("strategy")
    return "baseline" if strategy is None else str(strategy.name)


def yolopit_version() -> str | None:
    """Versione installata di yolopit, None se non c'e' (basta per la baseline)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("yolopit")
    except PackageNotFoundError:
        return None


def search_key(cfg: DictConfig) -> str:
    """training_key + strategia + yolopit + risoluzione dei costi. Niente hardware, niente
    dataset di destinazione: la ricerca gira sul dataset di pretraining."""
    if is_baseline(cfg):
        raise ValueError("search_key non esiste per strategy=baseline")
    payload = {
        "training_key": training_key(cfg),
        "strategy": _group(cfg, "strategy"),
        "yolopit": yolopit_version(),
        "cost_imgsz": int(cfg.model.imgsz),
    }
    if auto_n(cfg):
        # entra il VALORE di N misurato, non la board: due board con lo stesso N
        # condividono la ricerca
        payload["n_resolved"] = resolve_n(cfg)
    return _sha(payload)[:8]


def auto_n(cfg: DictConfig) -> bool:
    pit = ((cfg.get("strategy") or {}).get("search") or {}).get("pit") or {}
    return str(pit.get("n", "")) == "auto"


def resolve_n(cfg: DictConfig) -> int | None:
    """`pit.n: auto` -> N ottimo del probe per board, precisione e backend di questa config,
    sul compute target `strategy.n_source` (default: quello della config). Se il probe manca
    viene lanciato; in dry-run None."""
    from .stages.probe import ensure_n

    n, _ = ensure_n(cfg, cfg.strategy.get("n_source"))
    return n


def finetune_key(cfg: DictConfig) -> str:
    return _sha({"search_key": search_key(cfg), "finetune": _group(cfg, "finetune")})[:8]


def weights_key(cfg: DictConfig) -> str:
    """Chiave dei pesi che vanno all'export: l'adapt_key, o con `adapt=none` quella a monte."""
    return adapt_key(cfg) if adapt_enabled(cfg) else pre_adapt(cfg)[2]


def model_label(model: str, strategy: str | None) -> str:
    """Nome del modello negli export: `yolo26n` per la baseline, `yolo26n-pit_duccio` per le
    strategie. Sta qui perche' `tools.artifacts prune` lo ricostruisce dai risultati."""
    return model if strategy in (None, "baseline") else f"{model}-{strategy}"


def compose_export_key(model: str, tkey: str, quantization: str, backend: str,
                       arch: str, sdk_version: str | None = None) -> str:
    """Composizione dell'export_key da parti gia' note.

    Sta qui e non duplicata altrove perche' `tools.artifacts prune` deve
    ricostruire la stessa stringa dai risultati per sapere quali artefatti
    sono ancora referenziati: se le due formule divergessero, prune
    cancellerebbe engine e .axm che costano ore di compilazione sulla board.
    """
    key = f"{model}_{tkey}_{quantization}_{backend}_{arch}"
    if backend == "axelera" and sdk_version:
        # Un aggiornamento dell'SDK invalida i .axm compilati in precedenza,
        # che vengono rifiutati al caricamento.
        key += f"_sdk{sdk_version}"
    return key


def export_key(cfg: DictConfig) -> str:
    sdk = None
    if cfg.backend.name == "axelera":
        sdk = cfg.backend.build.sdk_version
    return compose_export_key(
        model_label(cfg.model.name, strategy_name(cfg)), weights_key(cfg),
        cfg.quantization.name, cfg.backend.name, cfg.hardware.arch, sdk,
    )


# Argomenti di quantizzazione che valgono solo per i backend che usano il QDQ di onnxruntime:
# per gli altri restano fuori dalla chiave della cella, che non cambia aggiungendoli.
ONNXRUNTIME_ONLY_ARGS = ("quantize_op_types", "quantize_exclude_op_types", "quantize_exclude_regex",
                         "calibration_method", "calibration_options")
QDQ_BACKENDS = ("onnxruntime", "openvino_qdq")


def cell_key(cfg: DictConfig) -> str:
    payload = {k: _group(cfg, k) for k in cfg.cell_key_fields}
    ort_args = (payload.get("quantization") or {}).get("backend_args", {}).get("onnxruntime")
    if isinstance(ort_args, dict) and cfg.backend.name not in QDQ_BACKENDS:
        for k in ONNXRUNTIME_ONLY_ARGS:
            ort_args.pop(k, None)
    # Il gruppo `dataset` contiene solo path e nomi delle classi: due dataset
    # diversi allo stesso path darebbero la stessa chiave, e al rilancio la
    # cella verrebbe saltata come gia' completata pur misurando un modello
    # diverso. Il training_key porta dentro l'hash dei dati.
    payload["training_key"] = training_key(cfg)
    payload["weights_key"] = weights_key(cfg)
    payload["strategy"] = _group(cfg, "strategy")
    return _sha(payload)[:12]


def training_slug(cfg: DictConfig) -> str:
    """Slug leggibile davanti, hash corto come suffisso disambiguante."""
    return f"{cfg.model.name}_{cfg.pretrain_dataset.name}_e{cfg.train.epochs}"


def weights_dir(cfg: DictConfig) -> Path:
    return (
        Path(cfg.artifacts_dir)
        / "weights"
        / f"{training_slug(cfg)}_{training_key(cfg)}"
    )


def adapt_dir(cfg: DictConfig) -> Path:
    label = model_label(cfg.model.name, strategy_name(cfg))
    return (Path(cfg.artifacts_dir) / "adapt"
            / f"{label}_{cfg.dataset.name}_{adapt_key(cfg)}")


def search_dir(cfg: DictConfig) -> Path:
    return (Path(cfg.artifacts_dir) / "search" / f"{cfg.model.name}_{cfg.pretrain_dataset.name}"
            f"_{strategy_name(cfg)}_{search_key(cfg)}")


def finetune_dir(cfg: DictConfig) -> Path:
    return (Path(cfg.artifacts_dir) / "finetune" / f"{cfg.model.name}_{cfg.pretrain_dataset.name}"
            f"_{strategy_name(cfg)}_{finetune_key(cfg)}")


def find_by_key(artifacts_dir: str | Path, kind: str, field: str, key: str) -> Path | None:
    """Cartella di artifacts/<kind>/ il cui meta.json ha <field> == key."""
    root = Path(artifacts_dir) / kind
    if not root.is_dir():
        return None
    for d in sorted(root.iterdir()):
        meta = read_json(d / "meta.json")
        if meta and meta.get(field) == key:
            return d
    return None


def final_weights(cfg: DictConfig) -> tuple[Path | None, str]:
    """(pesi che vanno all'export, chiave): best.pt dell'adattamento; con `adapt=none` quello
    del training (baseline) o del fine-tuning (strategie)."""
    kind, field, key = (("adapt", "adapt_key", adapt_key(cfg)) if adapt_enabled(cfg)
                        else pre_adapt(cfg))
    d = find_by_key(cfg.artifacts_dir, kind, field, key)
    return (d / "best.pt" if d else None), key


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

    Il lock non e' zelo: `run.py` consiglia il launcher joblib per il training,
    e due job che leggono l'indice vuoto e lo riscrivono si cancellano a
    vicenda l'entry. La scrittura atomica protegge dal file troncato, non
    dall'aggiornamento perso.
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with _index_lock(index_path):
        _update_index_locked(index_path, key, meta)


@contextmanager
def _index_lock(index_path: Path):
    lock = index_path.with_suffix(index_path.suffix + ".lock")
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _update_index_locked(index_path: Path, key: str, meta: dict) -> None:
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
