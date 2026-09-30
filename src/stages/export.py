"""Stadio `export` — dispatch al backend, cache, validazione.

Non implementa l'export: sceglie il backend, gli dice dove trovare i pesi e
dove scrivere, e si occupa di quello che vale per tutti — cache, sincronizza-
zione verso la board, validazione numerica, ispezione della testa, `meta.json`.

Nessun artefatto quantizzato entra nel benchmark senza essere passato di qui.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import OmegaConf

from ..backends import get_backend
from ..cache import export_dir as export_dir_for
from ..cache import (export_key, final_weights, is_baseline, strategy_name, training_key,
                     weights_key)
from ..env import collect_versions
from ..errors import MissingWeights
from ..jsonio import atomic_write_json
from ..remote.connection import LocalConnection, connection, is_remote
from ..remote.provision import ensure_env
from ..remote.sync import ensure_dataset, sync_artifact
from ..timing import PhaseTimer
from ..validation.graph import e2e_fallback_reason

log = logging.getLogger(__name__)


def run_export(cfg) -> Path:
    tkey = training_key(cfg)
    # baseline: best.pt del training; strategie (pit): best.pt del fine-tuning
    src_pt, wkey = final_weights(cfg)
    if src_pt is None:
        prima = ("`stage=train`" if is_baseline(cfg)
                 else "`stage=train`, `stage=search` e `stage=finetune`")
        raise MissingWeights(
            f"nessun peso per {cfg.model.name} con strategy={strategy_name(cfg)} "
            f"(chiave {wkey}): eseguire prima {prima}"
        )

    ekey = export_key(cfg)
    dst_dir = export_dir_for(cfg)
    if (dst_dir / "meta.json").exists() and not cfg.force_reexport:
        log.info("export gia' in cache: %s", ekey)
        return dst_dir

    if cfg.dry_run:
        log.info("[dry-run] export %s", ekey)
        return dst_dir

    backend = get_backend(cfg.backend.name, cfg)
    dst_dir.mkdir(parents=True, exist_ok=True)
    timer = PhaseTimer(device=cfg.hardware.board if backend.builds_on_target
                       else "workstation")

    if backend.builds_on_target:
        # L'artefatto e' legato all'hardware: si compila dove verra' usato, e
        # anche la validazione numerica deve girare li' — un engine TensorRT
        # sulla workstation non e' nemmeno caricabile.
        with connection(cfg) as conn:
            with timer.phase("setup"):
                ensure_env(conn, cfg)
                if cfg.quantization.get("calibration") or cfg.backend.name == "axelera":
                    ensure_dataset(conn, cfg)  # serve per la calibrazione INT8
                portable = backend.prepare(cfg, src_pt, dst_dir)
                remote_src = sync_artifact(conn, cfg, portable,
                                           subdir="artifacts", key=ekey)
            with timer.phase("build"):
                artifact = backend.export(conn, cfg, Path(remote_src), dst_dir)
            with timer.phase("validation"):
                reference = _reference_for(conn, cfg, src_pt, backend)
                validation = backend.validate(conn, cfg, artifact, reference)
                info = backend.inspect(artifact)
    else:
        # ONNX e IR sono portabili: export e validazione di primo livello
        # (artefatto contro PyTorch) restano sulla workstation, anche quando la
        # cella di destinazione e' una board.
        local = LocalConnection()
        with timer.phase("build"):
            artifact = backend.export(local, cfg, src_pt, dst_dir)
        with timer.phase("validation"):
            validation = backend.validate(local, cfg, artifact, src_pt)
            info = backend.inspect(artifact)

    complexity, model_info = _complexity_and_info(cfg, src_pt, wkey)

    env = collect_versions(project_root=cfg.project_root)
    fallback = e2e_fallback_reason(cfg, env)
    validation = dict(validation)
    validation["requested_e2e"] = True
    validation["actual_e2e"] = info.get("actual_e2e")
    if fallback:
        validation["e2e_fallback_reason"] = fallback
    if info.get("actual_e2e") is False:
        log.warning(
            "artefatto %s esportato con testa %s (e2e richiesto): %s",
            ekey, info.get("head"), fallback or "motivo non fra quelli noti",
        )
    if validation.get("status") == "degraded":
        log.error("validazione degradata per %s: %s", ekey,
                  validation.get("reason"))

    meta = {
        "export_key": ekey,
        "training_key": tkey,
        "weights_key": wkey,
        "strategy": strategy_name(cfg),
        "artifact": {
            "path": str(artifact),
            "name": Path(str(artifact)).name,
            "format": cfg.backend.export_format,
            "on_target": backend.builds_on_target,
            "board": cfg.hardware.board if backend.builds_on_target else None,
        },
        "validation": validation,
        "inspection": info,
        "complexity": complexity,
        "model_info": model_info,
        "timing": timer.block(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "env": env,
    }
    atomic_write_json(dst_dir / "meta.json", meta)
    log.info("export completato: %s (%s, validazione %s)",
             ekey, info.get("head"), validation.get("status"))
    return dst_dir


def _complexity_and_info(cfg, src_pt: Path, wkey: str) -> tuple[dict | None, dict]:
    """MAC e byte per il roofline (src/measure/complexity.py) e, per i modelli potati, la N
    usata dalla ricerca. Mai bloccante: senza, il roofline salta solo questo modello."""
    from ..measure.complexity import ensure_complexity, with_precision

    info = {"strategy": strategy_name(cfg), "pit_n": None}
    if not is_baseline(cfg):
        try:
            from ultralytics import YOLO

            meta = getattr(YOLO(str(src_pt)).model, "pit_meta", {}) or {}
            info["pit_n"] = meta.get("n")
        except Exception as exc:  # noqa: BLE001
            log.warning("N della ricerca non letta da %s: %s", src_pt, exc)
    try:
        cx = with_precision(ensure_complexity(cfg, src_pt, wkey), cfg.quantization.precision)
    except Exception as exc:  # noqa: BLE001
        log.warning("complessita' non calcolata per %s: %s", wkey, exc)
        cx = None
    return cx, info


def _reference_for(conn, cfg, src_pt: Path, backend):
    """Il riferimento FP32 con cui confrontare l'artefatto.

    Sulla board il `.pt` e' gia' stato sincronizzato per la build: si riusa
    quello, invece di ricopiarlo.
    """
    if not backend.builds_on_target or not is_remote(cfg):
        return src_pt
    # Sotto la chiave dei pesi: due modelli diversi hanno entrambi un `best.pt`,
    # e in una cartella piatta il secondo export sovrascriverebbe il primo. Per la
    # baseline e' il training_key; un modello potato ha lo stesso training_key della
    # sua baseline, quindi serve la chiave del fine-tuning.
    return sync_artifact(conn, cfg, src_pt, subdir="weights",
                         key=weights_key(cfg))
