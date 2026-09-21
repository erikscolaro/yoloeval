"""Verifica di correttezza dell'export — obbligatoria prima del benchmark.

Due livelli:

1. **Centrale (workstation):** output ONNX contro output PyTorch su un batch
   fisso. Intercetta errori di export.
2. **Sulla board:** output dell'artefatto hardware-specifico contro FP32.
   Intercetta errori di quantizzazione, che avviene in fase di build e dipende
   dall'hardware.

Le soglie vengono da `quantization.validation`: superarle non blocca l'export,
lo marca `degraded`. Nel report gli artefatti degradati compaiono in una
tabella di stato, cosi' un modello rotto e' visibile a colpo d'occhio invece di
nascondersi dentro un grafico di latenza.
"""

from __future__ import annotations

import logging
import shlex

from ..measure.accuracy import _push_helper, parse_marker
from ..remote.connection import is_local_conn
from ..remote.sync import ensure_dataset

log = logging.getLogger(__name__)

HELPER = "scripts/remote/compare_outputs.py"

#: quante immagini nel batch fisso di confronto
N_IMAGES = 8


def _images_dir(conn, cfg) -> str:
    """Directory di immagini usata per il confronto, sulla macchina giusta."""
    if is_local_conn(conn):
        return str(cfg.dataset.local_path)
    return ensure_dataset(conn, cfg)


def compare_with_reference(conn, cfg, artifact, ref, device: str | None = None) -> dict:
    """Blocco `validation` dello schema.

    FP32 e' il riferimento: non c'e' nulla da validare, e lo si dichiara invece
    di far finta di aver misurato qualcosa.
    """
    thresholds = cfg.quantization.get("validation")
    if not thresholds:
        return {
            "status": "ok",
            "reason": "fp32 e' il riferimento",
            "max_abs_diff_vs_fp32": 0.0,
        }

    import sys

    helper = _push_helper(conn, cfg, HELPER)
    python = (
        shlex.quote(sys.executable)
        if is_local_conn(conn)
        else (cfg.hardware.remote.get("python") or "python3")
    )
    cmd = (
        f"{python} {shlex.quote(str(helper))} "
        f"--ref {shlex.quote(str(ref))} "
        f"--target {shlex.quote(str(artifact))} "
        f"--images {shlex.quote(_images_dir(conn, cfg))} "
        f"--imgsz {cfg.model.imgsz} --n {N_IMAGES}"
    )
    if device:
        cmd += f" --device {shlex.quote(device)}"

    r = conn.run(cmd, hide=True, warn=True)
    if r.failed:
        # Il confronto non riuscito e' un fallimento della validazione, non un
        # via libera: l'artefatto resta non validato.
        log.error("confronto numerico fallito:\n%s", r.stderr or r.stdout)
        return {
            "status": "failed",
            "error": (r.stderr or r.stdout or "")[-2000:],
            "max_abs_diff_vs_fp32": None,
        }

    payload = parse_marker(r.stdout)
    return grade(payload, thresholds)


def grade(payload: dict, thresholds) -> dict:
    """Applica le soglie al risultato del confronto."""
    out = {
        "max_abs_diff_vs_fp32": payload.get("max_abs_diff_vs_fp32"),
        "mean_abs_diff_vs_fp32": payload.get("mean_abs_diff_vs_fp32"),
        "cosine_similarity": payload.get("cosine_similarity"),
        "n_images": payload.get("n_images"),
        "ref_shape": payload.get("ref_shape"),
        "target_shape": payload.get("target_shape"),
    }
    if payload.get("status") == "shape_mismatch":
        out["status"] = "degraded"
        out["reason"] = (
            f"forma diversa dal riferimento: {payload.get('ref_shape')} vs "
            f"{payload.get('target_shape')} (testa diversa, non solo "
            f"quantizzazione)"
        )
        return out

    limit = float(thresholds.get("max_abs_diff", float("inf")))
    diff = out["max_abs_diff_vs_fp32"]
    if diff is None:
        out["status"] = "failed"
        out["reason"] = "differenza non calcolabile"
    elif diff > limit:
        out["status"] = "degraded"
        out["reason"] = f"max_abs_diff {diff:.5f} > soglia {limit}"
    else:
        out["status"] = "ok"
    return out


def grade_accuracy(validation: dict, map50: float | None,
                   ref_map50: float | None, thresholds) -> dict:
    """Aggiunge `map50_delta` e declassa se il calo supera la tolleranza."""
    if map50 is None or ref_map50 is None or not thresholds:
        validation.setdefault("map50_delta", None)
        return validation
    delta = float(map50) - float(ref_map50)
    validation["map50_delta"] = round(delta, 5)
    tol = float(thresholds.get("map50_drop_tolerance", float("inf")))
    if -delta > tol:
        validation["status"] = "degraded"
        validation["reason"] = (
            f"{validation.get('reason', '')} | calo mAP50 {delta:.4f} "
            f"oltre la tolleranza {tol}"
        ).strip(" |")
    return validation
