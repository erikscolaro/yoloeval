"""mAP — calcolata una volta per artefatto esportato, non per cella.

mAP@50 e mAP@50-95 non dipendono da profilo di potenza, numero di core o
frequenza: ripeterle su ogni riga della matrice costerebbe ore e produrrebbe
lo stesso numero. Il risultato viene salvato accanto all'export e riletto dalle
celle successive.

Per la mAP la soglia di confidence va tenuta **bassa** (`eval.map_conf`),
altrimenti la coda della curva precision-recall viene troncata e la metrica
scende artificialmente. E' un parametro diverso da quello usato per la latenza.
"""

from __future__ import annotations

import json
import logging
import shlex
import sys
from pathlib import Path

import yaml

from ..remote.connection import is_local_conn
from ..remote.sync import ensure_dataset

log = logging.getLogger(__name__)

MARKER = "YOLOBENCH_JSON "
HELPER = "scripts/remote/eval_map.py"


def _python_for(conn, cfg) -> str:
    if is_local_conn(conn):
        return shlex.quote(sys.executable)
    return cfg.hardware.remote.get("python") or "python3"


def _push_helper(conn, cfg, name: str = HELPER) -> str:
    """Copia lo script helper dove gira, e ritorna il path da invocare."""
    local = Path(cfg.project_root) / name
    if is_local_conn(conn):
        return str(local)
    remote = f"{cfg.hardware.remote.workdir}/tools/{Path(name).name}"
    conn.run(f"mkdir -p {Path(remote).parent}", hide=True)
    conn.put(str(local), remote)
    return remote


def remote_data_yaml(conn, cfg) -> str:
    """Data yaml valido sulla macchina dove gira la validazione.

    Il file originale porta path assoluti della workstation: sulla board il
    dataset vive sotto `workdir/data/<name>`, quindi la chiave `path` va
    riscritta. Senza, Ultralytics valuta su un dataset vuoto e riporta mAP 0
    senza lamentarsi.
    """
    if is_local_conn(conn):
        return str(cfg.dataset.yaml)

    remote_root = ensure_dataset(conn, cfg)
    spec = yaml.safe_load(Path(cfg.dataset.yaml).read_text(encoding="utf-8"))
    spec["path"] = remote_root
    dest = f"{remote_root}/.bench_data.yaml"
    body = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
    conn.run(f"cat > {shlex.quote(dest)} <<'YAMLEOF'\n{body}\nYAMLEOF", hide=True)
    return dest


def parse_marker(stdout: str) -> dict:
    """Estrae la riga JSON marcata dallo stdout dell'helper."""
    from ..errors import ParseError

    for line in reversed((stdout or "").splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise ParseError(
        "output dell'helper non riconoscibile: nessuna riga " + MARKER.strip()
    )


def compute_map(conn, cfg, model_path, device: str | None = None) -> dict:
    """Esegue la validazione dove vive l'artefatto e ritorna il blocco accuracy."""
    helper = _push_helper(conn, cfg)
    data = remote_data_yaml(conn, cfg)
    python = _python_for(conn, cfg)

    cmd = (
        f"{python} {shlex.quote(str(helper))} "
        f"--model {shlex.quote(str(model_path))} "
        f"--data {shlex.quote(str(data))} "
        f"--imgsz {cfg.model.imgsz} "
        f"--conf {cfg.eval.map_conf} --iou {cfg.eval.map_iou} "
        f"--max-det {cfg.eval.max_det}"
    )
    if device:
        cmd += f" --device {shlex.quote(device)}"

    log.info("calcolo mAP di %s", Path(str(model_path)).name)
    r = conn.run(cmd, hide=True, warn=True)
    if r.failed:
        raise RuntimeError(
            f"calcolo mAP fallito ({r.return_code}):\n{r.stderr or r.stdout}"
        )
    payload = parse_marker(r.stdout)
    payload["source"] = "ultralytics_val"
    payload["measured_on"] = cfg.hardware.board
    return payload
