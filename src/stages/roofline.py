"""Stadio `roofline` — i due tetti del roofline model, misurati sulla board.

Per ogni compute target della board:
- picco di calcolo (GMAC/s) alla precisione della config: la migliore fra alcune reti dummy
  ad alta intensita' aritmetica (conv 3x3 larghe, lineari grandi);
- banda di memoria (GB/s): un tensore molto piu' grande delle cache per uno scalare, in fp32
  (i byte al secondo non dipendono dalla precisione).

Sono tetti EMPIRICI: quanto il runtime riesce a ottenere su quel device, con lo stesso tool
e lo stesso tuning del benchmark. I tetti da datasheet, se dichiarati in
`hardware.peaks.<compute_target>`, si aggiungono nel grafico di tools.roofline.

Output: results/roofline/<board>_<compute>_<precisione>_<backend>.json
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import OmegaConf, read_write

from ..backends import get_backend
from ..env import collect_versions
from ..jsonio import atomic_write_json
from ..measure.board import board_controller, tuned
from ..measure.compute import compute_targets, resolve_compute, resolve_freq
from ..probe import dummy
from ..remote.connection import connection
from ..remote.sync import ensure_support_files
from ..timing import PhaseTimer, now_iso
from .benchmark import _effective_cores, _ensure_profile
from .probe import SUPPORTED_BACKENDS, measure_file

log = logging.getLogger(__name__)


def roofline_path(cfg) -> Path:
    return (Path(cfg.results_dir) / "roofline" /
            f"{cfg.hardware.board}_{cfg.compute_target}_{cfg.quantization.name}_"
            f"{cfg.backend.name}.json")


def run_roofline(cfg) -> list[Path]:
    if cfg.backend.name not in SUPPORTED_BACKENDS:
        raise ValueError(f"il roofline supporta per ora solo backend={SUPPORTED_BACKENDS}")
    written = []
    for target in compute_targets(cfg):
        with read_write(cfg):
            cfg.compute_target = target
        resolve_freq(cfg)
        if resolve_compute(cfg).get("device") == "metis":
            log.info("compute_target=%s (acceleratore): roofline non supportato, salto", target)
            continue
        path = roofline_path(cfg)
        if path.exists() and not cfg.force:
            log.info("roofline gia' misurato: %s", path.name)
            written.append(path)
            continue
        if cfg.dry_run:
            log.info("[dry-run] roofline %s", path.name)
            continue
        written.append(_measure(cfg, path))
    return written


def _measure(cfg, path: Path) -> Path:
    st = cfg.stage
    timer = PhaseTimer(device=cfg.hardware.board)
    precision = cfg.quantization.precision
    int8_args = dict(cfg.quantization.get("backend_args", {}).get("onnxruntime", {}) or {})
    onnx_dir = Path(cfg.artifacts_dir) / "roofline" / "onnx"
    with read_write(cfg):                       # meno iterazioni: le reti sono grandi
        cfg.backend.benchmark.iters = int(st.iters)
        if "warmup_iters" in cfg.backend.benchmark and cfg.backend.benchmark.warmup_iters:
            cfg.backend.benchmark.warmup_iters = int(st.get("warmup_iters", 5))
    backend = get_backend(cfg.backend.name, cfg)
    shapes = [(dummy.Shape(kind=str(s.get("kind", "conv")), kernel=int(s.get("kernel", 1)),
                           hw=int(s.hw), layers=int(s.layers)), int(s.c))
              for s in st.compute_shapes]
    with timer.phase("setup"):
        files = {(sh, c): dummy.artifact(sh, c, precision, onnx_dir, int8_args)
                 for sh, c in shapes}
        bw_file = dummy.bandwidth_artifact(int(st.bandwidth.elements), onnx_dir)

    key = path.stem
    per_shape = {}
    with connection(cfg) as conn:
        env = collect_versions(conn, cfg, cfg.project_root)
        bc = board_controller(cfg)
        live = _ensure_profile(conn, cfg, bc)
        try:
            with timer.phase("setup"):
                ensure_support_files(live, cfg)
            with tuned(live, cfg, bc) as applied, timer.phase("compute"):
                ct = resolve_compute(cfg)
                n_cores = _effective_cores(live, cfg, bc, ct)
                for (sh, c), f in files.items():
                    lat = measure_file(live, cfg, backend, ct, n_cores, f, subdir="roofline",
                                       key=key, what=f"{sh.name}, C={c}")
                    ms = lat.median_ms or lat.mean_ms
                    macs = dummy.costs(sh, c)["macs"]
                    per_shape[f"{sh.name}_c{c}"] = {"macs": macs, "median_ms": ms,
                                                    "gmacs_per_s": macs / (ms * 1e-3) / 1e9}
                lat = measure_file(live, cfg, backend, ct, n_cores, bw_file,
                                   subdir="roofline", key=key, what="bandwidth")
                bw_ms = lat.median_ms or lat.mean_ms
                state = {"temp_start_c": applied.get("temp_start_c"),
                         "temp_end_c": bc.read_temp(live), "throttled": bc.read_throttle(live),
                         "freq_actual_khz": bc.read_freq(live)}
        finally:
            if live is not conn:
                live.close()

    best = max(per_shape, key=lambda k: per_shape[k]["gmacs_per_s"])
    bw_bytes = dummy.BANDWIDTH_BYTES_PER_ELEMENT * int(st.bandwidth.elements)
    datasheet = (cfg.hardware.get("peaks") or {}).get(cfg.compute_target)
    out = {
        "schema": "roofline/1", "board": cfg.hardware.board, "freq_target": cfg.freq_target,
        "compute_target": cfg.compute_target, "n_cores": n_cores,
        "quantization": cfg.quantization.name, "precision": precision,
        "backend": cfg.backend.name,
        "peak_gmacs": per_shape[best]["gmacs_per_s"], "peak_from": best,
        "per_shape": per_shape,
        "bandwidth_gbs": bw_bytes / (bw_ms * 1e-3) / 1e9, "bandwidth_bytes": bw_bytes,
        "ridge_mac_per_byte": None,
        "datasheet": OmegaConf.to_container(datasheet, resolve=True) if datasheet else None,
        "runtime_state": state, "env": env, "timing": timer.block(), "created_at": now_iso(),
    }
    out["ridge_mac_per_byte"] = out["peak_gmacs"] / out["bandwidth_gbs"]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, out)
    log.info("roofline %s: picco %.1f GMAC/s (%s), banda %.1f GB/s, ridge %.1f MAC/byte",
             path.stem, out["peak_gmacs"], best, out["bandwidth_gbs"],
             out["ridge_mac_per_byte"])
    return path
