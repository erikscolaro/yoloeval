"""Stadio `probe` — quale granularita' di canali (N) premia questo hardware.

Per ogni compute target della board misura la latenza di reti dummy (src/probe/dummy.py) al
variare dei canali C, con il tool nativo del backend e con lo stesso tuning della board del
benchmark. Dalla curva latenza(C) ricava N (src/probe/analysis.py): e' il valore che la
strategia PIT usa con `pit.n: auto`.

Output in results/probe/:
  <probe_id>.jsonl         una misura per riga (forma, C, MAC, latenza, throughput)
  <probe_id>.summary.json  N ottimo per forma e globale, test di ogni N, rumore
  <probe_id>.png           latenza e throughput in funzione di C (con stage.plot: true)

Oggi supporta i backend che misurano un ONNX portabile (onnxruntime, onnxruntime_py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

from omegaconf import OmegaConf, read_write

from ..backends import get_backend
from ..env import collect_versions
from ..jsonio import atomic_write_json, read_json
from ..measure.board import board_controller, tuned
from ..measure.compute import compute_targets, resolve_compute, resolve_freq
from ..probe import analysis, dummy
from ..remote.connection import connection
from ..remote.sync import ensure_support_files, sync_artifact
from ..schema import order_index
from ..timing import PhaseTimer, now_iso
from .benchmark import _effective_cores, _ensure_profile, _wrap_command

log = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("onnxruntime", "onnxruntime_py")
SCHEMA = "probe/1"


def probe_dir(cfg) -> Path:
    return Path(cfg.results_dir) / "probe"


def probe_id(cfg) -> str:
    """Board, profilo, compute target, precisione, backend e sweep: niente modello."""
    payload = {
        "board": cfg.hardware.board, "freq_target": cfg.freq_target,
        "compute_target": cfg.compute_target, "quantization": cfg.quantization.name,
        "backend": cfg.backend.name,
        "sweep": OmegaConf.to_container(cfg.stage.sweep, resolve=True),
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
    return (f"{cfg.hardware.board}_{cfg.compute_target}_{cfg.quantization.name}_"
            f"{cfg.backend.name}_{h}")


def shapes(cfg) -> list[dummy.Shape]:
    return [dummy.Shape(kernel=int(s.kernel), hw=int(s.hw), depthwise=bool(s.depthwise),
                        layers=int(s.layers)) for s in cfg.stage.sweep.shapes]


def run_probe(cfg) -> list[Path]:
    if cfg.backend.name not in SUPPORTED_BACKENDS:
        raise ValueError(f"il probe supporta per ora solo backend={SUPPORTED_BACKENDS}: "
                         f"{cfg.backend.name} compila sulla board, da aggiungere")
    written = []
    for target in compute_targets(cfg):
        with read_write(cfg):
            cfg.compute_target = target
        resolve_freq(cfg)
        ct = resolve_compute(cfg)
        if ct.get("device") == "metis":
            log.info("compute_target=%s (acceleratore): probe non supportato, salto", target)
            continue
        pid = probe_id(cfg)
        summary_path = probe_dir(cfg) / f"{pid}.summary.json"
        if summary_path.exists() and not cfg.force:
            log.info("probe gia' fatto: %s", pid)
            written.append(summary_path)
            continue
        if cfg.dry_run:
            sw = cfg.stage.sweep
            n_meas = len(shapes(cfg)) * len(range(int(sw.c_min), int(sw.c_max) + 1,
                                                   int(sw.c_step)))
            log.info("[dry-run] probe %s: %d misure", pid, n_meas)
            continue
        written.append(_run_one(cfg, pid, summary_path))
    return written


def _run_one(cfg, pid: str, summary_path: Path) -> Path:
    sw = cfg.stage.sweep
    timer = PhaseTimer(device=cfg.hardware.board)
    backend = get_backend(cfg.backend.name, cfg)
    precision = cfg.quantization.precision
    int8_args = dict(cfg.quantization.get("backend_args", {}).get("onnxruntime", {}) or {})
    onnx_dir = Path(cfg.artifacts_dir) / "probe" / "onnx"
    cs = list(range(int(sw.c_min), int(sw.c_max) + 1, int(sw.c_step)))
    plan = [(s, c) for s in shapes(cfg) for c in cs]

    with timer.phase("setup"):
        files = {(s, c): dummy.artifact(s, c, precision, onnx_dir, int8_args) for s, c in plan}
    # ordine casuale: la deriva (termica, carico) si distribuisce su tutti i C invece di
    # sembrare un effetto dei canali
    rng = random.Random(int(sw.get("seed", 0)))
    order = plan[:]
    rng.shuffle(order)
    # la stessa misura ripetuta all'inizio e alla fine stima il rumore e la deriva
    ref = plan[len(cs) - 1]
    order = [ref] + order + [ref]

    out_dir = probe_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{pid}.jsonl"
    records = []
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
                for i, (s, c) in enumerate(order):
                    remote = sync_artifact(live, cfg, files[(s, c)], subdir="probe",
                                           key=pid)
                    cmd = _wrap_command(live, cfg, backend.build_cmd(cfg, Path(remote)),
                                        ct, n_cores)
                    r = live.run(cmd, hide=True, warn=True)
                    if r.failed:
                        raise RuntimeError(f"misura fallita ({s.name}, C={c}): "
                                           f"{(r.stderr or r.stdout or '')[-1000:]}")
                    lat = backend.parse(r.stdout)
                    k = dummy.costs(s, c)
                    med = lat.median_ms or lat.mean_ms
                    records.append({
                        "schema": SCHEMA, "probe_id": pid, "board": cfg.hardware.board,
                        "freq_target": cfg.freq_target, "compute_target": cfg.compute_target,
                        "n_cores": n_cores, "quantization": cfg.quantization.name,
                        "backend": cfg.backend.name, "shape": s.name, "c": c, **k,
                        "median_ms": med, "mean_ms": lat.mean_ms, "p90_ms": lat.p90_ms,
                        "p99_ms": lat.p99_ms, "gmacs_per_s": k["macs"] / (med * 1e-3) / 1e9,
                        "repeat": i in (0, len(order) - 1), "sequence": i,
                        "timestamp": now_iso(), "order_index": order_index(),
                    })
                state = {"temp_start_c": applied.get("temp_start_c"),
                         "temp_end_c": bc.read_temp(live), "throttled": bc.read_throttle(live),
                         "freq_actual_khz": bc.read_freq(live)}
        finally:
            if live is not conn:
                live.close()

    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    summary = summarize(records, cfg, pid)
    summary.update(env=env, runtime_state=state, timing=timer.block())
    atomic_write_json(summary_path, summary)
    if cfg.stage.get("plot"):
        try:
            plot(records, summary, out_dir / f"{pid}.png")
        except Exception as e:  # noqa: BLE001 - il grafico non deve rompere il probe
            log.warning("grafico del probe non creato: %s", e)
    log.info("probe %s: N ottimo %s (per forma: %s, rumore %.1f%%)", pid, summary["n_opt"],
             {k: v["n_opt"] for k, v in summary["per_shape"].items()},
             100 * summary["noise_rel"])
    return summary_path


def summarize(records: list[dict], cfg, pid: str) -> dict:
    """N ottimo per forma e globale. Tolleranza = max(stage.sweep.tolerance, 3 x rumore)."""
    rep = [r["median_ms"] for r in records if r["repeat"]]
    noise = abs(rep[0] - rep[-1]) / min(rep) if len(rep) == 2 else 0.0
    tol = max(float(cfg.stage.sweep.tolerance), 3 * noise)
    given = cfg.stage.sweep.get("n_candidates")
    per_shape = {}
    for name in dict.fromkeys(r["shape"] for r in records):
        lat = {r["c"]: r["median_ms"] for r in records if r["shape"] == name and not r["repeat"]}
        res = analysis.best_n(lat, tol, list(given) if given else None)
        res["tested"] = {str(k): v for k, v in res["tested"].items()}
        per_shape[name] = res
    n_opt = analysis.global_n(per_shape)
    return {
        "schema": SCHEMA, "probe_id": pid, "board": cfg.hardware.board,
        "freq_target": cfg.freq_target, "compute_target": cfg.compute_target,
        "quantization": cfg.quantization.name, "backend": cfg.backend.name,
        "sweep": OmegaConf.to_container(cfg.stage.sweep, resolve=True),
        "noise_rel": noise, "tolerance": tol, "per_shape": per_shape, "n_opt": n_opt,
        "created_at": now_iso(),
    }


def plot(records: list[dict], summary: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["per_shape"])
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 7), squeeze=False)
    for j, name in enumerate(names):
        rows = sorted((r for r in records if r["shape"] == name and not r["repeat"]),
                      key=lambda r: r["c"])
        cs = [r["c"] for r in rows]
        n = summary["per_shape"][name]["n_opt"]
        for i, (key, label) in enumerate((("median_ms", "latency [ms]"),
                                          ("gmacs_per_s", "throughput [GMAC/s]"))):
            a = axes[i][j]
            a.plot(cs, [r[key] for r in rows], marker=".", linewidth=1)
            for m in range(-(-cs[0] // n) * n, cs[-1] + 1, n):
                a.axvline(m, color="grey", alpha=.25, linewidth=.8)
            a.set_xlabel("channels C")
            a.set_ylabel(label)
            a.grid(alpha=.3)
            if i == 0:
                a.set_title(f"{name}: N = {n}" +
                            (" (at least)" if summary["per_shape"][name]["at_least"] else ""))
    fig.suptitle(f"{summary['board']} {summary['compute_target']} {summary['quantization']} "
                 f"{summary['backend']} — N = {summary['n_opt']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def find_n(cfg, source: str | None = None) -> tuple[int, Path]:
    """N ottimo dal probe per la board, il compute target e la precisione di questa config.

    source: il compute target da cui prenderlo; default `stage`/strategia -> il compute
    target della config, o `n_source` della strategia (cpu_1 su CPU: il caso peggiore)."""
    target = source or cfg.compute_target
    if target is None:
        raise ValueError("pit.n: auto richiede un compute target (strategy.search.n_source o "
                         "compute_target=...) per scegliere quale probe usare")
    root = Path(cfg.results_dir) / "probe"
    found = []
    for p in sorted(root.glob("*.summary.json")) if root.is_dir() else []:
        s = read_json(p) or {}
        if (s.get("board"), s.get("compute_target"), s.get("quantization"),
                s.get("backend")) == (cfg.hardware.board, target, cfg.quantization.name,
                                      cfg.backend.name):
            found.append((s.get("created_at") or "", s, p))
    if not found:
        raise FileNotFoundError(
            f"nessun probe per board={cfg.hardware.board} compute_target={target} "
            f"quantization={cfg.quantization.name} backend={cfg.backend.name}: eseguire prima "
            f"`stage=probe` con questi valori")
    _, s, p = max(found, key=lambda t: t[0])       # il piu' recente
    return int(s["n_opt"]), p
