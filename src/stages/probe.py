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

import copy
import hashlib
import json
import logging
import random
import time
from pathlib import Path

from omegaconf import OmegaConf, open_dict, read_write

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
    return [dummy.Shape(kernel=int(s.get("kernel", 1)), hw=int(s.hw),
                        depthwise=bool(s.get("depthwise", False)), layers=int(s.layers),
                        kind=str(s.get("kind", "conv")), in_global=bool(s.get("in_global", True)))
            for s in cfg.stage.sweep.shapes]


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
                                                   int(sw.c_step))) * int(sw.get("repeats", 3))
            log.info("[dry-run] probe %s: %d misure", pid, n_meas)
            continue
        written.append(_run_one(cfg, pid, summary_path))
    return written


def _run_one(cfg, pid: str, summary_path: Path) -> Path:
    sw = cfg.stage.sweep
    timer = PhaseTimer(device=cfg.hardware.board)
    with read_write(cfg):                # iterazioni del probe, non quelle del benchmark
        if cfg.stage.get("iters"):
            cfg.backend.benchmark.iters = int(cfg.stage.iters)
        if cfg.stage.get("warmup_iters") and cfg.backend.benchmark.get("warmup_iters"):
            cfg.backend.benchmark.warmup_iters = int(cfg.stage.warmup_iters)
    backend = get_backend(cfg.backend.name, cfg)
    precision = cfg.quantization.precision
    int8_args = dict(cfg.quantization.get("backend_args", {}).get("onnxruntime", {}) or {})
    onnx_dir = Path(cfg.artifacts_dir) / "probe" / "onnx"
    cs = list(range(int(sw.c_min), int(sw.c_max) + 1, int(sw.c_step)))
    plan = [(s, c) for s in shapes(cfg) for c in cs]

    with timer.phase("setup"):
        files = {(s, c): dummy.artifact(s, c, precision, onnx_dir, int8_args) for s, c in plan}
    # `repeats` passate, ognuna in ordine casuale: la deriva (termica, carico) si distribuisce
    # su tutti i C invece di sembrare un effetto dei canali, e la dispersione fra le passate
    # stima il rumore
    rng = random.Random(int(sw.get("seed", 0)))
    order = []
    for rep in range(int(sw.get("repeats", 3))):
        passata = plan[:]
        rng.shuffle(passata)
        order += [(rep, s, c) for s, c in passata]

    out_dir = probe_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{pid}.jsonl"
    records = []
    log.info("probe %s: %d misure (%d forme x %d canali x %d passate), %s %s @ %s",
             pid, len(order), len(shapes(cfg)), len(cs), int(sw.get("repeats", 3)),
             cfg.hardware.board, cfg.compute_target, cfg.freq_target)
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
                t0, every = time.monotonic(), max(1, len(order) // 20)
                for i, (rep, s, c) in enumerate(order):
                    if i and i % every == 0:      # avanzamento ogni 5%, con stima del resto
                        left = (time.monotonic() - t0) / i * (len(order) - i)
                        log.info("probe %s: %d/%d (%d%%), passata %d, ~%d min alla fine",
                                 pid, i, len(order), 100 * i // len(order), rep + 1,
                                 round(left / 60))
                    lat = measure_file(live, cfg, backend, ct, n_cores, files[(s, c)],
                                       subdir="probe", key=pid, what=f"{s.name}, C={c}")
                    k = dummy.costs(s, c)
                    med = lat.median_ms or lat.mean_ms
                    records.append({
                        "schema": SCHEMA, "probe_id": pid, "board": cfg.hardware.board,
                        "freq_target": cfg.freq_target, "compute_target": cfg.compute_target,
                        "n_cores": n_cores, "quantization": cfg.quantization.name,
                        "backend": cfg.backend.name, "shape": s.name, "c": c, **k,
                        "median_ms": med, "mean_ms": lat.mean_ms, "p90_ms": lat.p90_ms,
                        "p99_ms": lat.p99_ms, "gmacs_per_s": k["macs"] / (med * 1e-3) / 1e9,
                        "repeat": rep, "sequence": i,
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
            plot(records, summary, out_dir / f"{pid}.png",
                 xtick=int(cfg.stage.get("plot_xtick", 4)))
        except Exception as e:  # noqa: BLE001 - il grafico non deve rompere il probe
            log.warning("grafico del probe non creato: %s", e)
    log.info("probe %s: N ottimo %s (per forma: %s, rumore %.1f%%)", pid, summary["n_opt"],
             {k: v["n_opt"] for k, v in summary["per_shape"].items()},
             100 * summary["noise_rel"])
    return summary_path


def point_latency(records: list[dict]) -> dict:
    """{(forma, C): mediana delle passate}."""
    from statistics import median

    by = {}
    for r in records:
        by.setdefault((r["shape"], r["c"]), []).append(r["median_ms"])
    return {k: median(v) for k, v in by.items()}, by


def measure_file(live, cfg, backend, ct, n_cores, path: Path, subdir: str, key: str,
                 what: str = ""):
    """Copia un ONNX sulla board e lo misura con il tool del backend -> LatencyResult."""
    remote = sync_artifact(live, cfg, path, subdir=subdir, key=key)
    cmd = _wrap_command(live, cfg, backend.build_cmd(cfg, Path(remote)), ct, n_cores)
    r = live.run(cmd, hide=True, warn=True)
    if r.failed:
        raise RuntimeError(f"misura fallita ({what or path.name}): "
                           f"{(r.stderr or r.stdout or '')[-1000:]}")
    return backend.parse(r.stdout)


def summarize(records: list[dict], cfg, pid: str) -> dict:
    """N ottimo per forma e globale. Tolleranza = max(stage.sweep.tolerance, rumore), con il
    rumore = mediana della dispersione relativa (max - min) / mediana fra le passate."""
    from statistics import median

    sw = cfg.stage.sweep
    lat, by = point_latency(records)
    spreads = [(max(v) - min(v)) / median(v) for v in by.values() if len(v) > 1]
    noise = median(spreads) if spreads else 0.0
    tol = max(float(sw.tolerance), noise)
    given = sw.get("n_candidates")
    per_shape = {}
    in_global = {sh.name: sh.in_global for sh in shapes(cfg)}
    for name in dict.fromkeys(r["shape"] for r in records):
        pts = {c: v for (sh, c), v in lat.items() if sh == name}
        macs = {r["c"]: r["macs"] for r in records if r["shape"] == name}
        res = analysis.best_n(pts, macs, tol, float(sw.get("max_violations", 0.2)),
                              int(sw.get("window", 32)), list(given) if given else None)
        res["tested"] = {str(k): v for k, v in res["tested"].items()}
        res["in_global"] = in_global.get(name, True)
        per_shape[name] = res
    n_opt = analysis.global_n(per_shape)
    if n_opt is None:
        log.warning("probe %s non concluso: nessun N passa (misure troppo rumorose? prova "
                    "piu' repeats o un range di canali piu' ampio)", pid)
    return {
        "schema": SCHEMA, "probe_id": pid, "board": cfg.hardware.board,
        "freq_target": cfg.freq_target, "compute_target": cfg.compute_target,
        "quantization": cfg.quantization.name, "backend": cfg.backend.name,
        "sweep": OmegaConf.to_container(sw, resolve=True),
        "noise_rel": noise, "tolerance": tol, "per_shape": per_shape, "n_opt": n_opt,
        "created_at": now_iso(),
    }


def plot(records: list[dict], summary: dict, path: Path, xtick: int = 4) -> Path:
    """Latenza e throughput in funzione di C; tacche dell'asse x ogni `xtick` canali."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["per_shape"])
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 7), squeeze=False)
    for j, name in enumerate(names):
        lat, _ = point_latency([r for r in records if r["shape"] == name])
        macs = {r["c"]: r["macs"] for r in records if r["shape"] == name}
        cs = sorted(c for _, c in lat)
        ms = [lat[(name, c)] for c in cs]
        series = (ms, [macs[c] / (m * 1e-3) / 1e9 for c, m in zip(cs, ms)])
        n = summary["per_shape"][name]["n_opt"]
        for i, label in enumerate(("latency [ms], median of the passes",
                                   "throughput [GMAC/s]")):
            a = axes[i][j]
            a.plot(cs, series[i], marker=".", linewidth=1)
            # linee ai multipli di N (non per N=1: sarebbero una per ogni C)
            for m in (range(-(-cs[0] // n) * n, cs[-1] + 1, n) if n and n > 1 else []):
                a.axvline(m, color="grey", alpha=.25, linewidth=.8)
            a.set_xlabel("neurons C" if name.startswith("linear") else "channels C")
            step = int(xtick or 4)
            ticks = list(range(-(-cs[0] // step) * step, cs[-1] + 1, step))
            a.set_xticks(ticks)
            a.tick_params(axis="x", labelsize=7 if len(ticks) > 20 else 9,
                          labelrotation=90 if len(ticks) > 20 else 0)
            a.set_ylabel(label)
            a.grid(alpha=.3)
            if i == 0:
                a.set_title(f"{name}: N = {n if n else 'not found'}" + (
                    "" if summary["per_shape"][name].get("in_global", True)
                    else " (informative)"))
    fig.suptitle(f"{summary['board']} {summary['compute_target']} {summary['quantization']} "
                 f"{summary['backend']} — N = {summary['n_opt']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def probe_backend(cfg) -> str:
    """Backend del probe da cui prendere N: `strategy.probe.backend`, default onnxruntime.
    Non quello della config: un export tensorrt int8 usa l'N del probe onnxruntime int8."""
    opts = (cfg.get("strategy") or {}).get("probe") or {}
    return str(opts.get("backend") or "onnxruntime")


def find_n(cfg, source: str | None = None) -> tuple[int, Path]:
    """N ottimo dal probe per la board, il compute target e la precisione di questa config,
    con il backend di probe_backend().

    source: il compute target da cui prenderlo; default `stage`/strategia -> il compute
    target della config, o `n_source` della strategia (cpu_1 su CPU: il caso peggiore)."""
    target = source or cfg.compute_target
    backend = probe_backend(cfg)
    if target is None:
        raise ValueError("pit.n: auto richiede un compute target (strategy.search.n_source o "
                         "compute_target=...) per scegliere quale probe usare")
    root = Path(cfg.results_dir) / "probe"
    found = []
    for p in sorted(root.glob("*.summary.json")) if root.is_dir() else []:
        s = read_json(p) or {}
        if (s.get("board"), s.get("compute_target"), s.get("quantization"),
                s.get("backend")) == (cfg.hardware.board, target, cfg.quantization.name,
                                      backend):
            found.append((s.get("created_at") or "", s, p))
    if not found:
        raise FileNotFoundError(
            f"nessun probe per board={cfg.hardware.board} compute_target={target} "
            f"quantization={cfg.quantization.name} backend={backend}: eseguire prima "
            f"`stage=probe` con questi valori")
    _, s, p = max(found, key=lambda t: t[0])       # il piu' recente
    if not s.get("n_opt"):
        raise ValueError(f"il probe {p.name} non e' concluso (nessun N passa): rifallo con "
                         f"piu' repeats o un range di canali piu' ampio")
    return int(s["n_opt"]), p


def ensure_n(cfg, source: str | None = None) -> tuple[int | None, Path | None]:
    """Come find_n, ma se il probe manca lo lancia (preset e profilo da `strategy.probe`) e
    poi rilegge N. In dry-run lo annuncia soltanto e restituisce (None, None)."""
    try:
        return find_n(cfg, source)
    except FileNotFoundError:
        pass
    pcfg = probe_cfg(cfg, source)
    what = (f"board={cfg.hardware.board} compute_target={pcfg.compute_target} "
            f"quantization={cfg.quantization.name} backend={pcfg.backend.name}")
    how = f"stage={pcfg.stage.preset} freq_target={pcfg.freq_target}"
    if cfg.dry_run:
        log.warning("[dry-run] nessun probe per %s: verrebbe lanciato ora (%s)", what, how)
        return None, None
    log.warning("nessun probe per %s: lo lancio ora (%s), poi si riparte da qui", what, how)
    run_probe(pcfg)
    n, path = find_n(cfg, source)
    log.warning("probe concluso: N=%d (%s)", n, path.name)
    return n, path


def probe_cfg(cfg, source: str | None = None):
    """Copia della config con lo stadio probe al posto di quello corrente.

    Preset: `strategy.probe.stage`, altrimenti probe_edge per la CPU delle board remote e probe
    per il resto. Profilo: `strategy.probe.freq_target`, altrimenti quello della config se la
    board lo dichiara, altrimenti il primo della board."""
    opts = (cfg.get("strategy") or {}).get("probe") or {}
    target = source or cfg.compute_target
    if target not in cfg.hardware.compute:
        raise ValueError(f"compute_target={target} non dichiarato da {cfg.hardware.board}")
    preset = opts.get("stage") or (
        "probe_edge" if "remote" in cfg.hardware
        and cfg.hardware.compute[target].get("device") == "cpu" else "probe")
    freq = opts.get("freq_target") or (
        cfg.freq_target if cfg.freq_target in cfg.hardware.freq else next(iter(cfg.hardware.freq)))
    stage = OmegaConf.load(Path(cfg.project_root) / "conf" / "stage" / f"{preset}.yaml")
    stage.preset = preset
    pcfg = copy.deepcopy(cfg)
    with read_write(pcfg), open_dict(pcfg):
        if cfg.backend.name != probe_backend(cfg):
            pcfg.backend = OmegaConf.load(
                Path(cfg.project_root) / "conf" / "backend" / f"{probe_backend(cfg)}.yaml")
        pcfg.stage = stage
        pcfg.compute_target = target
        pcfg.freq_target = freq
        pcfg.force = False
    return pcfg
