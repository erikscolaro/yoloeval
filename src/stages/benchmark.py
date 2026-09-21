"""Stadio `benchmark` — orchestrazione della singola cella.

Questo modulo **non cronometra**: compone la command line del tool nativo del
vendor, la esegue, ne parsa l'output e raccoglie il contesto (frequenza
effettiva, core online, temperatura, throttling, energia). Il tempo lo misura
un eseguibile C++ scritto dal vendor, cosi' l'overhead del binding Python non
entra nella misura.

Separazione delle responsabilita':

* `benchmark.py` orchestra — trova artefatti, risolve il compute, raccoglie
  il contesto;
* il backend compone il comando e parsa l'output;
* `remote/` gestisce la connessione;
* `measure/` legge e tocca lo stato della board.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from omegaconf import read_write

from ..backends import get_backend
from ..cache import cell_key, export_key, find_export, locate_artifact, training_key
from ..env import collect_versions
from ..errors import BenchmarkFailed, MissingExport
from ..jsonio import atomic_write_json, atomic_write_text, read_json
from ..measure.accuracy import compute_map
from ..measure.board import board_controller, tuned
from ..measure.compute import (
    affinity_spec,
    compute_targets,
    resolve_compute,
    resolve_freq,
    thread_env,
)
from ..measure.power import energy_sampler
from ..mirror import mirror_result
from ..remote.connection import LocalConnection, connection, shell_env
from ..remote.sync import ensure_support_files, sync_artifact
from ..schema import (
    STATUS_FAILED,
    base_record,
    failed,
    latency_block,
    ok,
    skipped,
    validate_record,
)
from ..timing import PhaseTimer
from ..validation.compat import is_valid
from ..validation.numerical import grade_accuracy

log = logging.getLogger(__name__)

#: device Ultralytics per il calcolo della mAP, dal device del compute target
MAP_DEVICE = {"cuda": "cuda", "cpu": "cpu", "metis": None}


def run_benchmark(cfg) -> list[Path]:
    """Esegue le celle di questo job.

    `compute_target: null` significa "tutti quelli dichiarati dalla board":
    Hydra non puo' espandere un default che dipende dal file dell'hardware,
    quindi l'espansione avviene qui e produce una cella per target.
    """
    written = []
    for target in compute_targets(cfg):
        with read_write(cfg):
            cfg.compute_target = target

        # Risoluzione delle chiavi prima di ogni altra cosa: un profilo o un
        # compute target che la board non dichiara e' un errore immediato, non
        # una cella da saltare. `compute_target=cpu_8` su rpi5 e' un refuso, e
        # un refuso deve fermare lo sweep invece di riempire results/ di celle
        # saltate che in analisi sembrano combinazioni non supportate.
        resolve_freq(cfg)
        resolve_compute(cfg)

        cid = cell_key(cfg)
        dest = Path(cfg.results_dir) / f"{cid}.json"

        if cfg.dry_run:
            valid, reason = is_valid(cfg)
            log.info("[dry-run] %s %s/%s/%s %s/%s/%s -> %s", cid,
                     cfg.model.name, cfg.quantization.name, cfg.backend.name,
                     cfg.hardware.board, cfg.freq_target, target,
                     "ok" if valid else f"skip ({reason})")
            continue

        if _should_skip(cfg, cid, dest):
            continue
        written.append(run_cell(cfg, cid, dest))
    return written


def _should_skip(cfg, cid: str, dest: Path) -> bool:
    """Resume: `ok` e `skipped` si saltano, `failed` solo senza retry_failed."""
    if not dest.exists() or cfg.force:
        return False
    prev = read_json(dest)
    if prev is None:
        log.warning("%s illeggibile, la rifaccio", dest.name)
        return False
    status = prev.get("status")
    if status == "ok":
        log.info("%s gia' completata, salto", cid)
        return True
    if status == "skipped":
        log.info("%s non applicabile (%s), salto", cid, prev.get("reason"))
        return True
    if status == "failed" and not cfg.retry_failed:
        log.info("%s fallita in precedenza, salto (usa +retry_failed=true)", cid)
        return True
    log.info("%s fallita, riprovo", cid)
    return False


def run_cell(cfg, cid: str, dest: Path) -> Path:
    """Esegue una cella e scrive **sempre** il suo JSON.

    Il `finally` non e' un dettaglio: senza, una cella che esplode non lascia
    traccia e verrebbe ritentata alla cieca a ogni rilancio, e il tempo che ha
    occupato la macchina sparirebbe dalle machine hours.
    """
    timer = PhaseTimer(device=cfg.hardware.board)
    rec = base_record(cfg, cid, training_key=training_key(cfg))
    try:
        valid, reason = is_valid(cfg)
        if not valid:
            _skip_or_raise(cfg, rec, reason)
        else:
            with connection(cfg) as conn:
                valid, reason = is_valid(cfg, conn)      # dinamica: core online
                if not valid:
                    _skip_or_raise(cfg, rec, reason)
                else:
                    rec["env"] = collect_versions(conn, cfg, cfg.project_root)
                    bc = board_controller(cfg)
                    # Dopo un riavvio autorizzato questa e' una connessione
                    # nuova: riusare la vecchia significherebbe parlare con un
                    # ControlPath stantio (errore 4c della specifica).
                    live = _ensure_profile(conn, cfg, bc)
                    try:
                        with timer.phase("setup"):
                            ensure_support_files(live, cfg)
                            _prepare_board_for_cell(live, cfg)
                        with tuned(live, cfg, bc) as applied:
                            payload = execute_benchmark(live, cfg, dest, timer,
                                                        bc, applied)
                    finally:
                        if live is not conn:
                            live.close()
                    ok(rec, payload)
                    # Letti dopo il ripristino: se la cella lascia la macchina
                    # in uno stato diverso da come l'ha trovata, si vede qui
                    # invece che nelle latenze della cella successiva.
                    rec["runtime_state"].update({
                        "state_before": applied.get("state_before"),
                        "state_after_restore": applied.get("state_after_restore"),
                    })
    except _CellSkipped:
        raise
    except Exception as exc:  # noqa: BLE001 - la cella fallita e' un risultato
        log.exception("cella %s fallita", cid)
        failed(rec, exc)
    finally:
        rec["timing"] = timer.block()
        atomic_write_json(dest, rec)
        problems = validate_record(rec)
        if problems:
            log.warning("record %s non conforme: %s", cid, "; ".join(problems))
        mirror_result(cfg, rec)

    if rec["status"] == STATUS_FAILED and not cfg.continue_on_error:
        # Il record e' gia' su disco: si ferma lo sweep, non si perde la cella.
        raise BenchmarkFailed(f"{cid}: {rec['error']}")
    return dest


class _CellSkipped(Exception):
    """Cella non applicabile con `skip_invalid=false`: interrompe lo sweep."""


def _skip_or_raise(cfg, rec: dict, reason: str) -> None:
    """Di norma una cella non valida e' un record `skipped`, non un errore.

    Con `skip_invalid=false` invece si ferma: serve quando si vuole essere
    certi che la matrice richiesta sia interamente eseguibile, e scoprire
    subito un asse sbagliato invece di ritrovarsi meta' celle saltate.
    """
    skipped(rec, reason)
    if not cfg.skip_invalid:
        raise _CellSkipped(reason)


def _ensure_profile(conn, cfg, bc):
    """Verifica (e se serve applica) il profilo di potenza richiesto.

    Il cambio di profilo e' un'operazione di livello **superiore** allo sweep:
    alcune transizioni richiedono il riavvio, quindi lo sweep va raggruppato
    per profilo. Qui si verifica soltanto che il profilo attivo sia quello
    dichiarato — eseguire una matrice intera su un profilo diverso produce
    risultati silenziosamente sbagliati.
    """
    if not hasattr(bc, "set_profile"):
        return conn
    return bc.set_profile(conn, cfg, cfg.freq_target)


def _prepare_board_for_cell(conn, cfg) -> None:
    """Container fermo per le celle che non usano l'acceleratore.

    Non basta che sia inattivo: daemon e processi interni competono comunque
    per CPU e memoria.
    """
    if not cfg.hardware.get("axelera_available"):
        return
    if cfg.backend.name == "axelera":
        return
    from ..backends.axelera import container_enabled, stop_container

    if container_enabled(cfg) or cfg.hardware.get("provision", {}).get("container"):
        stop_container(conn, cfg)


def resolve_artifact(conn, cfg, backend) -> tuple[str, Path]:
    """Path dell'artefatto dove verra' misurato, e directory dell'export."""
    ekey = export_key(cfg)
    export_dir = find_export(cfg.artifacts_dir, ekey)
    if export_dir is None:
        raise MissingExport(
            f"nessun artefatto con export_key={ekey}: eseguire prima "
            f"`stage=export`"
        )
    meta = read_json(export_dir / "meta.json") or {}

    if backend.builds_on_target:
        # L'artefatto e' stato compilato sulla board e li' e' rimasto: non e'
        # portabile, quindi non si ricopia, si verifica che ci sia ancora.
        remote = read_json(export_dir / "remote.json") or {}
        path = remote.get("remote_path") or meta.get("artifact", {}).get("path")
        if not path:
            raise MissingExport(
                f"{ekey} risulta compilato on-target ma manca remote.json"
            )
        if conn.run(f"test -e {path}", hide=True, warn=True).failed:
            raise MissingExport(
                f"artefatto {path} non presente su {cfg.hardware.board}: "
                f"ricompilare con `stage=export +force_reexport=true`"
            )
        return path, export_dir

    artifact = locate_artifact(export_dir, cfg.backend.export_format)
    remote = sync_artifact(conn, cfg, artifact, subdir="exports", key=ekey)
    return str(remote), export_dir


def execute_benchmark(conn, cfg, raw_dest: Path, timer: PhaseTimer, bc,
                      applied: dict) -> dict:
    backend = get_backend(cfg.backend.name, cfg)
    artifact, export_dir = resolve_artifact(conn, cfg, backend)

    export_meta = read_json(export_dir / "meta.json") or {}
    validation = dict(export_meta.get("validation") or {})
    if validation.get("status") == "failed":
        # Un modello quantizzato male e' velocissimo e predice rumore: senza
        # questo controllo comparirebbe in tabella come il risultato migliore.
        why = validation.get("reason") or validation.get("error")
        raise BenchmarkFailed(
            f"artefatto non validato numericamente ({why}): "
            f"non ammesso al benchmark"
        )

    backend.prepare_input(conn, cfg)

    ct = resolve_compute(cfg)
    n_cores = _effective_cores(conn, cfg, bc, ct)
    cmd = _wrap_command(conn, cfg, backend.build_cmd(cfg, Path(artifact)),
                        ct, n_cores)

    log.info("misura: %s", cmd)
    with energy_sampler(conn, cfg) as sampler:
        with timer.phase("compute"):
            t0 = time.time()
            r = conn.run(cmd, hide=True, warn=True)
            wall_s = time.time() - t0

    raw = (r.stdout or "") + (("\n--- stderr ---\n" + r.stderr) if r.stderr else "")
    atomic_write_text(raw_dest.with_suffix(".raw.txt"), raw)
    log.debug("stdout:\n%s", r.stdout)

    if r.failed:
        raise BenchmarkFailed(
            f"{cfg.backend.benchmark.tool} exit code {r.return_code}:\n"
            f"{(r.stderr or r.stdout or '')[-2000:]}"
        )

    lat = backend.parse(r.stdout)
    bench = cfg.backend.benchmark
    warmup = {}
    if bench.get("warmup_ms") is not None:
        warmup["warmup_ms"] = int(bench.warmup_ms)
    if bench.get("warmup_iters") is not None:
        warmup["warmup_iters"] = int(bench.warmup_iters)

    # Il perimetro della misura e' dichiarato nel YAML del backend: e' li' che
    # si allinea fra tool diversi, e deve finire nel risultato per poterlo
    # verificare in analisi.
    scope = bench.get("scope") or backend.scope
    result = {
        "latency": latency_block(lat, int(bench.iters), warmup, scope),
        "runtime_state": _runtime_state(conn, cfg, bc, applied, n_cores),
    }
    if all(result["latency"].get(k) is None for k in ("p90_ms", "p95_ms", "p99_ms")):
        log.warning(
            "%s non riporta percentili: in analisi restera' solo il valore "
            "centrale", bench.tool,
        )

    energy = sampler.result(duration_s=wall_s)
    if energy:
        result["energy"] = energy

    result["accuracy"] = _accuracy(conn, cfg, backend, artifact, export_dir,
                                   ct, timer)
    result["validation"] = _validation_block(
        cfg, validation, export_meta, result["accuracy"]
    )
    return result


def _effective_cores(conn, cfg, bc, ct) -> int | None:
    """Core davvero utilizzabili, letti a runtime.

    Su Jetson il profilo di potenza determina quanti core sono online: il
    numero dichiarato nel config e' un'intenzione, non un fatto.
    """
    n_cores = ct.get("n_cores")
    if n_cores in (None, "all"):
        return bc.online_cores(conn)
    return int(n_cores)


def _wrap_command(conn, cfg, cmd: str, ct, n_cores) -> str:
    """Affinity, thread e priorita' attorno al comando del backend."""
    device = ct.get("device")
    env = thread_env(device, n_cores)
    affinity = affinity_spec(device, n_cores)
    if affinity:
        cmd = f"taskset -c {affinity} {cmd}"
    prio = cfg.stage.get("scheduling", {}).get("rt_priority")
    if prio and cmd.startswith("docker exec"):
        # Il prefisso finirebbe sul client docker, non sul processo dentro il
        # container: nessun effetto sullo scheduling di chi misura davvero.
        log.debug("comando nel container: scheduling FIFO non applicabile")
    elif prio and _has_rt(conn):
        cmd = f"chrt -f {int(prio)} {cmd}"
    return shell_env(env) + cmd


_RT_CACHE: dict[str, bool] = {}


def _has_rt(conn) -> bool:
    """`chrt` richiede CAP_SYS_NICE: se non c'e', si misura senza e lo si dice."""
    host = str(getattr(conn, "host", "local"))
    if host not in _RT_CACHE:
        available = conn.run("chrt -f 80 true", hide=True, warn=True).ok
        if not available:
            log.warning(
                "scheduling FIFO non disponibile su %s: la misura avra' piu' "
                "jitter (serve CAP_SYS_NICE)", host,
            )
        _RT_CACHE[host] = available
    return _RT_CACHE[host]


def _runtime_state(conn, cfg, bc, applied: dict, n_cores) -> dict:
    fq = resolve_freq(cfg)
    return {
        "freq_requested_khz": fq.get("khz"),
        "freq_actual_khz": bc.read_freq(conn),
        "governor": bc.read_governor(conn),
        "nvpmodel_id": applied.get("nvpmodel_id"),
        "cores_online": bc.online_cores(conn),
        "omp_num_threads": n_cores,
        "temp_start_c": applied.get("temp_start_c"),
        "temp_end_c": bc.read_temp(conn),
        "throttled": bc.read_throttle(conn),
        "swap_off": bc.swap_is_off(conn),
        "load_avg": bc.load_average(conn),
        "isolation": "none",
        "post_reboot": bool(getattr(bc, "post_reboot", False)),
        "rt_scheduling": _RT_CACHE.get(str(getattr(conn, "host", "local")), False),
    }


def _accuracy(conn, cfg, backend, artifact, export_dir: Path, ct, timer) -> dict:
    """mAP: una volta per artefatto esportato, non per cella.

    Non dipende da profilo di potenza, numero di core o frequenza: ripeterla
    per ogni riga della matrice costerebbe ore e darebbe lo stesso numero.

    Dove si calcola dipende dall'artefatto. Un engine TensorRT o un modello
    Axelera esistono solo sulla board, quindi la validazione va li'. Un ONNX e'
    portabile e la sua mAP non dipende dalla macchina: calcolarla sul Pi
    significherebbe farlo girare per ore su una CPU ARM, quindi si fa sulla
    workstation.
    """
    acc_path = export_dir / "accuracy.json"
    cached = read_json(acc_path)
    if cached:
        return cached

    if backend.builds_on_target:
        target_conn = conn
        model_path = artifact
        device = MAP_DEVICE.get(ct.get("device"))
    else:
        target_conn = LocalConnection()
        model_path = locate_artifact(export_dir, cfg.backend.export_format)
        device = None            # lascia scegliere a Ultralytics sulla workstation

    try:
        with timer.phase("accuracy"):
            acc = compute_map(target_conn, cfg, model_path, device=device)
        atomic_write_json(acc_path, acc)
        return acc
    except Exception as exc:  # noqa: BLE001 - la latenza resta valida
        log.error("calcolo mAP fallito per %s: %s", export_dir.name, exc)
        return {"status": "failed", "error": str(exc)}


def _validation_block(cfg, validation: dict, export_meta: dict,
                      accuracy: dict) -> dict:
    """Riporta la validazione dell'export e aggiunge il delta di mAP."""
    inspection = export_meta.get("inspection") or {}
    validation.setdefault("requested_e2e", True)
    validation["actual_e2e"] = inspection.get("actual_e2e")
    validation["head"] = inspection.get("head")

    ref_map50 = None
    weights_meta = read_json(
        Path(cfg.artifacts_dir) / "weights" / "index.json", default={}
    ) or {}
    entry = weights_meta.get(export_meta.get("training_key"))
    if entry and entry.get("metrics"):
        ref_map50 = entry["metrics"].get("map50")
    return grade_accuracy(
        validation, accuracy.get("map50"), ref_map50,
        cfg.quantization.get("validation"),
    )
