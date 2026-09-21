"""Raccolta delle versioni di runtime — blocco `env` dello schema.

Ogni campo non determinabile va a `None`, mai omesso: la presenza della chiave
documenta il tentativo. Su celle remote le versioni vanno raccolte *sulla
board*, non sulla workstation — un `onnxruntime` 1.19 sulla workstation non
dice nulla su cosa gira sulla Jetson.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

#: pacchetti Python di cui si registra la versione
PY_PACKAGES = (
    "ultralytics",
    "torch",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "tensorrt",
    "openvino",
    "executorch",
    "hydra-core",
)

#: chiavi sempre presenti nel blocco env, anche a None
ENV_KEYS = (
    "ultralytics",
    "torch",
    "onnx",
    "onnxruntime",
    "tensorrt",
    "cuda",
    "jetpack",
    "openvino",
    "executorch",
    "axelera_sdk",
    "metis_dkms",
    "git_sha",
    "python",
    "os",
    "kernel",
    "arch",
    "hostname",
    "collected_on",
)

# Snippet eseguito sulla board: le versioni si leggono dal suo interprete.
_REMOTE_SNIPPET = r"""
import json, platform
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError
out = {"python": platform.python_version(), "arch": platform.machine()}
for pkg in %(packages)s:
    try:
        out[pkg] = version(pkg)
    except Exception:
        out[pkg] = None
print(json.dumps(out))
"""


def _run(cmd: list[str] | str, timeout: int = 20) -> str | None:
    """Esegue un comando locale, None se assente o fallito."""
    try:
        r = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or r.stderr).strip() or None


def _pkg_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # metadata corrotti: non deve far fallire la cella
        return None


def git_sha(root: str | Path | None = None) -> str | None:
    """Commit del tool. Sempre della workstation: e' il codice che orchestra."""
    cwd = str(root) if root else None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    sha = r.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd
    )
    if dirty.returncode == 0 and dirty.stdout.strip():
        sha += "-dirty"
    return sha or None


def _parse_jetpack(nv_tegra_release: str | None) -> str | None:
    """`# R36 (release), REVISION: 4.3, ...` -> "36.4.3"."""
    if not nv_tegra_release:
        return None
    m = re.search(r"R(\d+).*?REVISION:\s*([\d.]+)", nv_tegra_release)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2).rstrip('.')}"


def _parse_cuda(nvcc_out: str | None) -> str | None:
    if not nvcc_out:
        return None
    m = re.search(r"release\s+([\d.]+)", nvcc_out)
    return m.group(1) if m else None


def _parse_trtexec(out: str | None) -> str | None:
    if not out:
        return None
    m = re.search(r"TensorRT\.trtexec.*?\[TensorRT v(\d+)\]", out) or re.search(
        r"v?(\d+\.\d+\.\d+)", out
    )
    if not m:
        return None
    raw = m.group(1)
    # trtexec riporta la versione compatta (100300 -> 10.3.0)
    if raw.isdigit() and len(raw) >= 5:
        return f"{int(raw[:-4])}.{int(raw[-4:-2])}.{int(raw[-2:])}"
    return raw


def _local_versions(project_root: str | Path | None = None) -> dict:
    ort = _pkg_version("onnxruntime") or _pkg_version("onnxruntime-gpu")
    nv_tegra = None
    tegra_file = Path("/etc/nv_tegra_release")
    if tegra_file.exists():
        nv_tegra = tegra_file.read_text(errors="replace").splitlines()[0]

    trt = _pkg_version("tensorrt") or _parse_trtexec(
        _run("trtexec --version") or _run("/usr/src/tensorrt/bin/trtexec --version")
    )
    cuda = _parse_cuda(_run("nvcc --version"))
    if cuda is None:
        smi = _run("nvidia-smi") or ""
        m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
        cuda = m.group(1) if m else None

    axelera = None
    if shutil.which("axdevice"):
        axelera = _pkg_version("axelera-runtime") or _pkg_version("axelera-devkit")
        axelera = axelera or _run("axdevice --version")
    metis = _run("dpkg-query -W -f='${Version}' metis-dkms")

    return {
        "ultralytics": _pkg_version("ultralytics"),
        "torch": _pkg_version("torch"),
        "onnx": _pkg_version("onnx"),
        "onnxruntime": ort,
        "tensorrt": trt,
        "cuda": cuda,
        "jetpack": _parse_jetpack(nv_tegra),
        "openvino": _pkg_version("openvino"),
        "executorch": _pkg_version("executorch"),
        "axelera_sdk": axelera,
        "metis_dkms": metis,
        "git_sha": git_sha(project_root),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "kernel": platform.release(),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "collected_on": "local",
    }


def _remote_versions(conn, cfg=None) -> dict:
    """Stesse informazioni, lette sulla board attraverso la connessione."""
    env = {k: None for k in ENV_KEYS}
    env["collected_on"] = str(getattr(conn, "host", "remote"))

    python = "python3"
    if cfg is not None and cfg.hardware.get("remote", {}).get("python"):
        python = cfg.hardware.remote.python

    import json as _json

    snippet = _REMOTE_SNIPPET % {"packages": list(PY_PACKAGES)}
    r = conn.run(f"{python} - <<'PYEOF'\n{snippet}\nPYEOF", hide=True, warn=True)
    if r.ok:
        try:
            payload = _json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            payload = {}
            log.warning("versioni remote non parsabili: %s", r.stdout[:200])
        env["python"] = payload.get("python")
        env["arch"] = payload.get("arch")
        for pkg in ("ultralytics", "torch", "onnx", "openvino", "executorch"):
            env[pkg] = payload.get(pkg)
        env["onnxruntime"] = payload.get("onnxruntime") or payload.get("onnxruntime-gpu")
        env["tensorrt"] = payload.get("tensorrt")

    def sh(cmd):
        res = conn.run(cmd, hide=True, warn=True)
        return res.stdout.strip() if res.ok and res.stdout.strip() else None

    env["tensorrt"] = env["tensorrt"] or _parse_trtexec(
        sh("trtexec --version 2>&1 || /usr/src/tensorrt/bin/trtexec --version 2>&1")
    )
    env["jetpack"] = _parse_jetpack(sh("head -1 /etc/nv_tegra_release 2>/dev/null"))
    env["cuda"] = _parse_cuda(sh("nvcc --version 2>/dev/null"))
    env["axelera_sdk"] = sh("axdevice --version 2>/dev/null")
    env["metis_dkms"] = sh("dpkg-query -W -f='${Version}' metis-dkms 2>/dev/null")
    env["kernel"] = sh("uname -r")
    env["os"] = sh(". /etc/os-release 2>/dev/null && echo $PRETTY_NAME")
    env["hostname"] = sh("hostname")
    env["git_sha"] = git_sha()  # il commit del tool, che gira sulla workstation
    return env


def collect_versions(conn=None, cfg=None, project_root=None) -> dict:
    """Popola il blocco `env` dello schema.

    `conn` None o locale -> versioni di questa macchina; connessione remota ->
    versioni della board.
    """
    if conn is None or getattr(conn, "is_local", False):
        env = _local_versions(project_root)
    else:
        try:
            env = _remote_versions(conn, cfg)
        except Exception as exc:  # la raccolta non deve far fallire la cella
            log.warning("raccolta versioni remote fallita: %s", exc)
            env = {k: None for k in ENV_KEYS}
            env["collected_on"] = "remote-failed"
    for k in ENV_KEYS:
        env.setdefault(k, None)
    return env
