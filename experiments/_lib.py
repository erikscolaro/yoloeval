"""Codice comune degli esperimenti. Ogni run.py lo importa con

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/
    from _lib import ROOT

ROOT e' la root del repo e finisce in sys.path (per `src` e `yolopit`).

Gli helper `make_cfg`, `weights_or_exit`, `run_stage` e `val_images` servono a ricavare i pesi
dalla cache della pipeline e a lanciare gli stadi, senza path fissi dentro artifacts/.
"""
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PiRam(threading.Thread):
    """Campiona sul Pi ogni secondo: MB usati, MB disponibili, benchmark_app attivo, eval attivo."""
    LOOP = ("while true; do m=$(free -m | awk 'NR==2{print $3, $7}'); "
            "l=$(pgrep -c benchmark_app); a=$(pgrep -cf '[e]val_map.py'); "
            "echo \"$m $l $a\"; sleep 1; done")

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "rpi5", self.LOOP],
                                     stdout=subprocess.PIPE, text=True)

    def run(self):
        for line in self.proc.stdout:
            parts = line.split()
            if len(parts) == 4:
                self.samples.append([int(x) for x in parts])

    def stop(self) -> dict:
        self.proc.terminate()
        used = lambda pick: max((s[0] for s in self.samples if pick(s)), default=None)
        return {"before_mb": self.samples[0][0] if self.samples else None,
                "peak_latency_mb": used(lambda s: s[2] > 0),
                "peak_map_mb": used(lambda s: s[3] > 0),
                "peak_mb": used(lambda s: True),
                "min_available_mb": min((s[1] for s in self.samples), default=None)}


PYTHON = str(ROOT / ".venv/bin/python")


def make_cfg(*overrides: str):
    """Config composta come fa tests/conftest.py::make_cfg. `dry_run=true`: cercare chiavi e pesi
    non deve lanciare il probe di N (`pit.n: auto`) sulla board; il flag non entra nelle chiavi."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return compose("config", overrides=[f"project_root={ROOT}", "dry_run=true", *overrides])


def chain_hint(cfg, overrides, upto: str | None = None) -> str:
    """Gli `run.py stage=...` che producono i pesi di `final_weights(cfg)`, nell'ordine
    (fino a `upto` compreso, se dato)."""
    from src.cache import weights_chain

    stages = weights_chain(cfg)
    if upto:
        stages = stages[:stages.index(upto) + 1]
    return "\n".join(f"  .venv/bin/python run.py stage={s} {' '.join(overrides)}" for s in stages)


def weights_or_exit(*overrides: str) -> Path:
    """best.pt che la pipeline manda all'export per questa config (`src.cache.final_weights`);
    se manca, esce dicendo quali stadi lanciare."""
    from src.cache import final_weights

    cfg = make_cfg(*overrides)
    path, key = final_weights(cfg)
    if path is None or not path.exists():
        sys.exit(f"pesi {key} non in cache; lancia prima, dalla root del repo:\n"
                 + chain_hint(cfg, overrides))
    return path


def run_stage(stage: str, *overrides: str, multirun: bool = False, log=None,
              check: bool = True) -> int:
    """`run.py [-m] stage=<stage> <overrides>` dalla root; con `log` l'output va su file."""
    cmd = [PYTHON, "run.py", *(["-m"] if multirun else []), f"stage={stage}", *overrides]
    print(">>>", " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check, stdout=log,
                          stderr=subprocess.STDOUT if log else None,
                          stdin=subprocess.DEVNULL if log else None).returncode


def val_images(cfg) -> list[Path]:
    """Immagini dello split `val` del data yaml di cfg.dataset (path relativi alla sua cartella,
    oppure a `path:` se il yaml lo dice)."""
    import yaml

    yml = Path(cfg.dataset.yaml)
    conf = yaml.safe_load(yml.read_text())
    d = (yml.parent / conf.get("path", ".") / conf["val"]).resolve()
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
