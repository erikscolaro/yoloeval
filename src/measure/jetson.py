"""Jetson: nvpmodel, jetson_clocks, tegrastats.

Gerarchia: `nvpmodel` impone il tetto, il governor sfrutta il tetto. L'ordine
e' prima `nvpmodel -m <id>`, poi eventualmente cpufreq nel range consentito da
quel profilo — scrivere una frequenza superiore al tetto non ha effetto.

Il cambio di profilo **non e' sempre applicabile a runtime**: se la transizione
richiede un valore diverso di `tpc_pg_mask` serve il riavvio del sistema.
Questo rompe l'assunzione che `tuned()` possa cambiare profilo per singola
cella, ed e' il motivo per cui lo sweep va raggruppato per profilo.
"""

from __future__ import annotations

import logging
import re
import time

from ..errors import ProfileChangeRequiresReboot, ProfileMismatch
from .board import BoardController, _out
from .compute import resolve_freq

log = logging.getLogger(__name__)

#: secondi di stabilizzazione dopo un cambio di profilo
SETTLE_S = 30

_ACTIVE_RE = re.compile(r"NV Power Mode:\s*\S+\s*\n?\s*(\d+)", re.IGNORECASE)
_ACTIVE_RE_ALT = re.compile(r"^\s*(\d+)\s*$", re.MULTILINE)
_REBOOT_RE = re.compile(
    r"requires.*reboot|reboot.*required|tpc_pg_mask", re.IGNORECASE
)


def needs_reboot(stdout: str) -> bool:
    return bool(_REBOOT_RE.search(stdout or ""))


def parse_active_profile(stdout: str) -> int | None:
    """`nvpmodel -q` stampa il nome del profilo e, sulla riga dopo, l'id."""
    if not stdout:
        return None
    m = _ACTIVE_RE.search(stdout)
    if m:
        return int(m.group(1))
    m = _ACTIVE_RE_ALT.search(stdout)
    return int(m.group(1)) if m else None


class JetsonController(BoardController):
    name = "jetson"

    # --- stato -----------------------------------------------------------
    def read_state(self, conn) -> dict:
        return {
            "nvpmodel_id": parse_active_profile(
                _out(conn.run("sudo nvpmodel -q", hide=True, warn=True)) or ""
            ),
            "governor": self.read_governor(conn),
            "freq_khz": self.read_freq(conn),
            "swap_off": self.swap_is_off(conn),
        }

    def apply_state(self, conn, cfg) -> dict:
        fq = resolve_freq(cfg)
        # Il profilo e' gia' stato applicato a livello di sweep (set_profile):
        # qui si verifica soltanto, perche' una transizione che richiede reboot
        # non e' un'azione della singola cella.
        active = parse_active_profile(
            _out(conn.run("sudo nvpmodel -q", hide=True, warn=True)) or ""
        )
        if fq.get("nvpmodel_id") is not None and active != fq.nvpmodel_id:
            raise ProfileMismatch(
                f"profilo richiesto {fq.nvpmodel_id}, attivo {active}: "
                f"applicarlo con set_profile prima di eseguire le celle"
            )
        if fq.get("governor"):
            self.set_governor(conn, fq.governor)
        if fq.get("khz"):
            self.set_freq_khz(conn, int(fq.khz))
        if cfg.stage.get("scheduling", {}).get("swap_off", True):
            self.set_swap(conn, off=True)
        return {
            "nvpmodel_id": active,
            "governor": self.read_governor(conn),
            "freq_requested_khz": fq.get("khz"),
            "freq_actual_khz": self.read_freq(conn),
            "cores_online": self.online_cores(conn),
        }

    def restore_state(self, conn, saved: dict) -> None:
        if saved.get("governor"):
            self.set_governor(conn, saved["governor"])
        if saved.get("swap_off") is False:
            self.set_swap(conn, off=False)
        # Il profilo nvpmodel non viene ripristinato per cella: e' proprieta'
        # dello sweep, e cambiarlo qui costerebbe una stabilizzazione (o un
        # riavvio) per ogni misura.

    # --- letture ---------------------------------------------------------
    def read_temp(self, conn) -> float | None:
        raw = _out(conn.run(
            "cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null",
            hide=True, warn=True))
        if not raw:
            return None
        values = [int(v) / 1000.0 for v in raw.split() if v.strip().isdigit()]
        # Zone spurie a valori impossibili esistono: si tiene il massimo
        # plausibile, che e' quello che determina il throttling.
        values = [v for v in values if -20 < v < 130]
        return max(values) if values else None

    def read_throttle(self, conn) -> bool | None:
        r = conn.run(
            "cat /sys/devices/virtual/thermal/thermal_zone*/cdev*/cur_state "
            "2>/dev/null", hide=True, warn=True)
        raw = _out(r)
        if raw is None:
            return None
        return any(v.strip().isdigit() and int(v) > 0 for v in raw.split())

    def read_power(self, conn) -> float | None:
        sample = self._tegrastats(conn, n=1)
        return sample.get("mean_power_w") if sample else None

    def read_energy(self, conn, duration_s: float | None = None) -> dict | None:
        sample = self._tegrastats(conn, n=5)
        if not sample:
            return None
        if duration_s:
            sample["energy_j"] = round(sample["mean_power_w"] * duration_s, 3)
            sample["window_s"] = duration_s
        return sample

    def _tegrastats(self, conn, n: int = 5, interval_ms: int = 200) -> dict | None:
        from .power import parse_tegrastats

        r = conn.run(
            f"timeout {max(2, int(n * interval_ms / 1000) + 2)} "
            f"sudo tegrastats --interval {interval_ms} | head -n {n}",
            hide=True, warn=True,
        )
        if not getattr(r, "ok", False) or not r.stdout.strip():
            return None
        return parse_tegrastats(r.stdout)

    # --- profilo di potenza ---------------------------------------------
    def set_profile(self, conn, cfg, target: str):
        """Applica il profilo nvpmodel. Livello sweep, non livello cella.

        Se dopo l'applicazione il profilo attivo non corrisponde a quello
        richiesto il tool si ferma: eseguire una matrice intera su un profilo
        diverso da quello dichiarato produce risultati silenziosamente
        sbagliati.
        """
        fq = cfg.hardware.freq[target]
        if fq.get("nvpmodel_id") is None:
            return conn

        # `nvpmodel` chiede conferma interattiva quando serve il riavvio:
        # pty + risposta automatica, altrimenti il comando resta appeso.
        r = conn.run(
            f"echo NO | sudo nvpmodel -m {fq.nvpmodel_id}",
            warn=True, pty=True, hide=True,
        )
        out = (getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")

        if needs_reboot(out):
            if not cfg.allow_reboot:
                raise ProfileChangeRequiresReboot(
                    f"il profilo {target} richiede il riavvio della board. "
                    f"Rilancia con +allow_reboot=true, oppure riavvia a mano e "
                    f"riprendi lo sweep."
                )
            from ..remote.provision import ensure_env, reboot_and_wait

            conn = reboot_and_wait(conn, cfg)
            ensure_env(conn, cfg)  # lo stato post-reboot e' azzerato
            cfg._post_reboot = True

        time.sleep(SETTLE_S)
        active = parse_active_profile(
            _out(conn.run("sudo nvpmodel -q", hide=True, warn=True)) or ""
        )
        if active != fq.nvpmodel_id:
            raise ProfileMismatch(
                f"richiesto {fq.nvpmodel_id}, attivo {active}"
            )
        # jetson_clocks fissa CPU, GPU, DLA ed EMC al massimo del profilo
        # corrente: per i test su GPU fissare il clock GPU conta quanto
        # fissare quello CPU.
        if fq.get("jetson_clocks", True):
            conn.run("sudo jetson_clocks", hide=True, warn=True)
        log.info("profilo %s (nvpmodel %s) attivo", target, active)
        return conn
