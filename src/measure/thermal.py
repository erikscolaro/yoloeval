"""Gestione termica.

Uno sweep lungo scalda le board e le ultime celle risultano sistematicamente
piu' lente delle prime. Il cooldown e' **condizionale**: non spreca tempo
quando la board e' gia' fredda, attende davvero quando serve.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


def wait_thermal(conn, controller, threshold_c: float = 45,
                 timeout_s: int = 300, poll_s: int = 10) -> float | None:
    """Attende che la board scenda sotto soglia. Ritorna la temperatura finale.

    Allo scadere del timeout procede comunque, ma logga: la cella resta
    utilizzabile perche' `temp_start_c` e `order_index` finiscono nel risultato
    e in analisi si vede se il termico stava inquinando le misure.
    """
    t0 = time.time()
    temp = controller.read_temp(conn)
    if temp is None:
        log.debug("nessun sensore di temperatura su %s", controller.name)
        return None
    while temp is not None and temp > threshold_c:
        if time.time() - t0 >= timeout_s:
            log.warning("timeout termico, procedo a %.1f C (soglia %.1f)",
                        temp, threshold_c)
            return temp
        log.info("attesa raffreddamento: %.1f C > %.1f C", temp, threshold_c)
        time.sleep(poll_s)
        temp = controller.read_temp(conn)
    return temp
