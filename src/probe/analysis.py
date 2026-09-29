"""Dalla latenza in funzione dei canali alla granularita' N.

Potare a blocchi di N rende raggiungibili solo i multipli di N. Un N non fa perdere nulla se,
dentro ogni blocco ((k-1)N, kN], nessun C intermedio e' piu' veloce di kN oltre il rumore:
la latenza e' piatta (o peggiore, con canali non allineati) fino al multiplo successivo.
Si prova N = 1, 2, 4, ... finche' la condizione regge: l'ultimo N che la rispetta e'
l'N ottimo, cioe' la larghezza del gradino.
"""

from __future__ import annotations


def candidates(c_min: int, c_max: int, given=None) -> list[int]:
    """Gli N da provare: quelli dati, o le potenze di 2 con almeno un blocco intero nel range."""
    if given:
        return sorted({int(n) for n in given})
    out, n = [], 1
    while _blocks(c_min, c_max, n):
        out.append(n)
        n *= 2
    return out


def _blocks(c_min: int, c_max: int, n: int) -> list[int]:
    """Multipli kN con tutto il blocco ((k-1)N, kN] dentro [c_min, c_max]."""
    first = -(-(c_min + n - 1) // n) * n if n > 1 else c_min + 1
    return [m for m in range(first, c_max + 1, n) if m - n + 1 >= c_min]


def test_n(lat: dict, n: int, tol: float) -> dict:
    """lat: {C: latenza}. Ritorna pass e il guadagno peggiore trovato dentro un blocco."""
    worst_gain, where = 0.0, None
    cs = sorted(lat)
    for m in _blocks(cs[0], cs[-1], n):
        ref = lat.get(m)
        if ref is None:
            continue
        for c in range(m - n + 1, m):
            if c in lat:
                gain = (ref - lat[c]) / ref           # > 0: C intermedio piu' veloce
                if gain > worst_gain:
                    worst_gain, where = gain, (c, m)
    return {"pass": worst_gain <= tol, "worst_gain": worst_gain, "at": where}


def best_n(lat: dict, tol: float, given=None) -> dict:
    """{n_opt, at_least, tested: {N: {...}}}. at_least: anche l'N piu' grande provato passa,
    l'ottimo potrebbe essere piu' grande (serve un range di canali piu' ampio)."""
    cs = sorted(lat)
    tested, n_opt = {}, 1
    for n in candidates(cs[0], cs[-1], given):
        res = test_n(lat, n, tol)
        tested[n] = res
        if not res["pass"]:
            break
        n_opt = n
    at_least = all(r["pass"] for r in tested.values())
    for n, r in tested.items():
        r["optimal"] = n == n_opt
    return {"n_opt": n_opt, "at_least": at_least, "tested": tested}


def global_n(per_shape: dict) -> int:
    """N unico per la rete: il piu' piccolo fra le forme, cosi' non si perde nulla in nessuna."""
    return min(v["n_opt"] for v in per_shape.values())
