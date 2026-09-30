"""Dalla latenza in funzione dei canali alla granularita' N.

Potare a blocchi di N rende raggiungibili solo i multipli di N: conviene l'N piu' piccolo i
cui multipli girano TUTTI con l'efficienza migliore che l'hardware offre a quella scala.

- efficienza e(C) = MAC / latenza;
- inviluppo E(C) = la migliore efficienza vista fra C - window e C (finestra a sinistra: di
  solito l'efficienza cresce con C, e confrontarsi con C piu' grandi penalizzerebbe tutti);
- un C e' "allineato" se e(C) >= (1 - tol) * E(C);
- N passa se sono allineati almeno (1 - max_violations) dei suoi multipli nel range (una
  misura sporca isolata non basta a bocciarlo);
- N ottimo = il PIU' PICCOLO N che passa, provando 1, 2, 4, ...

Con un hardware senza preferenze (efficienza liscia in C) passa gia' N = 1. Con blocchi SIMD
da 16 i multipli di 8 non di 16 restano sotto l'inviluppo, e passa 16.
"""

from __future__ import annotations


def efficiency(lat: dict, macs: dict) -> dict:
    return {c: macs[c] / lat[c] for c in lat if lat[c] > 0}


def envelope(eff: dict, window: int) -> dict:
    cs = sorted(eff)
    return {c: max(eff[x] for x in cs if c - window <= x <= c) for c in cs}


def candidates(c_min: int, c_max: int, given=None, min_multiples: int = 2) -> list[int]:
    """Gli N da provare: quelli dati, o 1, 2, 4, ... con almeno `min_multiples` multipli."""
    if given:
        return sorted({int(n) for n in given})
    out, n = [], 1
    while len(multiples(c_min, c_max, n)) >= min_multiples:
        out.append(n)
        n *= 2
    return out


def multiples(c_min: int, c_max: int, n: int) -> list[int]:
    return [c for c in range(c_min, c_max + 1) if c % n == 0]


def test_n(eff: dict, env: dict, n: int, tol: float, max_violations: float) -> dict:
    cs = sorted(eff)
    ms = [c for c in multiples(cs[0], cs[-1], n) if c in eff]
    ratio = {c: eff[c] / env[c] for c in ms}
    bad = sorted(c for c, r in ratio.items() if r < 1 - tol)
    frac = len(bad) / len(ms) if ms else 1.0
    return {"pass": bool(ms) and frac <= max_violations, "violations": bad,
            "violation_fraction": frac, "worst_ratio": min(ratio.values()) if ratio else None}


def best_n(lat: dict, macs: dict, tol: float, max_violations: float = 0.2,
           window: int = 32, given=None) -> dict:
    """{n_opt, conclusive, tested: {N: {...}}}. n_opt None se nessun N passa (misure troppo
    rumorose o range troppo stretto)."""
    eff = efficiency(lat, macs)
    env = envelope(eff, window)
    cs = sorted(eff)
    tested, n_opt = {}, None
    for n in candidates(cs[0], cs[-1], given):
        res = test_n(eff, env, n, tol, max_violations)
        tested[n] = res
        if res["pass"]:
            n_opt = n
            break
    for n, r in tested.items():
        r["optimal"] = n == n_opt
    return {"n_opt": n_opt, "conclusive": n_opt is not None, "tested": tested}


def global_n(per_shape: dict) -> int | None:
    """N unico per la rete: il PIU' GRANDE fra le forme concluse che contano per l'N globale
    (`in_global`), cosi' i multipli sono allineati per tutte (i multipli di 16 lo sono anche
    di 8). Le forme informative (es. i lineari) restano solo nel summary. None se nessuna
    forma globale e' conclusa."""
    ns = [v["n_opt"] for v in per_shape.values() if v.get("n_opt") and v.get("in_global", True)]
    return max(ns) if ns else None
