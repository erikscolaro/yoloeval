"""Dalla latenza in funzione dei canali alla granularita' N.

Potare a blocchi di N rende raggiungibili solo i multipli di N: conviene l'N per cui i multipli
di N girano meglio dei canali intermedi. Regola locale, a raddoppi:

- efficienza e(C) = MAC / latenza;
- per ogni n = 1, 2, 4, ... si prendono i multipli "dispari" di n (multipli di n ma non di 2n,
  da `c_from` in su) e per ciascuno il rapporto r(C) = e(C) / media(e(C - n), e(C + n)), cioe'
  contro i due multipli di 2n che lo circondano;
- 2n passa se la mediana dei rapporti e' sotto 1 - tol: i multipli dispari di n girano
  sistematicamente peggio dei multipli di 2n;
- N ottimo = il PIU' GRANDE 2n che passa; 1 se nessuno passa.

Il confronto con i vicini toglie l'andamento di fondo (l'efficienza che sale con C) e la mediana
rende innocua una misura sporca isolata. Con un hardware senza preferenze (efficienza liscia in
C) nessun 2n passa e N = 1; con blocchi SIMD da 16 i multipli di 8 non di 16 restano sotto i
vicini e passa 16.
"""

from __future__ import annotations

from statistics import median


def efficiency(lat: dict, macs: dict) -> dict:
    return {c: macs[c] / lat[c] for c in lat if lat[c] > 0}


def odd_multiples(eff: dict, n: int, c_from: int) -> list[int]:
    """Multipli di n non di 2n, da c_from in su, con entrambi i vicini C - n e C + n misurati."""
    return [c for c in sorted(eff) if c >= c_from and c % n == 0 and c % (2 * n)
            and c - n in eff and c + n in eff]


def candidates(eff: dict, c_from: int, given=None, min_points: int = 3) -> list[int]:
    """Gli N da provare: quelli dati (> 1), o 2, 4, 8, ... con almeno `min_points` rapporti
    (con due la mediana e' una media e una misura sporca basta a far passare N)."""
    if given:
        return sorted({int(n) for n in given if int(n) > 1})
    out, n = [], 1
    while len(odd_multiples(eff, n, c_from)) >= min_points:
        out.append(2 * n)
        n *= 2
    return out


def check_n(eff: dict, n: int, tol: float, c_from: int) -> dict:
    """N passa se i multipli dispari di N/2 stanno sistematicamente sotto i multipli di N vicini."""
    h = n // 2
    ratio = {c: eff[c] / ((eff[c - h] + eff[c + h]) / 2) for c in odd_multiples(eff, h, c_from)}
    med = median(ratio.values()) if ratio else None
    return {"pass": med is not None and med < 1 - tol, "median_ratio": med,
            "points": len(ratio), "below": sorted(c for c, r in ratio.items() if r < 1 - tol)}


def best_n(lat: dict, macs: dict, tol: float, c_from: int = 8, given=None) -> dict:
    """{n_opt, conclusive, tested: {N: {...}}}. n_opt e' sempre almeno 1."""
    eff = efficiency(lat, macs)
    tested = {n: check_n(eff, n, tol, c_from) for n in candidates(eff, c_from, given)}
    n_opt = max([n for n, r in tested.items() if r["pass"]], default=1)
    for n, r in tested.items():
        r["optimal"] = n == n_opt
    return {"n_opt": n_opt, "conclusive": True, "tested": tested}


def global_n(per_shape: dict) -> int | None:
    """N unico per la rete: il PIU' GRANDE fra le forme concluse che contano per l'N globale
    (`in_global`), cosi' i multipli sono allineati per tutte (i multipli di 16 lo sono anche
    di 8). Le forme informative (es. i lineari) restano solo nel summary. None se nessuna
    forma globale e' conclusa."""
    ns = [v["n_opt"] for v in per_shape.values() if v.get("n_opt") and v.get("in_global", True)]
    return max(ns) if ns else None
