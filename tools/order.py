"""Ordinamento naturale dei nomi con numeri: cpu_1, cpu_2, cpu_4, cpu_16 invece di cpu_1,
cpu_16, cpu_2. I nomi nei config restano cpu_N: entrano nelle chiavi delle celle e dei probe.

    df.sort_values(["board", "compute_target"], key=natural_key)
    t.sort_index(axis=1, key=natural_key)
    sorted(names, key=natural)
"""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\d+")


def natural(value) -> str:
    """Chiave di ordinamento: i numeri dentro le stringhe riempiti a sinistra con zeri."""
    return _DIGITS.sub(lambda m: m.group().zfill(6), str(value))


def natural_key(col):
    """`key=` per sort_values/sort_index di pandas: ordine naturale per le colonne di testo
    (e per gli indici), le altre restano come sono."""
    from pandas.api.types import is_object_dtype, is_string_dtype

    if not (is_object_dtype(col) or is_string_dtype(col)):
        return col
    return col.map(lambda v: v if v is None or v != v else natural(v))
