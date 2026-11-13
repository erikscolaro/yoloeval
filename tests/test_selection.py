from src.selection import scegli

# dati del PI a 640 (B = 28 ms): intera, pavimento, mAP50-95
LAT = {"P-": 26.0, "P": 28.0, "PN": 44.6, "N-": 53.2, "N": 56.1, "N+": 70.5, "S": 142.7}
FLOOR = {"P-": (14.7, False), "P": (15.7, False), "PN": (18.5, False), "N-": (21.3, False), "N": (22.3, True),
         "N+": (26.2, False), "S": (41.0, True)}
MAP = {"P-": 0.570, "P": 0.569, "PN": 0.588, "N-": 0.596, "N": 0.596, "N+": 0.601, "S": 0.610}


def regola(budget=28, start=None, m=MAP):
    return scegli(LAT, m, FLOOR, budget, 0.85, 0.01, 0.004, start)


def test_stretto_sceglie_il_pavimento_piu_basso_fra_i_pari():
    r = regola()
    assert (r["caso"], r["ottimo"], r["tetto"], r["partenza"]) == ("stretto", "P-", "S", "N-")


def test_libero_se_il_tetto_sta_nel_budget():
    assert regola(budget=150)["caso"] == "libero"


def test_libero_se_il_tetto_perde_meno_di_un_punto():
    assert regola(m=MAP | {"S": 0.575, "N+": 0.574, "N": 0.573, "N-": 0.573, "PN": 0.572})["caso"] == "libero"


def test_scelta_manuale_avvisa_se_non_passa_il_pavimento():
    r = regola(start="N+")
    assert r["partenza"] == "N+" and r["avviso"]
    assert regola(start="N")["avviso"] is None
