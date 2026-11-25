"""Variante di optuna_kalman.py con l'obiettivo della torretta: F2 sui droni (recall pesato il doppio della precisione)
per fotogramma, IoU >= 0,5, con il vincolo che aerei ed elicotteri veri con un track drone sopra non siano piu' del
detector da solo (P- alla sua soglia con F2 massimo sui droni, sugli stessi fotogrammi). Stesso spazio di ricerca, stessi
R e Q, stessi tre studi; database e file separati, cosi' i risultati con l'F1 restano.
Uscita: results/kalman_f2_<studio>_<split>.json e .csv; prove in runs/kalman_f2_<split>.db.

    python experiments/tracking/taratura_kalman/optuna_kalman_f2.py [--split val] [--trials 1000]
"""
import argparse
import json
import os
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np
import optuna

from comune import FONTI, HERE, conta_drone, sintesi
from optuna_kalman import STUDI, _base, _dati, kalman, proponi

CONFS = np.round(np.r_[0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, np.arange(0.05, 0.95, 0.05)], 2)


def misura(segs, gen):
    """gen(seg) -> uscita per fotogramma; sintesi() per fonte e totale."""
    acc = {f: np.zeros(5) for f in FONTI}
    for s in segs:
        for (gc, gb, _), out in zip(s["frames"], gen(s)):
            acc[s["fonte"]] += conta_drone(out, gc, gb)
    return {f: sintesi(a) for f, a in acc.items() if a[0] + a[1] + a[3]} | {"totale": sintesi(sum(acc.values()))}


def detector(segs):
    return max((misura(segs, lambda s, c=c: (d[d[:, 4] >= c] for _, _, d in s["frames"])) | {"conf": float(c)}
                for c in CONFS), key=lambda m: m["totale"]["f2"])


def _apri(nome, split, rif=None):
    st = optuna.create_study(
        study_name=f"{nome}_{split}", direction="maximize", load_if_exists=True,
        storage=optuna.storages.RDBStorage(f"sqlite:///{HERE / 'runs' / f'kalman_f2_{split}.db'}",
                                           engine_kwargs={"connect_args": {"timeout": 120}}),
        sampler=optuna.samplers.TPESampler(multivariate=True, constraints_func=lambda t: [t.user_attrs["eccesso"]]))
    if rif is not None:
        st.set_user_attr("riferimento", rif)
    return st


def obiettivo(tr, segs, rq, limite):
    m = misura(segs, kalman(proponi(tr, rq)))
    tr.set_user_attr("eccesso", m["totale"]["aerei_come_drone"] - limite)   # <= 0: vincolo rispettato
    for f, v in m.items():
        for k, x in v.items():
            tr.set_user_attr(f"{f}_{k}", x)
    return m["totale"]["f2"]


def cerca(args):
    nome, split, n = args
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    segs, rq = _dati(nome, split)
    st = _apri(nome, split)
    limite = st.user_attrs["riferimento"]["totale"]["aerei_come_drone"]
    st.optimize(lambda tr: obiettivo(tr, segs, rq, limite), n_trials=n)


def migliore(st):
    ok = [t for t in st.trials if t.state.name == "COMPLETE" and t.user_attrs.get("eccesso", 1) <= 0]
    return max(ok, key=lambda t: t.value) if ok else None, len(ok)


def riassumi(args):
    nome, split = args
    segs, rq = _dati(nome, split)
    st = _apri(nome, split)
    best, n_ok = migliore(st)
    try:
        imp = optuna.importance.get_param_importances(st)
    except Exception as e:  # noqa: BLE001  l'importanza e' un di piu'
        imp = {"errore": str(e)}
    p = proponi(optuna.trial.FixedTrial(best.params), rq) if best else None
    out = {"studio": nome, "split": split, "prove": len(st.trials), "prove_nel_vincolo": n_ok, "rq": rq,
           "parametri": asdict(p) if p else None, "kalman_tarato": misura(segs, kalman(p)) if p else None,
           "kalman_partenza": misura(segs, kalman(_base(rq))), "detector_miglior_conf": st.user_attrs["riferimento"],
           "importanza": imp}
    (HERE / "results" / f"kalman_f2_{nome}_{split}.json").write_text(json.dumps(out, indent=1))
    df = st.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df.columns = [c.removeprefix("params_").removeprefix("user_attrs_") for c in df.columns]
    df.rename(columns={"value": "f2"}).sort_values("number").to_csv(
        HERE / "results" / f"kalman_f2_{nome}_{split}.csv", index=False, float_format="%.4g")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val"), ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--studi", nargs="*", default=list(STUDI))
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cpu = os.cpu_count()
    for nome in a.studi:                     # riferimento e prima prova (i parametri di partenza), prima del parallelo
        if not len(_apri(nome, a.split).trials):
            segs, rq = _dati(nome, a.split)
            st, b = _apri(nome, a.split, detector(segs)), _base(rq)
            st.enqueue_trial({k: getattr(b, k) for k in ("hi", "lo", "min_hits", "max_age", "gate", "v0", "iou_dup",
                                                         "vote")} | {"show_age": b.max_age})
            cerca((nome, a.split, 1))
    per, lavori = max(cpu // len(a.studi), 1), []
    for nome in a.studi:
        resto = max(a.trials - len([t for t in _apri(nome, a.split).trials if t.state.is_finished()]), 0)
        lavori += [(nome, a.split, resto // per + (i < resto % per)) for i in range(per)]
    with Pool(cpu) as pool:
        pool.map(cerca, [l for l in lavori if l[2]], chunksize=1)
        ris = pool.map(riassumi, [(n, a.split) for n in a.studi])
    print(f"{'studio':10s} {'fonte':10s} {'modo':16s}    F2 recall  prec aerei->drone")
    for r in ris:
        for modo in ("detector_miglior_conf", "kalman_partenza", "kalman_tarato"):
            for f, m in (r[modo] or {}).items():
                if isinstance(m, dict):
                    print(f"{r['studio']:10s} {f:10s} {modo:16s} {m['f2']:.3f} {m['recall']:.3f} {m['precisione']:.3f}"
                          f" {m['aerei_come_drone']:.4f}")
        if r["parametri"]:
            print(f"{r['studio']:10s} parametri: " + " ".join(f"{k}={v:.3g}" for k, v in r["parametri"].items()
                                                              if k != "fps"))


if __name__ == "__main__":
    main()
