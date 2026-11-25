"""Passo 3 della taratura: Optuna cerca i parametri del ciclo di vita del Kalman sulle detection salvate (detection.py),
con R e Q fissi da rq.py. Tre studi indipendenti: svanstrom, anti_uav e tutte (le due fonti insieme).
Obiettivo: F1 sui fotogrammi dei track mostrati (IoU >= 0,5 e stessa classe), contando ogni oggetto vero perso e ogni
track mostrato senza oggetto. Per confronto: il detector da solo alla soglia di conf con F1 massimo, e il Kalman con i
parametri di partenza (default + R e Q di rq.py).
Uscita: results/kalman_f1_<studio>_<split>.json (migliori parametri, metriche per fonte, importanza dei parametri) e
results/kalman_f1_<studio>_<split>.csv (tutte le prove);
le prove restano in runs/kalman_f1_<split>.db (si riprende da li' se si rilancia). Usa tutti i core: piu' processi per
studio, che condividono le prove nel database.

    python experiments/tracking/taratura_kalman/optuna_kalman.py [--split val] [--trials 200]
"""
import argparse
import json
import os
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np
import optuna

from comune import FONTI, HERE, NAMES, carica, conta, f1
from src.tracking import KalmanTracker, Params

STUDI = FONTI + ("tutte",)


def metriche(segs, fn_out):
    """fn_out(seg) -> output per fotogramma; tp, fn, fp, recall, falsi/fotogramma, F1 per fonte e in totale."""
    acc = {f: np.zeros(4) for f in FONTI}
    for s in segs:
        for (gc, gb, _), out in zip(s["frames"], fn_out(s)):
            acc[s["fonte"]] += (*conta(out, gc, gb), 1)
    res = {}
    for f, (tp, fn, fp, n) in list(acc.items()) + [("totale", sum(acc.values()))]:
        if n:
            res[f] = {"f1": f1(tp, fn, fp), "recall": tp / max(tp + fn, 1), "falsi_f": fp / n, "frame": int(n)}
    return res


def kalman(p):
    def run(s):
        kt, dt = KalmanTracker(len(NAMES), p), 1.0 / s["fps"]
        for _, _, d in s["frames"]:
            ts = kt.step(d[d[:, 4] >= p.lo], dt)
            yield np.array([[*t.box, t.conf, t.cls] for t in ts], float).reshape(-1, 6)
    return run


def detector(conf):
    return lambda s: (d[d[:, 4] >= conf] for _, _, d in s["frames"])


def proponi(tr, rq):
    hi = tr.suggest_float("hi", 0.005, 0.8, log=True)
    max_age = tr.suggest_int("max_age", 1, 30)
    return Params(hi=hi, lo=tr.suggest_float("lo", 0.001, hi, log=True), min_hits=tr.suggest_int("min_hits", 1, 8),
                  max_age=max_age, show_age=tr.suggest_int("show_age", 0, max_age),
                  gate=tr.suggest_float("gate", 2.0, 200.0, log=True), v0=tr.suggest_float("v0", 1.0, 500.0, log=True),
                  iou_dup=tr.suggest_float("iou_dup", 0.2, 0.9), vote=tr.suggest_float("vote", 0.0, 0.99),
                  q_pos=rq["q_pos"], q_size=rq["q_size"], r_pos=rq["r_pos"], r_size=rq["r_size"])


def _apri(nome, split):
    return optuna.create_study(study_name=f"{nome}_{split}", direction="maximize", load_if_exists=True,
                               storage=optuna.storages.RDBStorage(  # 24 processi scrivono insieme: aspettare, non fallire
                                   f"sqlite:///{HERE / 'runs' / f'kalman_f1_{split}.db'}",
                                   engine_kwargs={"connect_args": {"timeout": 120}}),
                               sampler=optuna.samplers.TPESampler(multivariate=True))


def _dati(nome, split):
    segs = [s for s in carica(split) if nome == "tutte" or s["fonte"] == nome]
    return segs, json.loads((HERE / "results" / f"rq_{split}.json").read_text())[nome]


def _base(rq):
    return Params(q_pos=rq["q_pos"], q_size=rq["q_size"], r_pos=rq["r_pos"], r_size=rq["r_size"])


def cerca(args):
    """Un processo: fa n prove dello studio. Piu' processi sullo stesso studio condividono le prove nel database."""
    nome, split, n = args
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    segs, rq = _dati(nome, split)
    _apri(nome, split).optimize(lambda tr: metriche(segs, kalman(proponi(tr, rq)))["totale"]["f1"], n_trials=n)


def riassumi(args):
    nome, split = args
    segs, rq = _dati(nome, split)
    st, base = _apri(nome, split), _base(rq)
    best = proponi(optuna.trial.FixedTrial(st.best_params), rq)
    confs = np.round(np.r_[0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, np.arange(0.05, 0.95, 0.05)], 2)
    det = max((metriche(segs, detector(c)) | {"conf": float(c)} for c in confs), key=lambda m: m["totale"]["f1"])
    try:
        imp = optuna.importance.get_param_importances(st)
    except Exception as e:  # noqa: BLE001  l'importanza e' un di piu': senza, i risultati restano validi
        imp = {"errore": str(e)}
    out = {"studio": nome, "split": split, "prove": len(st.trials), "rq": rq, "parametri": asdict(best),
           "kalman_tarato": metriche(segs, kalman(best)), "kalman_partenza": metriche(segs, kalman(base)),
           "detector_miglior_conf": det, "importanza": imp}
    (HERE / "results" / f"kalman_f1_{nome}_{split}.json").write_text(json.dumps(out, indent=1))
    csv_prove(st, nome, split)
    return out


def csv_prove(st, nome, split):
    """Tutte le prove dello studio, una riga per prova: numero, F1, stato, parametri, durata."""
    df = st.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    df.columns = [c.removeprefix("params_") for c in df.columns]
    df["duration"] = df["duration"].dt.total_seconds()
    df.rename(columns={"value": "f1", "duration": "durata_s"}).sort_values("number").to_csv(
        HERE / "results" / f"kalman_f1_{nome}_{split}.csv", index=False, float_format="%.4g")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val"), ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--studi", nargs="*", default=list(STUDI))
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cpu = os.cpu_count()
    for nome in a.studi:                     # la prima prova di ogni studio: i parametri di partenza
        st = _apri(nome, a.split)
        if not len(st.trials):
            b = _base(_dati(nome, a.split)[1])
            st.enqueue_trial({k: getattr(b, k) for k in ("hi", "lo", "min_hits", "max_age", "gate", "v0", "iou_dup",
                                                         "vote")} | {"show_age": b.max_age})
            cerca((nome, a.split, 1))        # qui, prima del parallelo: due processi non prendono la stessa prova
    per = max(cpu // len(a.studi), 1)        # tutti i core, divisi fra gli studi
    lavori = []
    for nome in a.studi:
        fatte = len([t for t in _apri(nome, a.split).trials if t.state.is_finished()])
        resto = max(a.trials - fatte, 0)
        lavori += [(nome, a.split, resto // per + (i < resto % per)) for i in range(per)]
    with Pool(cpu) as pool:
        pool.map(cerca, [l for l in lavori if l[2]], chunksize=1)
        ris = pool.map(riassumi, [(n, a.split) for n in a.studi])
    print(f"{'studio':10s} {'fonte':10s} {'modo':16s}    F1 recall falsi/f")
    for r in ris:
        for modo in ("detector_miglior_conf", "kalman_partenza", "kalman_tarato"):
            for f, m in r[modo].items():
                if isinstance(m, dict):
                    print(f"{r['studio']:10s} {f:10s} {modo:16s} {m['f1']:.3f} {m['recall']:.3f} {m['falsi_f']:.3f}")
        print(f"{r['studio']:10s} parametri: " + " ".join(f"{k}={v:.3g}" for k, v in r["parametri"].items()
                                                          if k != "fps"))


if __name__ == "__main__":
    main()
