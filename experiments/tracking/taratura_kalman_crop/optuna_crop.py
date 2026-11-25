"""Fase 2: Optuna sul tracker con crop (CropTracker: SCAN con P-, un crop 256 con S-), tutti i parametri insieme.
- SCAN: le detection di P- salvate da ../taratura_kalman/detection.py (stesso fotogramma ridotto a 640). TRACK: S- dal
  vivo sul ritaglio; il fotogramma si decodifica solo in TRACK.
- Fissi: R e Q di P- da ../taratura_kalman/rq.py, R di S- da rs.py, max_crops = 1 (S- sta nel budget con un crop).
- Obiettivo: F2 sui droni (recall pesato il doppio della precisione) per fotogramma, IoU >= 0,5, sulle due fonti insieme.
- Vincolo: la quota di aerei ed elicotteri veri con un track drone sopra non supera quella del detector da solo (P- alla
  sua soglia con F2 massimo sui droni). Le prove oltre il vincolo restano nel database ma non possono vincere.
- Prima prova: i migliori parametri di ../taratura_kalman con F2 (studio tutte), con i default del crop.
- --obiettivo f1: F1 sui droni al posto dell'F2 (precisione e recall alla pari), stesso vincolo con il detector alla
  sua soglia con F1 massimo, prima prova dal Kalman tarato con F1; studio e file a parte (kalman_crop_f1_*).
Piu' processi sulla GPU (--proc) lavorano sullo stesso studio, condividendo le prove nel database.
Uscita: results/kalman_crop_<obiettivo>_<split>.json (migliore, metriche per fonte, riferimento del detector, importanza) e .csv
(tutte le prove); le prove in runs/kalman_crop_<obiettivo>_<split>.db (si riprende da li').

    python experiments/tracking/taratura_kalman_crop/optuna_crop.py [--split val] [--trials 1000] [--proc 12] [--obiettivo f1]
"""
import argparse
import json
import time
import multiprocessing as mp

import cv2
import numpy as np
import optuna

from base import FONTI, HERE, KAL, NAMES, TRACK, V, Pigro, conta, dati, rq, sintesi
from src.tracking import Params

_S = None
CONFS = np.round(np.r_[0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, np.arange(0.05, 0.95, 0.05)], 2)


def _s_meno():
    global _S
    if _S is None:
        from ultralytics import YOLO
        _S = YOLO(str(TRACK))
    return _S


class Taratura(V.CropTracker):
    """CropTracker con lo SCAN dalle detection salvate (fit non riduce: le box salvate sono gia' nel fotogramma)."""
    def fit(self, frame):
        return frame, 1.0


def esegui(segs, scan, track, crop):
    """Accumulatori di conta() per fonte, con il tracker con crop."""
    acc = {f: np.zeros(5) for f in FONTI}
    n_track = {f: [0, 0] for f in FONTI}
    S = _s_meno()
    for s in segs:
        shape = cv2.imread(str(s["paths"][0])).shape
        salvate = iter([])

        def detect(images, imgsz, conf):
            if imgsz == 640:
                d = next(salvate)
                return [d[d[:, 4] >= conf]]
            return [r.boxes.data.cpu().numpy() for r in S.predict(images, imgsz=imgsz, conf=conf, verbose=False)]

        ct = Taratura(detect, len(NAMES), scan, scan_imgsz=640, max_crops=1, track=track, **crop)
        dt = 1.0 / s["fps"]
        for (gc, gb, d), p in zip(s["frames"], s["paths"]):
            salvate = iter([d])
            shown, info = ct.step(Pigro(p, shape), dt)
            out = np.array([[*t.box, t.conf, t.cls] for t in shown], float).reshape(-1, 6)
            acc[s["fonte"]] += conta(out, gc, gb)
            n_track[s["fonte"]][0] += info["mode"] == "TRACK"
            n_track[s["fonte"]][1] += 1
    return acc, n_track


def riassunto(acc, n_track):
    tot = sum(acc.values())
    r = {f: sintesi(a) | {"quota_track": n_track[f][0] / max(n_track[f][1], 1)} for f, a in acc.items()}
    r["totale"] = sintesi(tot) | {"quota_track": sum(v[0] for v in n_track.values()) /
                                  max(sum(v[1] for v in n_track.values()), 1)}
    return r


def proponi(tr, base_rq, rs):
    hi = tr.suggest_float("hi", 0.005, 0.8, log=True)
    max_age = tr.suggest_int("max_age", 1, 30)
    scan = Params(hi=hi, lo=tr.suggest_float("lo", 0.001, hi, log=True), min_hits=tr.suggest_int("min_hits", 1, 8),
                  max_age=max_age, show_age=tr.suggest_int("show_age", 0, max_age),
                  gate=tr.suggest_float("gate", 2.0, 200.0, log=True), v0=tr.suggest_float("v0", 1.0, 500.0, log=True),
                  iou_dup=tr.suggest_float("iou_dup", 0.2, 0.9), vote=tr.suggest_float("vote", 0.0, 0.99),
                  q_pos=base_rq["q_pos"], q_size=base_rq["q_size"], r_pos=base_rq["r_pos"], r_size=base_rq["r_size"])
    hi_t = tr.suggest_float("hi_track", 0.001, 0.8, log=True)
    track = {"hi": hi_t, "lo": tr.suggest_float("lo_track", 0.001, hi_t, log=True),
             "r_pos": rs["r_pos"], "r_size": rs["r_size"]}
    crop = {"k": tr.suggest_float("k", 0.5, 6.0), "ctx": tr.suggest_float("ctx", 0.0, 0.8),
            "fill": tr.suggest_float("fill", 0.2, 0.95), "rescan": tr.suggest_int("rescan", 0, 120),   # 0: mai
            "max_side": tr.suggest_int("max_side", 256, 512, step=16)}   # oltre 512 il costo della riduzione non e' misurato
    return scan, track, crop


def _suff(ob):
    return "" if ob == "f2" else f"_{ob}"


def _apri(split, base=None, ob="f2"):
    st = optuna.create_study(
        study_name=f"crop_{split}{_suff(ob)}", direction="maximize", load_if_exists=True,
        storage=optuna.storages.RDBStorage(f"sqlite:///{HERE / 'runs' / f'kalman_crop_{ob}_{split}.db'}",
                                           engine_kwargs={"connect_args": {"timeout": 120}}),
        sampler=optuna.samplers.TPESampler(multivariate=True, constraints_func=lambda t: [t.user_attrs["eccesso"]]))
    if base is not None:
        st.set_user_attr("riferimento", base)
    return st


def obiettivo(tr, segs, base_rq, rs, limite, ob="f2"):
    t0 = time.perf_counter()
    r = riassunto(*esegui(segs, *proponi(tr, base_rq, rs)))
    tr.set_user_attr("eccesso", r["totale"]["aerei_come_drone"] - limite)   # <= 0: vincolo rispettato
    for f, m in r.items():
        for k, v in m.items():
            tr.set_user_attr(f"{f}_{k}", v)
    tr.set_user_attr("secondi", time.perf_counter() - t0)
    return r["totale"][ob]


def cerca(args):
    split, n, ob = args
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    segs, st = dati(split), _apri(split, ob=ob)
    base_rq, rs = rq(split)["tutte"], json.loads((HERE / "results" / f"rs_{split}.json").read_text())["tutte"]
    limite = st.user_attrs["riferimento"]["totale"]["aerei_come_drone"]
    st.optimize(lambda tr: obiettivo(tr, segs, base_rq, rs, limite, ob), n_trials=n)


def detector(segs, ob="f2"):
    """P- da solo alla soglia con F2 (o F1) massimo sui droni: il riferimento del vincolo e del confronto."""
    best = None
    for c in CONFS:
        acc = {f: np.zeros(5) for f in FONTI}
        for s in segs:
            for gc, gb, d in s["frames"]:
                acc[s["fonte"]] += conta(d[d[:, 4] >= c], gc, gb)
        r = {f: sintesi(a) for f, a in acc.items()} | {"totale": sintesi(sum(acc.values())), "conf": float(c)}
        if best is None or r["totale"][ob] > best["totale"][ob]:
            best = r
    return best


def csv_prove(st, split, ob="f2"):
    df = st.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df.columns = [c.removeprefix("params_").removeprefix("user_attrs_") for c in df.columns]
    df.rename(columns={"value": ob}).sort_values("number").to_csv(
        HERE / "results" / f"kalman_crop_{ob}_{split}.csv", index=False, float_format="%.4g")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val"), ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--proc", type=int, default=12, help="processi in parallelo sulla GPU (12: prova di velocita' nel README)")
    ap.add_argument("--obiettivo", choices=["f2", "f1"], default="f2", help="F2 (recall pesato il doppio) o F1 sui droni")
    a = ap.parse_args()
    ob = a.obiettivo
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    segs = dati(a.split)
    st = _apri(a.split, ob=ob)
    if not len(st.trials):                     # riferimento e prima prova, prima del parallelo
        st = _apri(a.split, detector(segs, ob), ob)
        k = json.loads((KAL / "results" / f"kalman_{ob}_tutte_{a.split}.json").read_text())["parametri"]
        st.enqueue_trial({n: k[n] for n in ("hi", "lo", "min_hits", "max_age", "show_age", "gate", "v0", "iou_dup", "vote")}
                         | {"hi_track": k["hi"], "lo_track": k["lo"], "k": 3.0, "ctx": 0.2, "fill": 0.6, "rescan": 30,
                            "max_side": 512})
        cerca((a.split, 1, ob))
    fatte = len([t for t in st.trials if t.state.is_finished()])
    resto = max(a.trials - fatte, 0)
    if resto:
        with mp.get_context("spawn").Pool(a.proc) as pool:   # spawn: il processo principale ha gia' usato la GPU
            pool.map(cerca, [(a.split, resto // a.proc + (i < resto % a.proc), ob) for i in range(a.proc)], chunksize=1)
    st = _apri(a.split, ob=ob)
    ok = [t for t in st.trials if t.state.name == "COMPLETE" and t.user_attrs.get("eccesso", 1) <= 0]
    best = max(ok, key=lambda t: t.value) if ok else None
    try:
        imp = optuna.importance.get_param_importances(st)
    except Exception as e:  # noqa: BLE001  l'importanza e' un di piu'
        imp = {"errore": str(e)}
    rif = st.user_attrs["riferimento"]
    out = {"split": a.split, "prove": len(st.trials), "prove_nel_vincolo": len(ok), "detector": rif,
           "migliore": None if best is None else {"prova": best.number, ob: best.value, "parametri": best.params,
                                                  "metriche": {k: v for k, v in best.user_attrs.items()}},
           "fissi": {"rq": rq(a.split)["tutte"], "rs": json.loads((HERE / "results" / f"rs_{a.split}.json").read_text())["tutte"],
                     "max_crops": 1}, "importanza": imp}
    (HERE / "results" / f"kalman_crop_{ob}_{a.split}.json").write_text(json.dumps(out, indent=1, default=float))
    csv_prove(st, a.split, ob)
    print(f"detector da solo (conf {rif['conf']}): " + "  ".join(
        f"{f} {ob.upper()} {rif[f][ob]:.3f} aerei {rif[f]['aerei_come_drone']:.4f}" for f in (*FONTI, "totale")))
    if best:
        m = best.user_attrs
        print(f"migliore (prova {best.number}): " + "  ".join(
            f"{f} {ob.upper()} {m[f + '_' + ob]:.3f} aerei {m[f + '_aerei_come_drone']:.4f} TRACK {m[f + '_quota_track']:.0%}"
            for f in (*FONTI, "totale")))
        print("parametri: " + " ".join(f"{k}={v:.3g}" for k, v in best.params.items()))


if __name__ == "__main__":
    main()
