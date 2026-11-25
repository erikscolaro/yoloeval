"""Confronto sul val di data/tracking, con l'obiettivo della torretta (F2 sui droni per fotogramma, IoU >= 0,5, e quota
di aerei ed elicotteri chiamati drone) fra:
  - P- da solo e S- da solo sul fotogramma intero, ognuno alla sua soglia con F2 massimo sul totale;
  - Kalman con P- (parametri di ../taratura_kalman, studio F2 `tutte`);
  - Kalman con crop (P- in SCAN, S- su un crop in TRACK; parametri di ../taratura_kalman_crop).
Gruppi: per fonte e totale; fotogrammi con 0-1 oggetti (`1ogg`) e 2+ (`2+ogg`); solo i video che non hanno mai 2+
oggetti (`video_1ogg`, il caso del single tracking). Uscita: results/confronto_<split>.json e .txt.

    python experiments/tracking/confronto_p_s_crop/confronto.py [--split val] [--crop 128]

--crop cambia solo il lato del crop di S- (default 256), con gli stessi parametri trovati sul 256; uscita con suffisso
_c<lato>.
--track-model sostituisce S- nel TRACK (uscita con suffisso _<nome cartella>); --mappa-classi porta le sue classi
alle nostre (drone, airplane, helicopter, bird), per esempio 0,0,1,2,3.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "taratura_kalman_crop"))
from base import FONTI, KAL, carica, conta, dati, sintesi  # noqa: E402
import optuna_crop as OC  # noqa: E402
sys.path.insert(0, str(KAL))
import optuna_kalman as OK  # noqa: E402

CONFS = OC.CONFS


def gruppi(s, gb, multi):
    n = "1ogg" if len(gb) <= 1 else "2+ogg"
    out = [(f, c) for f in (s["fonte"], "totale") for c in ("tutti", n)]
    return out + ([] if multi else [(s["fonte"], "video_1ogg"), ("totale", "video_1ogg")])


def accumula(segs, uscite, multi):
    """uscite: per segmento, iteratore delle uscite per fotogramma -> accumulatori conta() per gruppo."""
    acc = {}
    for s, outs in zip(segs, uscite):
        for (gc, gb, _), o in zip(s["frames"], outs):
            for g in gruppi(s, gb, multi[s["seq"]]):
                acc[g] = acc.get(g, np.zeros(5)) + conta(o, gc, gb)
    return acc


def soglia_migliore(segs):
    def f2(c):
        a = sum(conta(d[d[:, 4] >= c], gc, gb) for s in segs for gc, gb, d in s["frames"])
        return sintesi(a)["f2"]
    return float(max(CONFS, key=f2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--track-model")
    ap.add_argument("--mappa-classi")
    a = ap.parse_args()
    suff = ("" if a.crop == 256 else f"_c{a.crop}") + (f"_{Path(a.track_model).parent.name}" if a.track_model else "")
    segs, segs_s = dati(a.split), carica(f"s_{a.split}")
    assert [s["seq"] for s in segs] == [s["seq"] for s in segs_s], "rifare detection.py --modello S-"
    multi = {}
    for s in segs:
        multi[s["seq"]] = multi.get(s["seq"], False) or any(len(gb) >= 2 for _, gb, _ in s["frames"])

    res, conf = {}, {}
    for nome, ss in (("P-", segs), ("S-", segs_s)):
        c = conf[nome] = soglia_migliore(ss)
        res[nome] = accumula(ss, [(d[d[:, 4] >= c] for *_, d in s["frames"]) for s in ss], multi)
        print(nome, "conf", c, flush=True)

    kj = json.loads((KAL / "results" / f"kalman_f2_tutte_{a.split}.json").read_text())
    rq = OC.rq(a.split)["tutte"]
    pk = OK.proponi(optuna.trial.FixedTrial({k: v for k, v in kj["parametri"].items() if k in
                                             ("hi", "lo", "min_hits", "max_age", "show_age", "gate", "v0", "iou_dup", "vote")}), rq)
    run = OK.kalman(pk)
    res["Kalman P-"] = accumula(segs, [run(s) for s in segs], multi)
    print("Kalman fatto", flush=True)

    cj = json.loads((OC.HERE / "results" / f"kalman_crop_f2_{a.split}.json").read_text())
    rs = json.loads((OC.HERE / "results" / f"rs_{a.split}.json").read_text())["tutte"]
    scan, track, crop = OC.proponi(optuna.trial.FixedTrial(cj["migliore"]["parametri"]), rq, rs)
    res["Kalman crop"], quota = crop_per_fotogramma(segs, scan, track, crop | {"crop": a.crop}, multi,
                                                    a.track_model, a.mappa_classi)

    righe = [f"{'gruppo':26s} {'modo':12s} {'F2':>6s} {'recall':>6s} {'prec':>6s} {'aerei->drone':>12s} {'droni':>6s}"]
    out = {"conf": conf, "quota_track": quota, "gruppi": {}}
    for g in sorted(res["P-"], key=lambda g: (g[0] != "totale", g[0], g[1])):
        for modo in res:
            m = sintesi(res[modo][g]) | {"droni": int(res[modo][g][0] + res[modo][g][1])}
            out["gruppi"].setdefault(f"{g[0]} {g[1]}", {})[modo] = m
            righe.append(f"{g[0] + ' ' + g[1]:26s} {modo:12s} {m['f2']:6.3f} {m['recall']:6.3f} {m['precisione']:6.3f} "
                         f"{m['aerei_come_drone']:12.4f} {m['droni']:6d}")
    righe.append(f"soglie: P- {conf['P-']}, S- {conf['S-']}; Kalman crop in TRACK: {quota}")
    txt = "\n".join(righe)
    print(txt)
    (HERE / "results" / f"confronto_{a.split}{suff}.txt").write_text(txt + "\n")
    (HERE / "results" / f"confronto_{a.split}{suff}.json").write_text(json.dumps(out, indent=1))


def crop_per_fotogramma(segs, scan, track, crop, multi, track_model=None, mappa=None):
    """Come OC.esegui, ma accumula per gruppo."""
    import cv2
    acc, tr = {}, {}
    if track_model:
        from ultralytics import YOLO
        S = YOLO(track_model)
    else:
        S = OC._s_meno()
    m = np.array([int(x) for x in mappa.split(",")], float) if mappa else None

    def rimappa(d):
        if m is not None and len(d):
            d[:, 5] = m[d[:, 5].astype(int)]
        return d
    for s in segs:
        shape = cv2.imread(str(s["paths"][0])).shape
        box = {"d": None}

        def detect(images, imgsz, conf):
            if imgsz == 640:
                d = box["d"]
                return [d[d[:, 4] >= conf]]
            return [rimappa(r.boxes.data.cpu().numpy()) for r in S.predict(images, imgsz=imgsz, conf=conf, verbose=False)]

        ct = OC.Taratura(detect, len(OC.NAMES), scan, scan_imgsz=640, max_crops=1, track=track, **crop)
        for (gc, gb, d), p in zip(s["frames"], s["paths"]):
            box["d"] = d
            shown, info = ct.step(OC.Pigro(p, shape), 1.0 / s["fps"])
            o = np.array([[*t.box, t.conf, t.cls] for t in shown], float).reshape(-1, 6)
            for g in gruppi(s, gb, multi[s["seq"]]):
                acc[g] = acc.get(g, np.zeros(5)) + conta(o, gc, gb)
                n = tr.setdefault(f"{g[0]} {g[1]}", [0, 0])
                n[0] += info["mode"] == "TRACK"
                n[1] += 1
    return acc, {k: round(v[0] / v[1], 3) for k, v in tr.items()}


if __name__ == "__main__":
    main()
