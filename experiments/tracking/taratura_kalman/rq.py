"""Passo 2 della taratura: R e Q del Kalman dai dati (detection salvate da detection.py), per fonte e per classe.
- R: scarto fra detection e verita' (IoU >= 0,5, stessa classe, conf >= 0,3), diviso l'altezza vera, come lo usa il filtro
  (src/tracking.py: deviazione = r * altezza). Deviazione robusta (1,4826 * MAD); r_pos da cx e cy, r_size da w e h.
- Q: per ogni valore di q_pos (q_size = q_pos / 2, come nei default) il filtro segue il solo oggetto vero, aggiornato con
  la sua detection abbinata; per ogni aggiornamento il NIS (innovazione normalizzata, 4 gradi di liberta'). Si sceglie il
  q_pos con NIS medio piu' vicino a 4: il filtro allora "crede" all'errore che fa davvero. Solo segmenti con un oggetto.
Uscita: results/rq_<split>.json.

    python experiments/tracking/taratura_kalman/rq.py [--split val]
"""
import argparse
import json

import numpy as np

from comune import FONTI, HERE, NAMES, abbina, carica, residui
from src.tracking import KalmanTracker, Params, xyxy2cxcywh

QS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def robusta(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def stima_r(segs):
    res = [r for s in segs for gc, gb, d in s["frames"] for r in residui(d[d[:, 4] >= 0.3], gc, gb)]
    out = {}
    for nome, sel in [("tutte", res)] + [(n, [r for r in res if r[0] == c]) for c, n in enumerate(NAMES)]:
        if len(sel) < 50:
            continue
        e = np.array([r[1] for r in sel])
        out[nome] = {"r_pos": robusta(e[:, :2].ravel()), "r_size": robusta(e[:, 2:].ravel()), "n": len(sel)}
    return out


def nis_medio(segs, q, r_pos, r_size):
    nis = []
    for s in segs:
        p = Params(q_pos=q, q_size=q / 2, r_pos=r_pos, r_size=r_size)
        kt, t, dt = KalmanTracker(len(NAMES), p), None, 1.0 / s["fps"]
        for gc, gb, d in s["frames"]:
            if len(gb) != 1:                    # solo fotogrammi con un oggetto: niente ambiguita' di identita'
                t = None
                continue
            m = abbina(d[d[:, 4] >= 0.3], gc, gb)
            if t is not None:
                kt._predict(t, dt)
            if not m:
                continue
            det = d[m[0][0]]
            if t is None:
                kt.tracks = []
                kt._new(det)
                t = kt.tracks[0]
                continue
            z = xyxy2cxcywh(det[:4])
            S = kt.H @ t.P @ kt.H.T + kt._R(max(z[3], 1.0))
            y = z - kt.H @ t.x
            nis.append(float(y @ np.linalg.solve(S, y)))
            kt._update(t, det)
    return float(np.mean(nis)) if nis else float("nan"), len(nis)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    dati = carica(a.split)
    out = {}
    for fonte in FONTI + ("tutte",):
        segs = [s for s in dati if fonte == "tutte" or s["fonte"] == fonte]
        R = stima_r(segs)
        r = R["tutte"]
        curva = {q: nis_medio(segs, q, r["r_pos"], r["r_size"]) for q in QS}
        q = min(QS, key=lambda v: abs(np.log(curva[v][0] / 4)))
        out[fonte] = {"R": R, "nis": {str(k): v[0] for k, v in curva.items()}, "q_pos": q, "q_size": q / 2,
                      "r_pos": r["r_pos"], "r_size": r["r_size"]}
        print(f"{fonte:10s} r_pos {r['r_pos']:.4f} r_size {r['r_size']:.4f} q_pos {q}  NIS: "
              + " ".join(f"{k}:{v[0]:.1f}" for k, v in curva.items()), flush=True)
        for c, v in R.items():
            print(f"    R {c:10s} r_pos {v['r_pos']:.4f} r_size {v['r_size']:.4f} (n {v['n']})")
    (HERE / "results" / f"rq_{a.split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
