"""Quale crop rende di piu' in mAP: per ogni oggetto del val un crop quadrato centrato su di lui (con un errore di
centratura che simula il Kalman), portato a CROP x CROP, con lo zoom scelto perche' l'oggetto occupi T px nel crop.
mAP50 e mAP50-95 per classe e per fascia di grandezza della tesi (lato sqrt(w*h) nel fotogramma originale), contro il
fotogramma intero a 640. Ogni crop vale per il suo oggetto: gli altri oggetti nel crop sono ignorati (le detection su
di loro non contano), le altre detection sono falsi positivi.

    python experiments/tracking/crop_zoom_map/run.py MODELLO.pt [--crop 256] [--target 16 24 32 ...] [--jitter 0.1]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "data/air-detection/all"
NAMES = ["drone", "airplane", "helicopter", "bird"]
# fasce della tesi (vedetta/tools/common.py, BAND_EDGES): potenze di 2 del lato sqrt(w*h) in px
BANDS = {"<8": (0, 8), "8-16": (8, 16), "16-32": (16, 32), "32-64": (32, 64), "64-128": (64, 128),
         "128-256": (128, 256), ">256": (256, 1e9)}
IOUS = np.linspace(0.5, 0.95, 10)


def iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def ap101(tp, conf, npos):
    """AP a 101 punti di recall, come COCO."""
    if npos == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp = tp[np.argsort(-conf, kind="mergesort")]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / npos, ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0 for i in idx]))


def valuta(campioni):
    """campioni: [(gt_cls, gt_xyxy, gt_ignorato, det, fascia)], uno per immagine o crop -> {classe: [AP per soglia]}.
    Con fascia (lo, hi), come COCO: una detection fuori fascia che non trova un GT non conta."""
    out = {}
    for c in range(len(NAMES)):
        tps, confs, npos = [[] for _ in IOUS], [[] for _ in IOUS], 0
        for gc, gb, gig, det, banda in campioni:
            g, ig = gb[gc == c], gig[gc == c]
            npos += int((~ig).sum())
            d = det[det[:, 5] == c]
            if not len(d):
                continue
            d = d[np.argsort(-d[:, 4], kind="mergesort")]
            dz = np.sqrt((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]))
            fuori = ~((dz >= banda[0]) & (dz < banda[1])) if banda else np.zeros(len(d), bool)
            o = np.argsort(ig, kind="mergesort")                     # prima i GT non ignorati
            g, ig = g[o], ig[o]
            M = iou(d[:, :4], g) if len(g) else np.zeros((len(d), 0))
            for t, thr in enumerate(IOUS):
                used = np.zeros(len(g), bool)
                for i in range(len(d)):
                    bj, best = -1, thr
                    for j in range(len(g)):
                        if used[j] or M[i, j] < best:
                            continue
                        if bj > -1 and not ig[bj] and ig[j]:
                            break
                        best, bj = M[i, j], j
                    if bj >= 0:
                        used[bj] = True
                        if not ig[bj]:
                            tps[t].append(1), confs[t].append(d[i, 4])
                    elif not fuori[i]:
                        tps[t].append(0), confs[t].append(d[i, 4])
        out[NAMES[c]] = [ap101(np.array(tps[t], float), np.array(confs[t]), npos) for t in range(len(IOUS))]
    return out


def gt():
    res = {}
    for f in sorted((VAL / "labels/val").glob("*.txt")):
        a = np.loadtxt(f, ndmin=2).reshape(-1, 5)
        c, x, y, w, h = a.T
        res[f.stem] = (c.astype(int), np.stack([(x - w / 2) * 640, (y - h / 2) * 512, (x + w / 2) * 640,
                                                (y + h / 2) * 512], 1))
    return res


def fascia(s):
    return next(k for k, (lo, hi) in BANDS.items() if lo <= s < hi)


def taglia(img, cx, cy, side, crop):
    s = max(int(round(side)), 1)
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    H, W = img.shape[:2]
    tela = np.full((s, s, 3), 114, np.uint8)
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if fx1 > fx0 and fy1 > fy0:
        tela[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = img[fy0:fy1, fx0:fx1]
    interp = cv2.INTER_AREA if s > crop else cv2.INTER_LINEAR
    return cv2.resize(tela, (crop, crop), interpolation=interp), x0, y0, crop / s


def riassunto(ap):
    return {k: [float(np.nanmean(v[:1])), float(np.nanmean(v))] for k, v in ap.items()}   # [AP50, AP50-95]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model"), p.add_argument("--crop", type=int, default=256)
    p.add_argument("--target", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96, 128, 160, 192])
    p.add_argument("--jitter", type=float, default=0.1, help="errore di centratura, frazione del lato del crop")
    p.add_argument("--seed", type=int, default=0), p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = p.parse_args()
    m, G, rng = YOLO(a.model), gt(), np.random.default_rng(a.seed)
    args = dict(conf=0.001, max_det=64, verbose=False)
    res = {"model": a.model, "crop": a.crop, "jitter": a.jitter, "bands": list(BANDS), "fotogramma_640": {},
           "crop_zoom_map": {}, "n": {}}

    # fotogramma intero a 640: per fascia, i GT fuori fascia ignorati e le detection fuori fascia non contate
    stems = list(G)
    det = {}
    for i in range(0, len(stems), a.batch):
        ims = [str(VAL / f"images/val/{s}.jpg") for s in stems[i:i + a.batch]]
        for s, r in zip(stems[i:i + a.batch], m.predict(ims, imgsz=640, **args)):
            det[s] = r.boxes.data.cpu().numpy()
    for b, (lo, hi) in BANDS.items():
        camp = []
        for s, (gc, gb) in G.items():
            sz = np.sqrt((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]))
            camp.append((gc, gb, ~((sz >= lo) & (sz < hi)), det[s], (lo, hi)))
        res["fotogramma_640"][b] = riassunto(valuta(camp))
        print(f"640 {b:8s}", {k: [round(x, 3) for x in v] for k, v in res["fotogramma_640"][b].items()}, flush=True)

    # crop: un campione per oggetto e per target
    ogg = [(s, k) for s, (gc, gb) in G.items() for k in range(len(gc))]
    for T in a.target:
        camp = {b: [] for b in BANDS}
        for i in range(0, len(ogg), a.batch):
            lotto, crops, info = ogg[i:i + a.batch], [], []
            for s, k in lotto:
                gc, gb = G[s]
                bx = gb[k]
                sz = float(np.sqrt((bx[2] - bx[0]) * (bx[3] - bx[1])))
                side = a.crop * sz / T                                  # l'oggetto occupa T px del crop
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                cx, cy = cx + rng.uniform(-1, 1) * a.jitter * side, cy + rng.uniform(-1, 1) * a.jitter * side
                img = cv2.imread(str(VAL / f"images/val/{s}.jpg"))
                c, x0, y0, sc = taglia(img, cx, cy, side, a.crop)
                crops.append(c), info.append((s, k, x0, y0, sc, sz))
            for r, (s, k, x0, y0, sc, sz) in zip(m.predict(crops, imgsz=a.crop, **args), info):
                gc, gb = G[s]
                g = (gb - [x0, y0, x0, y0]) * sc                         # GT nelle coordinate del crop
                ig = np.ones(len(gc), bool)
                ig[k] = False                                            # conta solo l'oggetto del crop
                camp[fascia(sz)].append((gc, g, ig, r.boxes.data.cpu().numpy(), None))
        res["crop_zoom_map"][T] = {b: riassunto(valuta(v)) for b, v in camp.items() if v}
        res["n"][T] = {b: len(v) for b, v in camp.items()}
        print(f"target {T}", {b: {k: [round(x, 3) for x in v] for k, v in d.items()}
                              for b, d in res["crop_zoom_map"][T].items()}, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"crop_map_{Path(a.model).parent.name}_c{a.crop}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
