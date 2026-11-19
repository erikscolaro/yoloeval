"""Val e test informali di p, n, s: mAP per classe (Ultralytics) e per fascia di grandezza (stile COCO, ignore fuori fascia).
Impostazioni: one-to-one (nms=False), imgsz 640, max_det 64, conf 0.001. Uscita: JSON in OUT."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path.home() / "Desktop/yoloeval"
DATA = ROOT / "data/air-detection/all/dataset.yaml"
W = ROOT / "artifacts/weights"
MODELS = {"p": "yolo26p_D126_air_detection_all_e300_59d8534a", "n": "yolo26n_D126_air_detection_all_e300_9dca8d15",
          "s": "yolo26s_D126_air_detection_all_e300_9ada3b33"}
NAMES = ["drone", "airplane", "helicopter", "bird"]
BANDS = {"all": (0, 1e9), "<16": (0, 16), "16-32": (16, 32), "32-64": (32, 64), "64-96": (64, 96), ">96": (96, 1e9)}
IOUS = np.linspace(0.5, 0.95, 10)
IMW, IMH = 640, 512
ARGS = dict(imgsz=640, nms=False, max_det=64, conf=0.001, batch=32, verbose=False)
OUT = Path(sys.argv[1])


def gt(split):
    res = {}
    for f in sorted((ROOT / f"data/air-detection/all/labels/{split}").glob("*.txt")):
        a = np.loadtxt(f, ndmin=2).reshape(-1, 5)
        c, x, y, w, h = a.T
        xyxy = np.stack([(x - w / 2) * IMW, (y - h / 2) * IMH, (x + w / 2) * IMW, (y + h / 2) * IMH], 1)
        res[f.stem] = (c.astype(int), xyxy)
    return res


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def size(z):
    return np.sqrt(np.clip(z[:, 2] - z[:, 0], 0, None) * np.clip(z[:, 3] - z[:, 1], 0, None))


def ap101(tp, conf, npos):
    if npos == 0:
        return np.nan
    if len(tp) == 0:
        return 0.0
    o = np.argsort(-conf, kind="mergesort")
    tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / npos, ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0 for i in idx]))


def band_eval(G, P):
    """{band: {class: [AP per soglia IoU]}} e numero di GT per band e classe."""
    out, npos_all = {}, {}
    for bname, (lo, hi) in BANDS.items():
        out[bname], npos_all[bname] = {}, {}
        for c in range(len(NAMES)):
            tps = [[] for _ in IOUS]
            confs = [[] for _ in IOUS]
            npos = 0
            for k, (gc, gb) in G.items():
                g = gb[gc == c]
                pc, pb, ps = P.get(k, (np.zeros(0, int), np.zeros((0, 4)), np.zeros(0)))
                m = pc == c
                d, ds = pb[m], ps[m]
                gig = ~((size(g) >= lo) & (size(g) < hi))
                npos += int((~gig).sum())
                if len(d) == 0:
                    continue
                o = np.argsort(-ds, kind="mergesort")
                d, ds = d[o], ds[o]
                go = np.argsort(gig, kind="mergesort")  # non ignorati prima
                g, gig = g[go], gig[go]
                M = iou(d, g) if len(g) else np.zeros((len(d), 0))
                dout = ~((size(d) >= lo) & (size(d) < hi))
                for t, thr in enumerate(IOUS):
                    used = np.zeros(len(g), bool)
                    for i in range(len(d)):
                        best, bj = thr, -1
                        for j in range(len(g)):
                            if used[j]:
                                continue
                            if bj > -1 and not gig[bj] and gig[j]:
                                break
                            if M[i, j] < best:
                                continue
                            best, bj = M[i, j], j
                        if bj >= 0:
                            used[bj] = True
                            if gig[bj]:
                                continue  # match con GT ignorato: det ignorato
                            tps[t].append(1); confs[t].append(ds[i])
                        elif not dout[i]:
                            tps[t].append(0); confs[t].append(ds[i])
            out[bname][NAMES[c]] = [ap101(np.array(tps[t], float), np.array(confs[t]), npos) for t in range(len(IOUS))]
            npos_all[bname][NAMES[c]] = npos
    return out, npos_all


def main():
    res = {}
    for split in ("val", "test"):
        G = gt(split)
        imgs = ROOT / f"data/air-detection/all/images/{split}"
        for mk, mdir in MODELS.items():
            model = YOLO(str(W / mdir / "best.pt"))
            with tempfile.TemporaryDirectory() as tmp:
                r = model.val(data=str(DATA), split=split, plots=False, project=tmp, name="v", exist_ok=True, **ARGS)
            ul = {"all": [float(r.box.map50), float(r.box.map)]}
            for i, ci in enumerate(r.box.ap_class_index):
                ul[NAMES[int(ci)]] = [float(r.box.ap50[i]), float(r.box.ap[i])]
            P = {}
            for p in YOLO(str(W / mdir / "best.pt")).predict(source=str(imgs), stream=True, **ARGS):
                b = p.boxes
                P[Path(p.path).stem] = (b.cls.cpu().numpy().astype(int), b.xyxy.cpu().numpy(), b.conf.cpu().numpy())
            be, npos = band_eval(G, P)
            res[f"{mk}_{split}"] = {"ultralytics": ul, "bands": be, "npos": npos}
            print(mk, split, ul["all"], flush=True)
            OUT.write_text(json.dumps(res, indent=1))
    labels = ROOT / "data/air-detection/all/labels"
    tc = labels / "test.cache"
    if tc.exists():
        tc.unlink()


if __name__ == "__main__":
    main()
