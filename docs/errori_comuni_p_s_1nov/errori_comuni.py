# Errori comuni a pico (59d8534a) e small (9ada3b33) sul val di air-detection/all, e loro fogli.
#   .venv/bin/python reports/errori_comuni_p_s_1nov/errori_comuni.py   (dalla radice di yoloeval)
# Stesso abbinamento della vecchia analisi (.old/train_26ott/analisi/errori_val_26ott): conf 0,25, IoU 0,45, senza
# guardare la classe, per confidenza decrescente; testa one-to-one (nms=False) come negli export.
# Errore comune su una label: sbagliata da entrambi (mancata o con la classe sbagliata). Falso positivo comune: un FP
# del pico e uno dello small con IoU >= 0,45 fra loro.
import csv, pickle
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
D = ROOT / "data/air-detection/all"
W = ROOT / "artifacts/weights"
MODELS = {"pico": "yolo26p_D126_air_detection_all_e300_59d8534a", "small": "yolo26s_D126_air_detection_all_e300_9ada3b33"}
NAMES = ["drone", "airplane", "helicopter", "bird"]
CONF, LOW, IOU = 0.25, 0.05, 0.45
PER, COLS, SIZE, HEAD = 24, 4, 260, 44
ORDER = ["bird", "drone", "airplane", "helicopter"]
OUT = HERE / "fogli"


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    ar = lambda z: (z[:, 2:] - z[:, :2]).prod(1)
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def labels(im, Wd=640, H=512):
    f = D / "labels/val" / (Path(im).stem + ".txt")
    g = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.zeros((0, 5))
    return g[:, 0].astype(int), np.c_[(g[:, 1] - g[:, 3] / 2) * Wd, (g[:, 2] - g[:, 4] / 2) * H,
                                      (g[:, 1] + g[:, 3] / 2) * Wd, (g[:, 2] + g[:, 4] / 2) * H]


# predizioni con conf >= LOW su tutto il val, in cache
cache = HERE / "pred.pkl"
if cache.exists():
    pred = pickle.load(open(cache, "rb"))
else:
    pred = {}
    for k, d in MODELS.items():
        for r in YOLO(W / d / "best.pt").predict(str(D / "images/val"), conf=LOW, imgsz=640, nms=False, verbose=False,
                                                 stream=True, batch=64):
            b = r.boxes
            pred[k, Path(r.path).name] = (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())
    pickle.dump(pred, open(cache, "wb"))


def match(gb, pb, ps):
    """Abbinamento greedy per confidenza: {label: (predizione, IoU)} e insieme delle predizioni abbinate."""
    M = iou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
    gm, pm = {}, set()
    for j in np.argsort(-ps):
        cand = [(M[i, j], i) for i in range(len(gb)) if i not in gm and M[i, j] >= IOU]
        if cand:
            v, i = max(cand); gm[i] = (j, v); pm.add(j)
    return gm, pm, M


man = {Path(r["immagine"]).name: r for r in csv.DictReader(open(D / "manifest.csv")) if r["split"] == "val"}
side = lambda b: float(np.sqrt((b[2] - b[0]) * (b[3] - b[1])))
rows, tot = [], Counter()
for im in sorted(p.name for p in (D / "images/val").glob("*.jpg")):
    gc, gb = labels(im)
    st, fp = {}, {}
    for k in MODELS:
        pb, pc, ps = pred[k, im]
        keep = ps >= CONF
        pb, pc, ps = pb[keep], pc[keep], ps[keep]
        gm, pm, M = match(gb, pb, ps)
        s = []
        for i in range(len(gb)):
            if i not in gm:
                s.append(("mancato", "", "", round(float(M[i].max()) if len(pb) else 0, 3)))
            elif pc[gm[i][0]] != gc[i]:
                j = gm[i][0]; s.append(("classe_sbagliata", NAMES[pc[j]], round(float(ps[j]), 3), round(float(gm[i][1]), 3)))
            else:
                s.append(None)
            tot[k, "label_err"] += s[-1] is not None
        st[k] = s
        fp[k] = [(pb[j], NAMES[pc[j]], round(float(ps[j]), 3)) for j in range(len(pb)) if j not in pm]
        tot[k, "fp"] += len(fp[k])
    mt = man.get(im, {})
    meta = {c: mt.get(c, "") for c in ("fonte", "scenario", "tipo", "sessione")}
    for i in range(len(gb)):
        a, b = st["pico"][i], st["small"][i]
        if a and b:
            err = a[0] if a[0] == b[0] else "misto"
            rows.append(dict(immagine=im, errore=err, classe=NAMES[gc[i]], pico=a[1] or a[0], conf_pico=a[2], iou_pico=a[3],
                             small=b[1] or b[0], conf_small=b[2], iou_small=b[3],
                             **dict(zip(("x1", "y1", "x2", "y2"), np.round(gb[i], 1))), lato_px=round(side(gb[i]), 1), **meta))
    if fp["pico"] and fp["small"]:
        M = iou(np.array([f[0] for f in fp["pico"]]), np.array([f[0] for f in fp["small"]]))
        used = set()
        for a in np.argsort([-f[2] for f in fp["pico"]]):
            c = [(M[a, j], j) for j in range(M.shape[1]) if j not in used and M[a, j] >= IOU]
            if c:
                v, j = max(c); used.add(j)
                (q, ca, sa), (_, cb, sb) = fp["pico"][a], fp["small"][j]
                rows.append(dict(immagine=im, errore="falso_positivo", classe=ca, pico=ca, conf_pico=sa, iou_pico="",
                                 small=cb, conf_small=sb, iou_small=round(float(v), 3),
                                 **dict(zip(("x1", "y1", "x2", "y2"), np.round(q, 1))), lato_px=round(side(q), 1), **meta))

FIELDS = list(rows[0])
with open(HERE / "errori_comuni.csv", "w", newline="") as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
n_lab = sum(r["errore"] != "falso_positivo" for r in rows)
n_fp = len(rows) - n_lab
print(f"label sbagliate da entrambi: {n_lab} (pico {tot['pico', 'label_err']}, small {tot['small', 'label_err']})")
print(f"falsi positivi comuni: {n_fp} (pico {tot['pico', 'fp']}, small {tot['small', 'fp']})")


def panel(im, k, b, err, x0, y0, x1, y1):
    t = cv2.imread(str(D / "images/val" / im))[y0:y1, x0:x1]
    z = SIZE / max(x1 - x0, y1 - y0)
    t = cv2.resize(t, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST if z > 2 else cv2.INTER_AREA)
    P = lambda q, d=0: ((int((q[0] - x0) * z) - d, int((q[1] - y0) * z) - d), (int((q[2] - x0) * z) + d, int((q[3] - y0) * z) + d))
    tag = lambda s, p, col: cv2.putText(t, s, (p[0], max(10, p[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    for c, q in zip(*labels(im)):
        p0, p1 = P(q, 3); cv2.rectangle(t, p0, p1, (0, 200, 0), 1); tag(NAMES[c], p0, (0, 200, 0))
    pb, pc, ps = pred[k, im]
    for j in np.argsort(ps):
        p0, p1 = P(pb[j], 1)
        col = (150, 150, 150) if ps[j] < CONF else (0, 0, 255)
        cv2.rectangle(t, p0, p1, col, 1)
        tag(f"{NAMES[pc[j]]} {ps[j]:.2f}", (p0[0], p1[1] + 13), col)
    if err != "falso_positivo":
        cv2.rectangle(t, *P(b, 6), (0, 255, 255), 1)
    t = cv2.copyMakeBorder(t, 0, SIZE - t.shape[0], 0, SIZE - t.shape[1], cv2.BORDER_CONSTANT, value=0)
    cv2.putText(t, k, (3, SIZE - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def tile(r, n):
    b = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    sz = max(5 * max(b[2] - b[0], b[3] - b[1]), 120)
    x0, y0 = int(np.clip(cx - sz / 2, 0, max(0, 640 - sz))), int(np.clip(cy - sz / 2, 0, max(0, 512 - sz)))
    x1, y1 = int(min(640, x0 + sz)), int(min(512, y0 + sz))
    t = np.hstack([panel(r["immagine"], k, b, r["errore"], x0, y0, x1, y1) for k in MODELS])
    t = cv2.copyMakeBorder(t, HEAD, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    what = f"FP {r['pico']} / {r['small']}" if r["errore"] == "falso_positivo" else f"{r['classe']}: pico {r['pico']}, small {r['small']}"
    lines = [f"#{n} {what}", f"lato {float(r['lato_px']):.0f}px  conf {r['conf_pico'] or '-'} / {r['conf_small'] or '-'}",
             f"{r['fonte']} {r['scenario']} {r['tipo']}"[:60]]
    for k, s in enumerate(lines):
        cv2.putText(t, s, (3, 13 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))


groups = defaultdict(list)
for r in rows:
    groups[r["errore"], r["classe"]].append(r)
OUT.mkdir(exist_ok=True)
idx = csv.writer(open(OUT / "indice.csv", "w", newline=""))
idx.writerow(["foglio", "n", *FIELDS])
key = lambda g: (["mancato", "classe_sbagliata", "misto", "falso_positivo"].index(g[0]), ORDER.index(g[1]))
for g in sorted(groups, key=key):
    rs = sorted(groups[g], key=lambda r: (r["pico"], float(r["lato_px"])))  # per classe predetta, poi dal più piccolo
    for s in range(-(-len(rs) // PER)):
        name = f"{g[0]}_{g[1]}_{s + 1:02d}.jpg"
        tiles = []
        for n, r in enumerate(rs[s * PER:(s + 1) * PER], 1):
            tiles.append(tile(r, n)); idx.writerow([name, n, *r.values()])
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % COLS)
        cv2.imwrite(str(OUT / name), np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(*g, len(rs))
