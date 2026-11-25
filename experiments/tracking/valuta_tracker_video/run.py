"""Valutazione del tracker sui video di val (o test) di ../video_val_test, tre modi sugli stessi fotogrammi:
  detector  solo lo SCAN a ogni fotogramma (P-, fotogramma ridotto con INTER_AREA a 640 di lato lungo come
            CropTracker.fit: Svanstrom 640x512, Anti-UAV 1920x1080 -> 640x360 + bordo a 640x384), detection con conf >= lo
  kalman    le stesse detection nel KalmanTracker (src/tracking.py): i track mostrati
  crop      CropTracker (src/crop_tracker.py): SCAN con P-, ritagli 256 con S-; i track mostrati
Le sequenze si dividono nei buchi (frame mancanti). Risultati separati per fonte (svanstrom, anti_uav) e totali:
  mAP50 e mAP50-95 per classe e per fascia (lato sqrt(w*h) nel fotogramma originale), con ../crop_zoom_map/run.py;
  anche separate per fotogrammi con un oggetto o nessuno (`1ogg`, il caso del deploy) e con 2+ oggetti (`2+ogg`);
  punto di lavoro (detection con conf >= hi, track mostrati): recall, falsi allarmi per fotogramma, fotogrammi con
  oggetto e senza uscita; per crop la quota di fotogrammi in TRACK. Associazione senza classe, IoU >= 0,5.

    python experiments/tracking/valuta_tracker_video/run.py [--split val] [--max-seq N] [--tag nome]
        [--p hi=0.4 q_pos=4 ...] [--ct rescan=30 k=3 ...]
--p cambia i Params del Kalman, --track-p quelli dei fotogrammi TRACK (soglie e R di S-), --ct gli argomenti di CropTracker: servono alla taratura. --det-conf e' la soglia del
punto di lavoro del detector (default hi): per un confronto onesto, la sua soglia con F1 massimo.
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.crop_tracker import CropTracker  # noqa: E402
from src.tracking import KalmanTracker, Params, iou  # noqa: E402

spec = importlib.util.spec_from_file_location("czm", HERE.parent / "crop_zoom_map/run.py")
czm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(czm)

W = ROOT / "artifacts/weights"
SCAN = W / "yolo26w0.125_D101_air_detection_all_e300_e7d00a5f/best.pt"     # P-
TRACK = W / "yolo26w0.5_D101_air_detection_all_e300_21d2b4bd/best.pt"     # S-
MODI = ("detector", "kalman", "crop")


def coppie(lista):
    out = {}
    for s in lista or []:
        k, v = s.split("=")
        out[k] = type(getattr(Params(), k, 0.0))(v) if hasattr(Params(), k) else (int(v) if v.isdigit() else float(v))
    return out


def segmenti(seq):
    """Fotogrammi consecutivi della sequenza; un buco apre un segmento nuovo."""
    seg = []
    for p in sorted((seq / "images").glob("*.jpg"), key=lambda q: int(q.stem)):
        if seg and int(p.stem) != int(seg[-1].stem) + 1:
            yield seg
            seg = []
        seg.append(p)
    if seg:
        yield seg


def verita(img_path, Wd, Hd):
    f = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    a = np.loadtxt(f, ndmin=2).reshape(-1, 5) if f.exists() and f.stat().st_size else np.zeros((0, 5))
    c, x, y, w, h = a.T
    return c.astype(int), np.stack([(x - w / 2) * Wd, (y - h / 2) * Hd, (x + w / 2) * Wd, (y + h / 2) * Hd], 1)


def da_track(ts):
    return np.array([[*t.box, t.conf, t.cls] for t in ts], float).reshape(-1, 6)


def punto(det, gb):
    """tp, fn, fp senza classe, IoU >= 0,5, abbinamento goloso per confidenza."""
    if not len(gb):
        return 0, 0, len(det)
    if not len(det):
        return 0, len(gb), 0
    M, used, tp = iou(det[np.argsort(-det[:, 4]), :4], gb), np.zeros(len(gb), bool), 0
    for i in range(len(M)):
        j = np.where(~used & (M[i] >= 0.5))[0]
        if len(j):
            used[j[M[i, j].argmax()]] = True
            tp += 1
    return tp, len(gb) - tp, len(det) - tp


def metriche(frames, det_conf):
    out = {}
    for modo in MODI:
        if modo not in frames[0][2]:
            continue
        r = {"tutto": czm.riassunto(czm.valuta([(gc, gb, np.zeros(len(gc), bool), d[modo], None)
                                                for gc, gb, d in frames]))}
        for b, (lo, hi_b) in czm.BANDS.items():
            camp = []
            for gc, gb, d in frames:
                sz = np.sqrt((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]))
                camp.append((gc, gb, ~((sz >= lo) & (sz < hi_b)), d[modo], (lo, hi_b)))
            r[b] = czm.riassunto(czm.valuta(camp))
        tp = fn = fp = vuoti = con = 0
        for gc, gb, d in frames:
            x = d[modo] if modo != "detector" else d[modo][d[modo][:, 4] >= det_conf]
            a, b_, c = punto(x, gb)
            tp, fn, fp = tp + a, fn + b_, fp + c
            con += len(gb) > 0
            vuoti += len(gb) > 0 and a == 0
        media = np.nanmean([v for v in r["tutto"].values()], axis=0)
        r["sintesi"] = {"mAP50": float(media[0]), "mAP50-95": float(media[1]), "recall": tp / max(tp + fn, 1),
                        "falsi_per_fotogramma": fp / len(frames), "fotogrammi_persi": vuoti / max(con, 1),
                        "fotogrammi": len(frames)}
        out[modo] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val"), ap.add_argument("--max-seq", type=int, default=0)
    ap.add_argument("--tag", default="base"), ap.add_argument("--p", nargs="*"), ap.add_argument("--ct", nargs="*")
    ap.add_argument("--track-p", nargs="*", help="Params diversi nei fotogrammi TRACK (S-), per esempio hi=0.1 lo=0.02")
    ap.add_argument("--seq", nargs="*", help="solo queste sequenze (nomi delle cartelle)")
    ap.add_argument("--det-conf", type=float, help="soglia del punto di lavoro del detector (default: hi del Kalman)")
    ap.add_argument("--scan", default=str(SCAN)), ap.add_argument("--track", default=str(TRACK))
    a = ap.parse_args()
    par, ctk = Params(**coppie(a.p)), coppie(a.ct)
    if a.track_p:
        ctk["track"] = coppie(a.track_p)
    det_conf = par.hi if a.det_conf is None else a.det_conf
    P, S = YOLO(a.scan), YOLO(a.track)
    crop = int(ctk.get("crop", 256))

    def rileva(images, imgsz, conf):
        m = S if imgsz == crop else P
        return [r.boxes.data.cpu().numpy() for r in m.predict(images, imgsz=imgsz, conf=conf, verbose=False)]

    per_fonte, n_track = defaultdict(list), defaultdict(lambda: [0, 0])
    seqs = sorted(p for p in (ROOT / "data/tracking" / a.split).iterdir() if (p / "seq.json").exists())
    visti = defaultdict(int)
    if a.seq:
        seqs = [p for p in seqs if p.name in a.seq]
    for seq in seqs:
        info = json.loads((seq / "seq.json").read_text())
        fonte, dt = info["fonte"], 1.0 / info["fps"]
        if a.max_seq and visti[fonte] >= a.max_seq:
            continue
        visti[fonte] += 1
        for seg in segmenti(seq):
            kt = KalmanTracker(len(czm.NAMES), par)
            ct = CropTracker(rileva, len(czm.NAMES), par, scan_imgsz=640, **ctk)
            for img_path in seg:
                fr = cv2.imread(str(img_path))
                Hd, Wd = fr.shape[:2]
                gc, gb = verita(img_path, Wd, Hd)
                small, r = ct.fit(fr)
                det = rileva([small], 640, min(par.lo, det_conf))[0].copy()
                det[:, :4] /= r
                shown, inf = ct.step(fr, dt)
                n_track[fonte][0] += inf["mode"] == "TRACK"
                n_track[fonte][1] += 1
                per_fonte[fonte].append((gc, gb, {"in_track": inf["mode"] == "TRACK", "detector": det, "kalman": da_track(kt.step(det[det[:, 4] >= par.lo], dt)),
                                                  "crop": da_track(shown)}))
        print(f"{seq.name:32s} {fonte:10s} {sum(len(s) for s in segmenti(seq)):5d} fotogrammi", flush=True)
    res = {"split": a.split, "params": vars(par), "crop_tracker": ctk, "scan": a.scan, "track": a.track,
           "det_conf": det_conf, "fonti": {}}
    tutti = [f for v in per_fonte.values() for f in v]
    gruppi = list(per_fonte.items()) + [("totale", tutti)]
    for nome, frames in list(gruppi):     # il caso del deploy (un oggetto, o nessuno) separato da 2+ oggetti
        for suff, sel in (("1ogg", lambda gb: len(gb) <= 1), ("2+ogg", lambda gb: len(gb) >= 2)):
            sub = [f for f in frames if sel(f[1])]
            if sub and len(sub) < len(frames):
                gruppi.append((f"{nome} {suff}", sub))
    for fonte, frames in gruppi:
        res["fonti"][fonte] = metriche(frames, det_conf) | {"fotogrammi": len(frames),
                                                            "quota_track": np.mean([d["in_track"] for *_, d in frames])}
    (HERE / "results" / f"{a.split}_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n{'fonte':10s} {'modo':9s} {'mAP50':>6s} {'50-95':>6s} {'recall':>6s} {'falsi/f':>7s} {'persi':>6s}")
    for fonte, r in res["fonti"].items():
        for modo in MODI:
            s = r[modo]["sintesi"]
            print(f"{fonte:10s} {modo:9s} {s['mAP50']:6.3f} {s['mAP50-95']:6.3f} {s['recall']:6.3f} "
                  f"{s['falsi_per_fotogramma']:7.3f} {s['fotogrammi_persi']:6.3f}")
        print(f"{fonte:10s} fotogrammi in TRACK: {r['quota_track']:.1%} (su {r['fotogrammi']})")


if __name__ == "__main__":
    main()
