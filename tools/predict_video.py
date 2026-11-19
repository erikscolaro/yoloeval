"""Predict su un video o una cartella di video, con le box dal vivo e in sovrimpressione la risoluzione del video,
l'ingresso del modello, il tempo di inferenza e gli FPS veri (lettura + inferenza + disegno). q chiude il video,
Esc tutto. I video annotati vanno in OUT/<nome del video>.mp4.

    python -m tools.predict_video MODELLO.pt SORGENTE [--conf 0.25] [--imgsz 640 | --imgsz 736,1280] [--mode fit|crop]
                                  [--out DIR] [--no-show]

--imgsz: un numero e' il lato lungo (16:9 a 640 entra 640x384); ALTEZZA,LARGHEZZA e' l'ingresso di deploy (480p 480,864,
720p 736,1280, 1080p 1088,1920).
--mode fit (default): il fotogramma si scala per stare nell'ingresso e si completa a multipli di 32 (letterbox).
--mode crop: dal centro del fotogramma si ritaglia un rettangolo grande quanto l'ingresso, a risoluzione nativa e senza
scalare (gli oggetti restano grandi in pixel come nel training); se il fotogramma e' piu' piccolo si completa con il
grigio del letterbox. Sul video si vede il riquadro del ritaglio.
--kalman: tracker con filtro di Kalman (src/tracking.py) sulle detection: classe a voto sulla storia del track, box
filtrata, detection deboli (--lo <= conf < --hi) tenute solo se confermano un track. Le box del tracker sono spesse con
l'ID, quelle grezze del modello sottili e grigie. Con --kalman il modello gira a conf --lo, e --conf non conta.
--kalman-crop: versione con il crop adattativo (src/crop_tracker.py): il fotogramma intero (--imgsz, fit) solo in SCAN,
poi un crop --crop-size per ogni track centrato sulla predizione del Kalman, re-scan ogni --rescan fotogrammi e dopo un
miss, fotogramma intero anche con piu' di --max-crops track. I crop sono i riquadri bianchi; in alto lo stato e il tempo del passo (modello + crop + tracker).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.crop_tracker import CropTracker, yolo_detect
from src.tracking import KalmanTracker, Params

VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def testo(img, righe):
    for i, t in enumerate(righe):
        y = 28 + 26 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


KF = ("hi", "lo", "min_hits", "max_age", "vote", "iou_min", "iou_dup")
COLORI = [(255, 80, 0), (0, 200, 255), (0, 0, 255), (0, 220, 0)]   # BGR per classe: drone, airplane, helicopter, bird


def disegna(img, dets, tracks, names):
    for x1, y1, x2, y2, *_ in dets:                              # grezze: sottili e grigie
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), 1)
    for t in tracks:
        x1, y1, x2, y2 = t.box.astype(int)
        c = COLORI[t.cls % len(COLORI)]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        etichetta = f"#{t.id} {names[t.cls]} {t.conf:.2f}" + (" pred" if t.age else "")
        cv2.putText(img, etichetta, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)


def un_video(model, src: Path, a) -> bool:
    """False se l'utente ha chiesto di chiudere tutto (Esc)."""
    cap = cv2.VideoCapture(str(src))
    w, h, fps_v = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS) or 30
    # ingresso del modello: letterbox di Ultralytics con un .pt (scala per stare in imgsz, poi multipli di 32)
    H, W = a.imgsz if len(a.imgsz) == 2 else a.imgsz * 2
    if a.mode == "crop":
        H, W = (-(-x // 32) * 32 for x in (H, W))
        ch, cw = min(h, H), min(w, W)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        iw, ih = W, H
    else:
        s = min(H / h, W / w)
        iw, ih = (-(-round(x * s) // 32) * 32 for x in (w, h))
    out = None
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(Path(a.out) / f"{src.stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))
    tr = KalmanTracker(len(model.names), Params(**{k: getattr(a, k) for k in KF})) if a.kalman else None
    conf = a.lo if a.kalman else a.conf
    ct = CropTracker(yolo_detect(model), len(model.names), Params(**{k: getattr(a, k) for k in KF}),
                     scan_imgsz=a.imgsz if len(a.imgsz) == 2 else a.imgsz[0], crop=a.crop_size,
                     rescan=a.rescan, max_crops=a.max_crops) if a.kalman_crop else None
    fps, t0 = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if ct:
            ts = time.perf_counter()
            shown, info = ct.step(frame, 1 / fps_v)
            ms = 1000 * (time.perf_counter() - ts)
            img = frame.copy()
            for x0, y0, lato, tid, pb in info["rois"]:   # il crop usato davvero e la box predetta che lo centra
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + lato), int(y0 + lato)), (255, 255, 255), 2)
                cv2.putText(img, f"crop #{tid} {lato:.0f}px -> {a.crop_size} (x{a.crop_size / lato:.2f})",
                            (int(x0) + 4, int(y0 + lato) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                x1, y1, x2, y2 = pb.astype(int)
                for i in range(x1, x2, 8):                                    # predetta: tratteggio bianco
                    cv2.line(img, (i, y1), (min(i + 4, x2), y1), (255, 255, 255), 1)
                    cv2.line(img, (i, y2), (min(i + 4, x2), y2), (255, 255, 255), 1)
                for j in range(y1, y2, 8):
                    cv2.line(img, (x1, j), (x1, min(j + 4, y2)), (255, 255, 255), 1)
                    cv2.line(img, (x2, j), (x2, min(j + 4, y2)), (255, 255, 255), 1)
            disegna(img, info["dets"], shown, model.names)
            stato = (f"SCAN ingresso {iw}x{ih}" if info["mode"] == "SCAN"
                     else f"TRACK {len(info['rois'])} crop {a.crop_size}x{a.crop_size}")
            testo(img, [f"video {w}x{h} @ {fps_v:.0f} fps  |  kalman-crop: {stato}",
                        f"passo {ms:.1f} ms  |  {fps:.1f} FPS  |  {len(ct.tr.tracks)} track  hi {a.hi} lo {a.lo}"])
        elif a.mode == "crop":
            tela = np.full((H, W, 3), 114, np.uint8)   # gia' alla forma dell'ingresso: il letterbox non scala
            tela[:ch, :cw] = frame[y0:y0 + ch, x0:x0 + cw]
            r = model.predict(tela, imgsz=[H, W], conf=conf, verbose=False)[0]
            d = r.boxes.data.cpu().numpy()
            d = d[(d[:, 0] < cw) & (d[:, 1] < ch)]                  # niente box nel grigio di riempimento
            d[:, [0, 2]] = d[:, [0, 2]].clip(0, cw) + x0
            d[:, [1, 3]] = d[:, [1, 3]].clip(0, ch) + y0
            img = frame.copy()
            if not tr:
                img[y0:y0 + ch, x0:x0 + cw] = r.plot()[:ch, :cw]
            cv2.rectangle(img, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (255, 255, 255), 2)
        else:
            r = model.predict(frame, imgsz=a.imgsz if len(a.imgsz) == 2 else a.imgsz[0], conf=conf, verbose=False)[0]
            d = r.boxes.data.cpu().numpy()
            img = frame.copy() if tr else r.plot()
        if tr:
            disegna(img, d, tr.step(d, 1 / fps_v), model.names)
        t1 = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (t1 - t0) if fps else 1 / (t1 - t0)   # media mobile: numero leggibile
        t0 = t1
        if not ct:
            testo(img, [f"video {w}x{h} @ {fps_v:.0f} fps  |  {a.mode}, ingresso {iw}x{ih}",
                        f"inferenza {r.speed['inference']:.1f} ms  |  {fps:.1f} FPS  |  {len(d)} box  "
                        + (f"kalman hi {a.hi} lo {a.lo}, {len(tr.tracks)} track" if tr else f"conf {a.conf}")])
        if out:
            out.write(img)
        if a.show:
            cv2.imshow("predict", img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == 27:
                cap.release(), out and out.release()
                return False
    cap.release()
    if out:
        out.release()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model"), p.add_argument("source")
    p.add_argument("--conf", type=float, default=0.25), p.add_argument("--imgsz", type=lambda x: [int(v) for v in x.split(",")], default=[640])
    p.add_argument("--mode", choices=["fit", "crop"], default="fit")
    p.add_argument("--out"), p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--kalman", action="store_true")
    p.add_argument("--kalman-crop", action="store_true")
    p.add_argument("--crop-size", type=int, default=256), p.add_argument("--rescan", type=int, default=30)
    p.add_argument("--max-crops", type=int, default=4, help="kalman-crop: con piu' track confermati, fotogramma intero")
    d = Params()
    for k in KF:
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(getattr(d, k)), default=getattr(d, k))
    a = p.parse_args()
    s = Path(a.source)
    videos = sorted(f for f in s.iterdir() if f.suffix.lower() in VIDEO) if s.is_dir() else [s]
    assert videos, f"nessun video in {s}"
    model = YOLO(a.model)
    for v in videos:
        print(v.name, flush=True)
        if not un_video(model, v, a):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
