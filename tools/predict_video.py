"""Predict su un video o una cartella di video, con le box dal vivo e in sovrimpressione la risoluzione del video,
l'ingresso del modello, il tempo di inferenza e gli FPS veri (lettura + inferenza + disegno). q chiude il video,
Esc tutto. I video annotati vanno in OUT/<nome del video>.mp4.

    python -m tools.predict_video MODELLO.pt SORGENTE [--conf 0.25] [--imgsz 640] [--out predict/risultati/X] [--no-show]
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def testo(img, righe):
    for i, t in enumerate(righe):
        y = 28 + 26 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def un_video(model, src: Path, a) -> bool:
    """False se l'utente ha chiesto di chiudere tutto (Esc)."""
    cap = cv2.VideoCapture(str(src))
    w, h, fps_v = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS) or 30
    # ingresso del modello: letterbox di Ultralytics con un .pt (lato lungo a imgsz, l'altro arrotondato a 32)
    s = a.imgsz / max(w, h)
    iw, ih = (-(-round(x * s) // 32) * 32 for x in (w, h))
    out = None
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(Path(a.out) / f"{src.stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))
    fps, t0 = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        img = r.plot()
        t1 = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (t1 - t0) if fps else 1 / (t1 - t0)   # media mobile: numero leggibile
        t0 = t1
        testo(img, [f"video {w}x{h} @ {fps_v:.0f} fps  |  ingresso {iw}x{ih}",
                    f"inferenza {r.speed['inference']:.1f} ms  |  {fps:.1f} FPS  |  {len(r.boxes)} box  conf {a.conf}"])
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
    p.add_argument("--conf", type=float, default=0.25), p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out"), p.add_argument("--no-show", dest="show", action="store_false")
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
