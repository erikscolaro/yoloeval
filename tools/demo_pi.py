"""Demo sul Raspberry Pi 5: tracker con Kalman e crop (P- in SCAN, S- su un crop 256 in TRACK, parametri tarati da Optuna)
con OpenVINO fp16 su 4 core, sui video di una cartella. Solo finestra, niente salvataggio. Gira sul Pi: la prepara e la
lancia scripts/demo_pi.sh (dal launcher: utilita' -> demo-pi).

    python demo_pi.py --scan SCAN.xml --track TRACK.xml --parametri parametri.json --video ~/Videos

--parametri: i Params del Kalman, quelli del TRACK e gli argomenti di CropTracker, scritti da scripts/demo_pi.sh con
tools/predict_video.preset dal file di Optuna; "maschera" (o null): soglia per classe sotto cui, nei fotogrammi SCAN, track
e box non si disegnano (solo grafica). Tasti: spazio pausa, . passo, n/p video dopo/prima, m maschera, q o Esc esce. I video girano
in cerchio.
"""
import argparse
import json
import select
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

QUI = Path(__file__).resolve().parent
sys.path[:0] = [str(QUI), str(QUI.parent)]          # src/ accanto (sul Pi) o nella repo
from src.crop_tracker import CropTracker  # noqa: E402
from src.tracking import Params  # noqa: E402

VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
NOMI = ["drone", "airplane", "helicopter", "bird"]
COLORI = [(255, 80, 0), (0, 200, 255), (0, 0, 255), (0, 220, 0)]   # BGR per classe, come tools/predict_video.py


def modello(core, xml):
    """IR compilato come il benchmark: CPU, 4 thread, uno stream, fp16. Restituisce (richiesta, (H, W) dell'ingresso)."""
    m = core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": 4, "NUM_STREAMS": 1,
                                             "INFERENCE_PRECISION_HINT": "f16", "PERFORMANCE_HINT": "LATENCY"})
    return m.create_infer_request(), tuple(m.input(0).shape[2:])


def rileva(req, hw, img, conf):
    """Un'immagine BGR uint8 (al massimo H x W) -> detection (n, 6) xyxy conf cls nelle sue coordinate. Il bordo grigio
    va a destra e in basso, cosi' le coordinate non si spostano."""
    H, W = hw
    tela = np.full((H, W, 3), 114, np.uint8)
    h, w = img.shape[:2]
    tela[:h, :w] = img[:H, :W]
    x = np.ascontiguousarray(tela[..., ::-1].transpose(2, 0, 1)[None], np.float32) / 255   # RGB NCHW come Ultralytics
    d = np.asarray(req.infer({0: x})[0], float).reshape(-1, 6)       # testa one-to-one: (64, 6), niente NMS
    return d[d[:, 4] >= conf]


def disegna(img, info, shown, testo):
    for x0, y0, lato, *_ in info["rois"]:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + lato), int(y0 + lato)), (255, 255, 255), 1)
    for t in shown:
        x1, y1, x2, y2 = t.box.astype(int)
        c = COLORI[t.cls % len(COLORI)]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"#{t.id} {NOMI[t.cls]} {t.conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    for i, r in enumerate(testo):
        cv2.putText(img, r, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, r, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True), ap.add_argument("--track", required=True)
    ap.add_argument("--parametri", required=True), ap.add_argument("--video", default=str(Path.home() / "Videos"))
    ap.add_argument("--schermo-intero", action="store_true",
                    help="finestra a schermo intero: sul Pi costa circa 11 ms a fotogramma (24 FPS invece di 32)")
    ap.add_argument("--meta", action="store_true", help="niente finestra: una riga JSON per fotogramma su stdout (video, "
                    "fotogramma, modo, crop, track, ms) e i tasti da stdin; la disegna tools/vedi_demo_pi.py sulla workstation")
    a = ap.parse_args()
    video = sorted(f for f in Path(a.video).expanduser().iterdir() if f.suffix.lower() in VIDEO)
    assert video, f"nessun video in {a.video}"
    p = json.loads(Path(a.parametri).read_text())
    core = ov.Core()
    (rs, hw_s), (rt, hw_t) = modello(core, a.scan), modello(core, a.track)
    assert hw_t[0] == hw_t[1], f"il modello del TRACK vuole un ingresso quadrato, non {hw_t}"

    def detect(images, imgsz, conf):                 # i crop (imgsz = lato del crop) a S-, il fotogramma ridotto a P-
        req, hw = (rt, hw_t) if imgsz == hw_t[0] else (rs, hw_s)
        return [rileva(req, hw, im, conf) for im in images]

    soglie = np.array([p["maschera"][n] for n in NOMI]) if p.get("maschera") else None
    maschera = soglie is not None
    if a.meta:
        pass
    elif a.schermo_intero:
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:   # alla misura del video: imshow 3,5 ms invece di 10,8 (misura del 25/11/2026, monitor 1920x1080)
        cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
        print("avviso: finestra alla misura del video, non a schermo intero (costerebbe circa 11 ms a fotogramma); "
              "--schermo-intero per averla", flush=True)
    i = 0
    while True:
        cap = cv2.VideoCapture(str(video[i % len(video)]))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
        ct = CropTracker(detect, len(NOMI), Params(**p["kalman"]), scan_imgsz=max(hw_s), crop=hw_t[0], max_crops=1,
                         track=p["track"], **p["crop"])
        fps, pausa, n_scan, da_scan, uno = 0.0, False, 0, 0, False
        while True:
            if not pausa:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                pred = {t.id: ct.tr.peek(t, 1 / fps_v)[0][:2] for t in ct.tr.tracks} if a.meta else {}   # per la scia
                ts = time.perf_counter()
                shown, info = ct.step(frame, 1 / fps_v)
                ms = 1000 * (time.perf_counter() - ts)
                n_scan, da_scan = (n_scan + 1, 0) if info["mode"] == "SCAN" else (n_scan, da_scan + 1)
                stato = "SCAN" if info["mode"] == "SCAN" else "TRACK crop %d" % hw_t[0]
                stato += f"  (SCAN {n_scan}, ultimo {da_scan} fotogrammi fa, rescan ogni {p['crop'].get('rescan', 30)})"
                dets = info["dets"]
                if maschera and info["mode"] == "SCAN":          # solo grafica: il tracker vede tutto
                    shown = [t for t in shown if t.conf >= soglie[t.cls]]
                    dets = [d for d in dets if d[4] >= soglie[int(d[5])]]
                    stato += "  maschera"
                if a.meta:   # 0,3-2 KB: la workstation ha lo stesso video e disegna lei (tools/vedi_demo_pi.py)
                    q = lambda xs, n=1: [round(float(x), n) for x in xs]
                    print(json.dumps({
                        "v": video[i % len(video)].name, "f": int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, "fpsv": fps_v,
                        "m": info["mode"], "stato": stato, "ms": round(ms, 1), "fps": round(fps, 1),
                        "ntr": len(ct.tr.tracks), "hi": p["kalman"]["hi"], "lo": p["kalman"]["lo"], "crop": hw_t[0],
                        "masc": None if soglie is None else maschera,
                        "d": [q(d[:4]) + [round(float(d[4]), 3), int(d[5])] for d in dets],
                        "rois": [q([x0, y0, lato]) + [int(tid)] + q(pb) for x0, y0, lato, tid, pb in info["rois"]],
                        "t": [[t.id, t.cls, round(float(t.conf), 3), t.age] + q(t.box) for t in shown],
                        "c": {t.id: [q(pred[t.id]) if t.id in pred else None, None if t.age else q(t.x[:2])]
                              for t in ct.tr.tracks}}), flush=True)
                else:
                    disegna(frame, info, shown, [f"{video[i % len(video)].name}  {frame.shape[1]}x{frame.shape[0]}",
                                                 f"{stato}  passo {ms:.1f} ms  {fps:.1f} FPS"]
                            + ([] if a.schermo_intero else ["finestra ridotta: lo schermo intero costa ~11 ms"]))
                    cv2.imshow("demo", frame)
            if a.meta:   # i tasti arrivano dalla workstation su stdin
                k = ord(sys.stdin.read(1) or "q") if select.select([sys.stdin], [], [], 0.03 if pausa else 0)[0] else 255
            else:
                k = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty("demo", cv2.WND_PROP_VISIBLE) < 1:   # chiusa con la X: OpenCV la riaprirebbe
                    k = ord("q")
            if k in (ord("q"), 27):
                return
            if k == ord("n"):
                break
            if k == ord("p"):                                   # video prima
                i -= 2
                break
            if uno:                                             # dopo un passo: di nuovo in pausa
                pausa, uno = True, False
            if k == ord("."):
                pausa, uno = False, True
            if k == ord(" "):
                pausa = not pausa
            if k == ord("m") and soglie is not None:
                maschera = not maschera
            if not pausa:
                dt = time.perf_counter() - t0
                fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
        cap.release()
        i += 1


if __name__ == "__main__":
    main()
