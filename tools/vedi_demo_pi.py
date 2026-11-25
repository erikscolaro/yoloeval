"""Sulla workstation: avvia la demo sul Pi senza finestra (demo_pi.py --meta) e la mostra qui con la stessa interfaccia
di tools/predict_video.py (bande, spia DRONE, pulsanti, scia, crop trasparente con la box predetta, detection grezze).
In rete passano solo i metadati (0,3-2 KB a fotogramma); il fotogramma lo legge la workstation dalla sua copia
(data/tracking/_demo640, la stessa che scripts/demo_pi.sh copia sul Pi).

    python tools/vedi_demo_pi.py [--modo f2|f2-maschera|f1|f1-maschera] [--video data/tracking/_demo640] [--allarme 15]

Tasti e pulsanti: spazio pausa, . passo, n/p video dopo/prima, m maschera (vanno al Pi), t scia (solo qui), Esc esce.
FPS e ms sono quelli del Pi. Niente barra dei fotogrammi: il Pi non salta dentro il video.
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.predict_video import (ALFA_CROP, DOPO, ESCI, PRIMA, UI, conta_drone, cornice, disegna,  # noqa: E402
                                 disegna_scia, spia, tasto)

R = "/home/bench/bench/demo"
NOMI = {0: "drone", 1: "airplane", 2: "helicopter", 3: "bird"}
FINESTRA = "demo pi"
TASTI = "spazio pausa | . passo | t scia | m maschera | n/-> dopo | p/<- prima | Esc esci"
AL_PI = {"pausa": " ", "passo": ".", "maschera": "m", DOPO: "n", PRIMA: "p", ESCI: "q"}


def tratteggio(img, b):
    x1, y1, x2, y2 = map(int, b)
    for i in range(x1, x2, 8):
        cv2.line(img, (i, y1), (min(i + 4, x2), y1), (255, 255, 255), 1)
        cv2.line(img, (i, y2), (min(i + 4, x2), y2), (255, 255, 255), 1)
    for j in range(y1, y2, 8):
        cv2.line(img, (x1, j), (x1, min(j + 4, y2)), (255, 255, 255), 1)
        cv2.line(img, (x2, j), (x2, min(j + 4, y2)), (255, 255, 255), 1)


def quadro(img, d, scie, serie, a, pausa):
    """Come il ramo kalman-crop di predict_video: crop e box predetta trasparenti, scritte piene, detection, track, scia,
    bande con stato e spia, pulsanti."""
    sopra = img.copy()
    for x0, y0, lato, tid, *pb in d["rois"]:
        cv2.rectangle(sopra, (int(x0), int(y0)), (int(x0 + lato), int(y0 + lato)), (255, 255, 255), 2)
        tratteggio(sopra, pb)
    img = cv2.addWeighted(sopra, ALFA_CROP, img, 1 - ALFA_CROP, 0)
    for x0, y0, lato, tid, *_ in d["rois"]:
        cv2.putText(img, f"crop #{tid} {lato:.0f}px -> {d['crop']} (x{d['crop'] / lato:.2f})",
                    (int(x0) + 4, int(y0 + lato) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    shown = [SimpleNamespace(id=t[0], cls=t[1], conf=t[2], age=t[3], box=np.array(t[4:])) for t in d["t"]]
    disegna(img, d["d"], shown, NOMI)
    if UI.scia:
        disegna_scia(img, scie, shown)
    h, w = img.shape[:2]
    stato = "SCAN ingresso 640" if d["m"] == "SCAN" else f"TRACK {len(d['rois'])} crop {d['crop']}x{d['crop']}"
    righe = [f"{d['v']}  {w}x{h} @ {d['fpsv']:.0f} fps  |  kalman-crop sul Pi: {stato}",
             f"passo {d['ms']:.1f} ms  |  {d['fps']:.1f} FPS sul Pi  |  {d['ntr']} track  hi {d['hi']:.2g} lo {d['lo']:.2g}",
             d["stato"]]
    pulsanti = ([("<< prima", PRIMA), ("riprendi" if pausa else "pausa", "pausa"), ("passo", "passo"), ("dopo >>", DOPO),
                 ("scia: si" if UI.scia else "scia: no", "scia")]
                + ([("maschera: si" if d["masc"] else "maschera: no", "maschera")] if d["masc"] is not None else []))
    q, UI.rett = cornice(img, righe, spia(serie, a.allarme), TASTI, pulsanti)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", default="f2", choices=["f2", "f2-maschera", "f1", "f1-maschera"])
    ap.add_argument("--video", default=str(Path(__file__).resolve().parents[1] / "data/tracking/_demo640"))
    ap.add_argument("--host", default="rpi5")
    ap.add_argument("--allarme", type=int, default=15, help="fotogrammi di fila con classe drone per la spia rossa (0: spenta)")
    ap.add_argument("--zoom", type=float, default=2.0, help="ingrandimento della finestra")
    a = ap.parse_args()
    pi = subprocess.Popen(["ssh", "-o", "BatchMode=yes", a.host, f"{R}/avvia.sh {a.modo} meta"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    righe = queue.Queue()
    threading.Thread(target=lambda: [righe.put(r) for r in pi.stdout] + [righe.put(None)], daemon=True).start()
    cv2.namedWindow(FINESTRA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(FINESTRA, UI.clic)
    UI.finestra = FINESTRA
    cap, nome, pos, img, d, scie, serie, pausa, misura = None, None, -1, None, None, {}, {}, False, False

    def manda(c):
        pi.stdin.write(c)
        pi.stdin.flush()
    try:
        while True:
            try:
                r = righe.get(timeout=0.02)
            except queue.Empty:
                r = ""
            if r is None:                                   # il Pi ha chiuso la demo
                break
            if r.startswith("{"):
                d = json.loads(r)
                if d["v"] != nome or d["f"] < pos:          # video nuovo (o ricominciato): lo apre la workstation
                    cap and cap.release()
                    cap, nome, pos, scie, serie = cv2.VideoCapture(str(Path(a.video) / d["v"])), d["v"], -1, {}, {}
                    if not cap.isOpened():
                        sys.exit(f"manca {d['v']} in {a.video}: rifare scripts/demo_pi.sh prepara")
                while pos < d["f"]:                         # allo stesso fotogramma del Pi
                    ok, img = cap.read()
                    pos += 1
                    if not ok:
                        sys.exit(f"{d['v']}: il Pi e' al fotogramma {d['f']}, qui il video finisce a {pos}")
                scie = {int(k): (scie.get(int(k), []) + [tuple(v)])[-30:] for k, v in d["c"].items()}
                serie = conta_drone(serie, [SimpleNamespace(id=t[0], cls=t[1]) for t in d["t"]], 0)
            elif r:
                print(r, end="")                            # messaggi del Pi che non sono metadati
            if d is not None:                               # ridisegna anche in pausa: pulsanti e scia rispondono
                q = quadro(img.copy(), d, scie, serie, a, pausa)
                cv2.imshow(FINESTRA, q)
                if not misura:
                    cv2.resizeWindow(FINESTRA, int(q.shape[1] * a.zoom), int(q.shape[0] * a.zoom))
                    misura = UI.misura = True
            az = tasto(cv2.waitKeyEx(1)) or UI.prendi()
            if az == "scia":
                UI.scia = not UI.scia
            elif az in AL_PI:
                manda(AL_PI[az])
                if az == "pausa":
                    pausa = not pausa
                if az == "passo":
                    pausa = True
                if az == ESCI:
                    break
    except KeyboardInterrupt:                               # Ctrl+C: chiude come Esc
        pass
    finally:
        try:
            manda("q")
            pi.stdin.close()
            pi.wait(10)
        except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
            pi.kill()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
