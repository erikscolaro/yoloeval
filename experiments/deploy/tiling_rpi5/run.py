"""Tiling dei primi strati di YOLO26 sul Pi 5 (OpenVINO fp16): parallelismo sull'ingresso invece che dentro il nodo.

Testa = strati 0..k di P e N (k = 0..3, tutti in sequenza: il primo ramo laterale parte dallo strato 4).
- non tiled: la testa sull'immagine 640x640, 1 stream da 4 thread (come il deploy) e da 1 thread (quanto
  OpenVINO divide il nodo fra i core);
- tiled: la testa su T tile (griglia 2x2, 2x4, 4x4) con halo, 4 stream da 1 thread e 4 richieste in volo,
  cioe' ogni core una tile con gli stessi pesi. Tempo dell'immagine = ceil(T/4) x mediana di una tile
  (le tile interne, le piu' grandi: stima per eccesso), e per controllo T / throughput.

L'halo e' il piu' piccolo multiplo del passo per cui le tile ricucite danno esattamente l'uscita intera
(verificato in fp32 sull'host prima di misurare). Cucitura e copia delle tile non sono misurate.

  python experiments/deploy/tiling_rpi5/run.py build      # IR + verifica halo (host)
  flock ~/Desktop/yoloeval/runs/.pi.lock python experiments/deploy/tiling_rpi5/run.py measure
  python experiments/deploy/tiling_rpi5/run.py table
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

M = Path.home() / "Desktop/yoloeval"
WEIGHTS = {"P": M / "artifacts/weights/yolo26p_D126_air_detection_all_e300_59d8534a/best.pt",
           "N": M / "artifacts/weights/yolo26n_D126_air_detection_all_e300_9dca8d15/best.pt"}
CUTS = (0, 1, 2, 3)
GRIDS = {4: (2, 2), 8: (2, 4), 16: (4, 4)}       # T: (righe, colonne)
SIZE = 640
IR = HERE / "runs/ir"
RES = HERE / "results"


def head(name: str, k: int):
    import torch.nn as nn
    from ultralytics import YOLO
    m = YOLO(str(WEIGHTS[name])).model.float().fuse().eval()
    return nn.Sequential(*m.model[:k + 1])


def stride(net) -> int:
    import torch
    with torch.no_grad():
        return SIZE // net(torch.zeros(1, 3, SIZE, SIZE)).shape[-1]


def tiles(T: int, h: int):
    """(y0, y1, x0, x1) di ogni tile senza halo, e lato della tile interna con halo."""
    r, c = GRIDS[T]
    th, tw = SIZE // r, SIZE // c
    boxes = [(i * th, (i + 1) * th, j * tw, (j + 1) * tw) for i in range(r) for j in range(c)]
    return boxes, (th + 2 * h, tw + 2 * h)


def stitched(net, x, T: int, h: int, s: int):
    """Uscita ricucita dalle tile con halo h (ai bordi dell'immagine niente halo: c'e' lo zero padding)."""
    import torch
    out = None
    for y0, y1, x0, x1 in tiles(T, h)[0]:
        a, b, c, d = max(y0 - h, 0), min(y1 + h, SIZE), max(x0 - h, 0), min(x1 + h, SIZE)
        with torch.no_grad():
            o = net(x[..., a:b, c:d])
        o = o[..., (y0 - a) // s:(y0 - a + y1 - y0) // s, (x0 - c) // s:(x0 - c + x1 - x0) // s]
        if out is None:
            out = torch.zeros(1, o.shape[1], SIZE // s, SIZE // s)
        out[..., y0 // s:y1 // s, x0 // s:x1 // s] = o
    return out


def halo(net, s: int) -> int:
    """Piu' piccolo multiplo di s che rende le tile ricucite uguali all'uscita intera, per tutte le griglie."""
    import torch
    torch.manual_seed(0)
    x = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = net(x)
    for h in range(0, 64 + 1, s):
        if all(torch.allclose(stitched(net, x, T, h, s), ref, atol=1e-4) for T in GRIDS):
            return h
    raise RuntimeError("halo oltre 64 px")


def to_ir(net, hw: tuple[int, int], dst: Path) -> None:
    """ONNX fp32 -> IR fp16 con I/O NHWC, come le reti della LUT (src/lut/pi.py)."""
    import io

    import openvino as ov
    import torch
    from openvino.preprocess import PrePostProcessor
    buf = io.BytesIO()
    torch.onnx.export(net, torch.zeros(1, 3, *hw), buf, opset_version=17, dynamo=False)
    m = ov.Core().read_model(buf.getvalue(), b"")
    pp = PrePostProcessor(m)
    for t in (pp.input(), pp.output()):
        t.tensor().set_layout(ov.Layout("NHWC"))
        t.model().set_layout(ov.Layout("NCHW"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(pp.build(), str(dst), compress_to_fp16=True)


def build() -> None:
    plan = []
    for name in WEIGHTS:
        for k in CUTS:
            net = head(name, k)
            s = stride(net)
            h = halo(net, s)
            to_ir(net, (SIZE, SIZE), IR / f"{name}_k{k}_full.xml")
            plan.append({"modello": name, "k": k, "T": 1, "passo": s, "halo": h, "tile": f"{SIZE}x{SIZE}",
                         "xml": f"{name}_k{k}_full.xml"})
            for T in GRIDS:
                hw = tiles(T, h)[1]
                to_ir(net, hw, IR / f"{name}_k{k}_T{T}.xml")
                plan.append({"modello": name, "k": k, "T": T, "passo": s, "halo": h, "tile": f"{hw[0]}x{hw[1]}",
                             "xml": f"{name}_k{k}_T{T}.xml"})
            print(name, k, "passo", s, "halo", h, flush=True)
    (IR / "plan.json").write_text(json.dumps(plan, indent=1))


def measure() -> None:
    from src.lut.pi import make_cfg, measure as one, session
    plan = json.loads((IR / "plan.json").read_text())
    cfg = make_cfg()
    cfg.backend.benchmark.iters = 200
    remote = f"{cfg.hardware.remote.workdir}/tiling"
    RES.mkdir(parents=True, exist_ok=True)

    def upload(live):
        for p in IR.glob("*"):
            live.put(str(p), f"{remote}/{p.name}")

    out = RES / "misure.csv"
    with session(cfg, remote, RES, upload) as s, open(out, "w", newline="") as f:
        w = None
        for p in plan:
            base = s.cmd(f"{remote}/{p['xml']}")
            modes = ([("4 thread", base), ("1 thread", base.replace("-nthreads 4", "-nthreads 1"))]
                     if p["T"] == 1 else [("4 stream x 1 thread", base.replace("-nstreams 1", "-nstreams 4 -nireq 4"))])
            for mode, cmd in modes:
                lat, _, st, redo = one(s, cmd, p["xml"])
                r = {**{k: p[k] for k in ("modello", "k", "T", "passo", "halo", "tile")}, "modo": mode,
                     "mediana_ms": lat.median_ms, "media_ms": lat.mean_ms, "fps": lat.throughput_qps,
                     "temp_c": st["temp_c"], "freq_khz": st["freq_khz"], "governor": st["governor"],
                     "throttled": st["throttled"], "rifatte": redo}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(r))
                    w.writeheader()
                w.writerow(r)
                f.flush()
                print(r, flush=True)


def table() -> None:
    rows = list(csv.DictReader(open(RES / "misure.csv")))
    base = {(r["modello"], r["k"]): float(r["mediana_ms"]) for r in rows if r["modo"] == "4 thread"}
    out = []
    for r in rows:
        T, med = int(r["T"]), float(r["mediana_ms"])
        img = med if T == 1 else math.ceil(T / 4) * med
        out.append({**{k: r[k] for k in ("modello", "k", "passo", "halo", "T", "tile", "modo")},
                    "tile_ms": round(med, 3), "immagine_ms": round(img, 3),
                    "immagine_da_fps_ms": round(T * 1000 / float(r["fps"]), 3) if T > 1 else "",
                    "speedup": round(base[(r["modello"], r["k"])] / img, 2)})
    with open(RES / "tabella.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print("\t".join(str(v) for v in r.values()))


if __name__ == "__main__":
    {"build": build, "measure": measure, "table": table}[sys.argv[1]]()
