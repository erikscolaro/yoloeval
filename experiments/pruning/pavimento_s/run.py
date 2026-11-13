"""Pavimento di latenza della S sul Pi 5: YOLO26 w0.5 D126 con ogni maschera potabile di yolopit al minimo
(un solo blocco da n = 2 canali), resto invariato. Controlli: lo stesso per il nano w0.25 D126, e le due reti intere.

  python experiments/pruning/pavimento_s/run.py build     # reti -> runs/nets/<net>, results/nets.csv, layers.csv
  python experiments/pruning/pavimento_s/run.py measure   # Pi: 3 ripetizioni x ITERS_B4R iterazioni -> results/raw.csv
Costruzione, export e misura sono quelli di src.lut (B4-R): stesso IR fp16 640, stessi strati e stessa guardia termica.
"""
import csv
import logging
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.lut import b4r, pi  # noqa: E402

D = Path(__file__).resolve().parent
NETS = [("s_floor", 0.5, True), ("nano_floor", 0.25, True), ("s_full", 0.5, False), ("nano_full", 0.25, False)]


def floor(masks):
    import torch
    for m in masks:
        v = torch.zeros(len(m.sizes))
        v[len(m.sizes) - 1 - (1 if m.ordered_remainder else 0)] = 1     # il blocco tenuto sempre acceso da B4-R
        with torch.no_grad():
            m.block.copy_(v)


def build():
    from src.backends import get_backend
    from src.remote.connection import LocalConnection
    from yolopit.search import PITYOLO
    cfg = pi.make_cfg()
    backend = get_backend("openvino", cfg)
    (D / "results").mkdir(exist_ok=True)
    nets, layers = [], []
    for name, width, flr in NETS:
        w = b4r.pit_model(width, "D126")
        ms = b4r.maskers(w)
        if flr:
            floor(ms)
        p = PITYOLO.__new__(PITYOLO)
        p.model, p.trainer = w, None
        tmp = Path(tempfile.mkdtemp())
        yo = p.export_pruned(tmp / f"{name}.pt")
        ls = b4r.conv_layers(yo.model.eval())
        ir = backend.export(LocalConnection(), cfg, tmp / f"{name}.pt", tmp / "ov")
        dst = D / "runs" / "nets" / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(ir, dst)
        shutil.copy(tmp / f"{name}.pt", dst / f"{name}.pt")
        shutil.rmtree(tmp)
        costs = {m: float(v) for m, v in w.real_costs().items()}
        nets.append({"net": name, "width": width, "floor": int(flr), "masks": len(ms), "layers": len(ls),
                     **{f"cost_{m}": v for m, v in costs.items()},
                     "nonconv_elems": b4r.nonconv_elems(next(dst.glob("*.xml")))})
        layers += [{"net": name, "order": i, "layer": n, "family": f, "c_in": ci, "c_out": co}
                   for i, (n, f, ci, co) in enumerate(ls)]
        print(name, len(ms), "maschere", len(ls), "strati; c_out > 2:",
              [(n, ci, co) for n, _, ci, co in ls if flr and co > 2], flush=True)
    for path, rows in ((D / "results/nets.csv", nets), (D / "results/layers.csv", layers)):
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    {"build": build, "measure": lambda: pi.b4r(D, [])}[sys.argv[1]]()
