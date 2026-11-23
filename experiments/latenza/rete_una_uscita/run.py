"""Neck e testa a una sola scala: 3 reti lineari (backbone -> Detect su P3, P4 o P5) contro il nano (N, D126).
Nessun training (pesi casuali). Per ogni rete: parametri e MAC dell'ONNX fp32 a 640x640 e latenza sul Pi 5
(OpenVINO fp16, 4 core, governor performance, come la griglia: src/lut/pi.py).
    run.py build     # yaml, .pt, ONNX, IR fp16 e complessita' in runs/<rete>/, results/complessita.csv
    run.py measure   # latenza sul Pi, 3 passate in ordine casuale, results/latenza.csv (con il lock del Pi)
    run.py table     # results/tabella.md"""
import copy
import csv
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.lut import pi as lut  # noqa: E402
from src.models import arch  # noqa: E402

NC = 4
# rete -> indice del layer del backbone che entra nel Detect (None = nano intero)
RETI = {"nano": None, "L3": 4, "L4": 6, "L5": 10}
DESCR = {"nano": "nano intero: neck PAN, Detect(P3, P4, P5)",
         "L3": "backbone fino a P3 (layer 4, stride 8) -> Detect",
         "L4": "backbone fino a P4 (layer 6, stride 16) -> Detect",
         "L5": "backbone fino a P5 (layer 10, stride 32, con SPPF e C2PSA) -> Detect"}


def yaml_rete(nome: str) -> dict:
    d = copy.deepcopy(arch(0.25, "D126", NC))
    i = RETI[nome]
    if i is not None:
        d["backbone"] = d["backbone"][:i + 1]
        d["head"] = [[[i], 1, "Detect", [NC]]]
    return d


def build():
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    from src.backends import get_backend
    from src.measure.complexity import onnx_complexity
    from src.remote.connection import LocalConnection
    cfg = lut.make_cfg()
    backend = get_backend("openvino", cfg)
    rows = []
    for nome in RETI:
        d = HERE / "runs" / nome
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        YAML.save(d / f"{nome}.yaml", yaml_rete(nome))
        yo = YOLO(str(d / f"{nome}.yaml"), task="detect")
        yo.save(d / f"{nome}.pt")
        onnx = YOLO(str(d / f"{nome}.pt"), task="detect").export(format="onnx", imgsz=640, batch=1, simplify=True,
                                                                  dynamic=False, device="cpu")
        cx = onnx_complexity(onnx)
        ir = backend.export(LocalConnection(), cfg, d / f"{nome}.pt", d / "ov")
        shutil.copytree(ir, d / "ir")
        rows.append({"rete": nome, "descrizione": DESCR[nome], "params": cx["params"], "gmac": round(cx["macs"] / 1e9, 4),
                     "layer_calcolo": cx["compute_layers"]})
        print(rows[-1], flush=True)
    with open(HERE / "results/complessita.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def measure(repeats=3):
    cfg = lut.make_cfg()
    lut.set_iters(cfg, 300)
    remote = f"{cfg.hardware.remote.workdir}/neck_lineare"
    order = []
    rng = random.Random(0)
    for rep in range(repeats):
        p = list(RETI)
        rng.shuffle(p)
        order += [(rep, n) for n in p]

    def upload(live):
        for n in RETI:
            xml = next((HERE / "runs" / n / "ir").glob("*.xml"))
            live.run(f"mkdir -p {remote}/{n}", hide=True)
            live.put(str(xml), f"{remote}/{n}/net.xml")
            live.put(str(xml.with_suffix(".bin")), f"{remote}/{n}/net.bin")

    out = HERE / "results/latenza.csv"
    with lut.session(cfg, remote, HERE / "results", upload) as s, open(out, "w", newline="") as fh:
        w = None
        for rep, n in order:
            lat, _, st, rem = lut.measure(s, s.cmd(f"{remote}/{n}/net.xml"), n)
            r = {"rete": n, "passata": rep, **lut.row(s, lat, st, rem)}
            if w is None:
                w = csv.DictWriter(fh, list(r))
                w.writeheader()
            w.writerow(r)
            fh.flush()
            print(n, rep, f"{lat.median_ms:.2f} ms {st['temp_c']:.1f} C", flush=True)


def table():
    import numpy as np
    cx = {r["rete"]: r for r in csv.DictReader(open(HERE / "results/complessita.csv"))}
    lat = {}
    for r in csv.DictReader(open(HERE / "results/latenza.csv")):
        lat.setdefault(r["rete"], []).append(float(r["ms"]))
    base = np.median(lat["nano"])
    lines = ["| rete | parametri | GMAC | latenza (ms) | rispetto al nano |", "|---|---|---|---|---|"]
    for n in RETI:
        m = np.median(lat[n])
        lines.append(f"| {n} | {int(cx[n]['params']):,} | {float(cx[n]['gmac']):.2f} | {m:.1f} | {m / base:.0%} |"
                     .replace(",", "."))
    (HERE / "results/tabella.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"build": build, "measure": measure, "table": table}[sys.argv[1]]()
