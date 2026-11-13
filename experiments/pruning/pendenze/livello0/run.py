"""Livello 0 delle pendenze dello stimatore di latenza: solo il costo di latenza, niente loss di detection, niente dati.

  python run.py search    # per ogni modalita': ottimizza le maschere dal nano D126 al target, esporta la rete potata
                          # -> results/trace_<modo>.csv, summary.csv, layers.csv, nets.csv, runs/nets/<modo>/
  python run.py measure   # Pi: le reti di nets.csv (src.lut.pi.b4r: stesse condizioni della LUT e del benchmark) -> results/raw.csv
  python run.py report    # latenza misurata contro stimata -> results/confronto.csv

Impostazioni di pit_latency (conf/strategy/pit_latency.yaml): n = 2, remainder, discrete_cost, DUCCIO con target in ms
sul modello intero, maschere con AdamW lr 0.01 costante, wd 0. Il passo e' un'iterazione di ricerca senza batch: la
loss e' solo il termine di DUCCIO, con task_loss = 1 (AdamW e' quasi invariante alla scala del gradiente). Le STEPS
iterazioni sono divise in EPOCHS "epoche" finte: DUCCIO (forza che cresce nella prima meta') e la D di `secant` seguono
lo stesso calendario della ricerca vera (epoca finta = 30 come search.epochs).
"""
import csv
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.lut import b4r, pi  # noqa: E402

D = Path(__file__).resolve().parent
RES, NETS = D / "results", D / "runs" / "nets"
WEIGHTS = REPO / "artifacts/weights/yolo26n_D126_air_detection_all_e300_9dca8d15/best.pt"
LUT_DIR, FIT, ARCH = REPO / "artifacts/lut/rpi5", REPO / "artifacts/lut/rpi5/fit.json", "w0.25_D126"
MODES = ["lut", "smooth", "secant", "best"]
TARGET_MS, N = 28.0, 2
OPTS = {"adamw": dict(lr=0.01), "sgd": dict(lr=float(os.environ.get("SGD_LR", 10.0)), momentum=0.0)}
STEPS, EPOCHS = int(os.environ.get("STEPS", 3000)), 30
WINDOW = 0.2        # finestra finale per le oscillazioni: ultimo 20 % dei passi
STEP_REL = 0.05     # gradino: -2 canali fanno calare la LUT del layer di oltre il 5 % (STEP_MIN di yolopit.latency)


def build(mode):
    from yolopit.search import build_pit_model
    lat = dict(arch=ARCH, estimator=mode, lut_dir=str(LUT_DIR), fit=str(FIT))
    return build_pit_model(str(WEIGHTS), n=N, remainder=True, trace_imgsz=640, verbose=False,
                           discrete_cost=True, latency=lat)


def export(w, name):
    """Rete potata -> IR come lo stadio export; ritorna (conv_layers, MACs non servono: solo i layer)."""
    from src.backends import get_backend
    from src.remote.connection import LocalConnection
    from yolopit.search import PITYOLO
    p = PITYOLO.__new__(PITYOLO)
    p.model, p.trainer = w, None
    tmp = Path(tempfile.mkdtemp())
    yo = p.export_pruned(tmp / f"{name}.pt")
    layers = b4r.conv_layers(yo.model.eval())
    ir = backend().export(LocalConnection(), CFG, tmp / f"{name}.pt", tmp / "ov")
    dst = NETS / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(ir, dst)
    shutil.copy(tmp / f"{name}.pt", dst / f"{name}.pt")
    shutil.rmtree(tmp)
    return layers, b4r.nonconv_elems(next(dst.glob("*.xml")))


CFG = None


def backend():
    from src.backends import get_backend
    return get_backend("openvino", CFG)


def optimize(mode, opt_name):
    """Maschere con il solo costo di latenza. Ritorna il modello, la traccia [(passo, ms)] e le maschere al primo
    passo con latenza <= target (None se non ci arriva)."""
    import torch
    from yolopit.search import resolve_regularizer
    w = build(mode)
    reg, _ = resolve_regularizer(w, {"mode": "duccio", "target": {"latency": TARGET_MS}})
    w.net.train_features = True                 # maschere addestrabili, come la fase di ricerca
    w.train()
    params = list(w.net.nas_parameters())
    opt = (torch.optim.AdamW(params, weight_decay=0.0, **OPTS[opt_name]) if opt_name == "adamw"
           else torch.optim.SGD(params, **OPTS[opt_name]))
    first = None
    reg.start(1.0)
    per_epoch = STEPS / EPOCHS
    trace = []
    for s in range(STEPS):
        e = min(int(s / per_epoch), EPOCHS - 1)
        for lut in w.latency_luts.values():
            lut.progress = e / (EPOCHS - 1)
        loss = reg(w.net, e, EPOCHS)
        loss.backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            trace.append((s, float(w.net.get_cost("latency")) + w.fixed["latency"]))
            if first is None and trace[-1][1] <= TARGET_MS:
                first = [p.clone() for p in params]
    return w, trace, first


def inversions(lat):
    side = [x > TARGET_MS for x in lat]
    return sum(a != b for a, b in zip(side, side[1:]))


def layer_rows(mode, w, layers, init, a):
    """Per layer: canali finali e se e' appena sopra un gradino (vedi README)."""
    rows = []
    for (name, f, ci, co), (n0, _, ci0, co0) in zip(layers, init):
        assert name == n0
        lut = w.latency_luts[f]
        here = float(lut.bilinear(ci, co))
        d_out = here - float(lut.bilinear(ci, co - 2)) if co - 2 >= lut.ys[0] else 0.0
        d_in = here - float(lut.bilinear(ci - 2, co)) if ci - 2 >= lut.xs[0] else 0.0   # lo stem ha c_in fisso
        rows.append({"net": mode, "layer": name, "family": f, "c_in0": ci0, "c_out0": co0, "c_in": ci, "c_out": co,
                     "lut_ms": round(here, 5), "drop_out_ms": round(d_out, 5), "drop_in_ms": round(d_in, 5),
                     "step_out": int(d_out > STEP_REL * here), "step_in": int(d_in > STEP_REL * here),
                     "gain_ms": round(a[f] * (d_out if d_out > STEP_REL * here else 0.0), 5)})
    return rows


def write(path, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def search():
    global CFG
    from yolopit.latency import load_fit
    CFG = pi.make_cfg()
    RES.mkdir(parents=True, exist_ok=True)
    a, _ = load_fit(FIT, ARCH)
    full = build("smooth")
    init, nc = export(full, "nano_full")
    nets = [{"net": "nano_full", "mode": "", "est_ms": float(full.real_costs()["latency"]), "nonconv_elems": nc}]
    summary, layers_all = [], []
    for opt_name in OPTS:
      for mode in MODES:
        t0 = time.time()
        w, trace, first = optimize(mode, opt_name)
        lat = [x for _, x in trace]
        write(RES / f"trace_{mode}_{opt_name}.csv", [{"step": s, "est_ms": round(x, 5)} for s, x in trace])
        k = int(len(lat) * WINDOW)
        tail = lat[-k:]
        hit = next((s for s, x in trace if x <= TARGET_MS), "")
        variants = [("final", None)] + ([("first", first)] if first else [])
        for variant, masks in variants:
            if masks:       # maschere al primo passo sotto il target
                import torch
                with torch.no_grad():
                    for p_, v in zip(w.net.nas_parameters(), masks):
                        p_.copy_(v)
            est = float(w.real_costs()["latency"])
            name = f"{mode}_{opt_name}_{variant}"
            layers, nc = export(w, name)
            rows = layer_rows(name, w, layers, init, a)
            summary.append({
                "net": name, "mode": mode, "opt": opt_name, "variant": variant, "target_ms": TARGET_MS,
                "est_ms": round(est, 4), "reached": int(est <= TARGET_MS), "first_step_le_target": hit,
                "min_ms": round(min(lat), 4), "window_steps": k, "inversions": inversions(tail),
                "changes": sum(x != y for x, y in zip(tail, tail[1:])), "range_ms": round(max(tail) - min(tail), 4),
                "channels_out": sum(r["c_out"] for r in rows), "channels_out0": sum(r["c_out0"] for r in rows),
                "layers": len(rows), "above_step_out": sum(r["step_out"] for r in rows),
                "above_step_any": sum(max(r["step_out"], r["step_in"]) for r in rows),
                "seconds": round(time.time() - t0)})
            nets.append({"net": name, "mode": mode, "est_ms": est, "nonconv_elems": nc})
            layers_all += rows
            print(summary[-1], flush=True)
    write(RES / "summary.csv", summary)
    write(RES / "layers.csv", layers_all)
    write(RES / "nets.csv", nets)


def report():
    nets = {r["net"]: float(r["est_ms"]) for r in csv.DictReader(open(RES / "nets.csv"))}
    meas = {}
    for r in csv.DictReader(open(RES / "raw.csv")):
        meas.setdefault(r["net"], []).append(float(r["ms"]))
    rows = [{"net": n, "est_ms": round(e, 3), "meas_ms": round(statistics.median(meas[n]), 3), "repeats": len(meas[n]),
             "err_ms": round(statistics.median(meas[n]) - e, 3),
             "err_pct": round(100 * (statistics.median(meas[n]) / e - 1), 2)} for n, e in nets.items() if n in meas]
    write(RES / "confronto.csv", rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    {"search": search, "measure": lambda: pi.b4r(D, []), "report": report}[sys.argv[1]]()
