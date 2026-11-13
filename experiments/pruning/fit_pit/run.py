"""Reti PIT dentro il fit dello stimatore di latenza del Pi 5: genera, misura, rifa il fit e confronta.

  python run.py gen       # reti PIT (solo maschere, DUCCIO sulla latenza, CPU) -> results/nets.csv, layers.csv, runs/nets/
  python run.py measure   # Pi: src.lut.pi.b4r sulle reti di nets.csv -> results/raw.csv (riprende)
  python run.py fit       # fit B4-R + PIT(train), confronto col vecchio -> results/confronto_*.csv, fit_pit.json

Riusa livello0/run.py (export dell'IR, stesse impostazioni di DUCCIO e delle maschere) e src.lut.fit (solve, cv_lambda...).
Il codice di ricerca e' quello del livello 0, con in piu' le maschere salvate lungo la traiettoria.
"""
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
D = Path(__file__).resolve().parent
RES, NETS = D / "results", D / "runs" / "nets"
B4R = REPO / "experiments/pruning/b4r_composizione/results"
L0 = REPO / "experiments/pruning/pendenze/livello0"
LUT_DIR, FIT_OLD = REPO / "artifacts/lut/rpi5", REPO / "artifacts/lut/rpi5/fit_b4r.json"
if not FIT_OLD.exists():                       # prima della sostituzione il vecchio e' fit.json
    FIT_OLD = REPO / "artifacts/lut/rpi5/fit.json"
STEPS, EPOCHS = int(os.environ.get("STEPS", 1500)), 30
SNAPS = tuple(range(54, 21, -2))               # ms (nano): maschere al primo passo che li attraversa (fitte: ~8 reti per traiettoria)
THIN = 0.03                                    # reti della stessa traiettoria: almeno il 3 % della latenza intera di distanza
NOISE = 0.2                                    # seed > 0: rumore sulle maschere iniziali (valore 1, soglia 0,5)
# (arch, target in frazione della rete intera, seed, split). Il nano intero e' 56,5 ms: 0,62 = 35, 0,35 = 20.
FRACS_TRAIN = (0.62, 0.40)      # il minimo raggiungibile del nano e' 22,08 ms (0,39): il target si porta almeno a 1,05 x minimo
MODES, OPTS = ("lut", "smooth", "secant", "best"), ("adamw", "sgd")
NEAR = ("w0.1875_D126", "w0.25_D101", "w0.25_D170")


PRIORITY = ("w0.125_D101", "w0.125_D126", "w0.09375_D101", "w0.09375_D126", "w0.09375_D170", "w0.125_D170",
            "w0.1875_D101", "w0.1875_D170", "w0.0625_D101", "w0.0625_D126", "w0.0625_D170",
            "w0.1875_D126", "w0.25_D101", "w0.25_D170")        # ordine di priorita': le vicine al budget di 30 ms per prime
# per ogni architettura: nucleo = 3 traiettorie di train (seed 1 e 2) + 2 di test (target 0,47, seed 3); rinforzo = altre,
# solo se servono: fit_eval.roles sceglie 15 reti di train + 5 di test per architettura, le altre sono `extra` (fuori dal fit)
CORE = [("smooth", "adamw", 1, "train"), ("secant", "sgd", 2, "train"), ("lut", "adamw", 1, "train"),
        ("smooth", "adamw", 3, "test"), ("secant", "sgd", 3, "test")]
TOPUP = [("smooth", "sgd", 1, "train"), ("secant", "adamw", 2, "train"), ("smooth", "sgd", 4, "train"),
         ("secant", "adamw", 5, "train"), ("smooth", "sgd", 6, "test")]


def specs():
    """Il solo costo con maschere tutte uguali le fa attraversare la soglia insieme (AdamW salta al minimo, 22,08 ms):
    il seed > 0 rompe la parita' con rumore sulle maschere, cosi' le reti lungo la traiettoria sono diverse."""
    s = [("w0.25_D126", m, o, 0.62, 0, "train") for m in MODES for o in OPTS]
    s += [("w0.25_D126", m, o, 0.40, 0, "train") for m in MODES for o in OPTS]
    s += [("w0.25_D126", m, o, 0.40, seed, "train") for seed in (1, 2) for m in ("lut", "smooth", "secant") for o in OPTS]
    s += [("w0.25_D126", m, o, 0.47, 3, "test") for m in MODES for o in OPTS]      # target e seed mai usati nel train
    s += [(a, "smooth", o, 0.40, 1, "train") for a in NEAR for o in OPTS]
    # seconda tornata (14/11): le altre architetture della griglia B4-R. Interleave per priorita': prima i train/test
    # di ogni architettura in ordine, cosi' una corsa interrotta ha gia' tutte le vicine al budget
    s += [(a, m, o, 0.47 if sp == "test" else 0.40, seed, sp) for lst in (CORE, TOPUP) for a in PRIORITY
          for m, o, seed, sp in lst]
    return s


def load_l0():
    spec = importlib.util.spec_from_file_location("livello0_run", L0 / "run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def weights(arch):
    """Pesi e_300 dell'architettura (il nano e' yolo26n); w0.125_D126 ha solo e_0 (BN non addestrate: IR come lo yaml)."""
    w, d = arch.split("_")
    stem = "yolo26n_" + d if arch == "w0.25_D126" else f"yolo26{w}_{d}"
    found = sorted((REPO / "artifacts/weights").glob(f"{stem}_air_detection_all_e300_*/best.pt")) or \
        sorted((REPO / "artifacts/weights").glob(f"{stem}_air_detection_all_e0_*/best.pt"))
    return found[0]


def build(arch, mode):
    from yolopit.search import build_pit_model
    from src.lut.pi import N_STEP
    best = weights(arch)
    lat = dict(arch=arch, estimator=mode, lut_dir=str(LUT_DIR), fit=str(FIT_OLD))   # solo la pendenza e C: i pesi a_f del fit
    return build_pit_model(str(best), n=N_STEP, remainder=True, trace_imgsz=640, verbose=False, discrete_cost=True, latency=lat)


def trajectory(arch, mode, opt_name, target, seed, full_ms):
    """Come livello0.optimize, ma salva le maschere ai primi passi sotto ciascuna soglia di SNAPS (scalate sull'arch)."""
    import torch
    from yolopit.search import resolve_regularizer
    w = build(arch, mode)
    reg, _ = resolve_regularizer(w, {"mode": "duccio", "target": {"latency": target}})
    w.net.train_features = True
    w.train()
    params = list(w.net.nas_parameters())
    if seed:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in params:
                p.add_(NOISE * torch.randn(p.shape, generator=g))
    opts = {"adamw": lambda: torch.optim.AdamW(params, weight_decay=0.0, lr=0.01),
            "sgd": lambda: torch.optim.SGD(params, lr=10.0, momentum=0.0)}
    opt = opts[opt_name]()
    reg.start(1.0)
    per_epoch, snaps, todo = STEPS / EPOCHS, {}, [s * full_ms / 56.55 for s in SNAPS if s * full_ms / 56.55 > target + 1]
    for s in range(STEPS):
        e = min(int(s / per_epoch), EPOCHS - 1)
        for lut in w.latency_luts.values():
            lut.progress = e / (EPOCHS - 1)
        reg(w.net, e, EPOCHS).backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            ms = float(w.net.get_cost("latency")) + w.fixed["latency"]
            for t in [t for t in todo if ms <= t]:
                snaps[round(t, 1)] = [p.clone() for p in params]
                todo.remove(t)
    return w, params, snaps


def gen(k=0, n=1):
    import fcntl
    import torch
    from src.lut import b4r, pi
    l0 = load_l0()
    l0.CFG, l0.NETS = pi.make_cfg(), NETS
    RES.mkdir(parents=True, exist_ok=True)
    NETS.mkdir(parents=True, exist_ok=True)
    nets_p, layers_p = RES / "nets.csv", RES / "layers.csv"
    dp = RES / "trajs.txt"                          # traiettorie fatte, anche quelle senza reti nuove
    full = {}
    for arch, mode, opt, frac, seed, split in [x for x in specs() if x[0] in os.environ.get("ARCHS", x[0]).split()][k::n]:       # shard k di n: processi in parallelo, scrittura con lock
        nets_rows = list(csv.DictReader(open(nets_p))) if nets_p.exists() else []
        done = {r["traj"] for r in nets_rows} | (set(dp.read_text().split()) if dp.exists() else set())
        traj = f"{arch}_{mode}_{opt}_f{frac}_s{seed}"
        if traj in done:
            continue
        t0 = time.time()
        if arch not in full:
            w = build(arch, "smooth")
            full[arch] = (float(w.real_costs()["latency"]), w.min_total["latency"])
        target = max(frac * full[arch][0], 1.05 * full[arch][1])
        w, params, snaps = trajectory(arch, mode, opt, target, seed, full[arch][0])
        seen = {r["sig"] for r in nets_rows}
        variants = [(f"at{t}", m) for t, m in sorted(snaps.items(), reverse=True)] + [("final", None)]
        new, new_layers, last = [], [], None
        for tag, masks in variants:
            if masks:
                with torch.no_grad():
                    for p, v in zip(params, masks):
                        p.copy_(v)
            name = f"pit_{traj}_{tag}"
            costs = {m: float(v) for m, v in w.real_costs().items()}
            if last is not None and abs(costs["latency"] - last) < THIN * full[arch][0]:
                continue                                 # snapshot troppo vicino al precedente: non si esporta
            last = costs["latency"]
            layers, nc = l0.export(w, name)
            sig = arch + ":" + ",".join(f"{ci}x{co}" for _, _, ci, co in layers)
            if sig in seen:                              # stessa architettura gia' generata
                shutil.rmtree(NETS / name, ignore_errors=True)
                continue
            seen.add(sig)
            wd = arch.split("_")
            new.append({"net": name, "arch": arch, "width": wd[0][1:], "depth": wd[1], "seed": seed,
                        "level": round(costs["latency"], 3), "layers": len(layers),
                        **{f"cost_{m}": v for m, v in costs.items()}, "nonconv_elems": nc, "off_grid": "",
                        "split": split, "traj": traj, "mode": mode, "opt": opt, "target_ms": round(target, 2),
                        "est_ms": round(costs["latency"], 4), "sig": sig})
            new_layers += [{"net": name, "order": i, "layer": ln, "family": f, "c_in": ci, "c_out": co}
                           for i, (ln, f, ci, co) in enumerate(layers)]
        with open(RES / ".lock", "w") as lock:           # un solo processo alla volta aggiorna i csv
            fcntl.flock(lock, fcntl.LOCK_EX)
            nets_rows = list(csv.DictReader(open(nets_p))) if nets_p.exists() else []
            layer_rows = list(csv.DictReader(open(layers_p))) if layers_p.exists() else []
            seen = {r["sig"] for r in nets_rows}
            for r in new[:]:                              # un altro shard puo' aver scritto la stessa rete nel frattempo
                if r["sig"] in seen:
                    new.remove(r)
                    shutil.rmtree(NETS / r["net"], ignore_errors=True)
                seen.add(r["sig"])
            keep = {r["net"] for r in new}
            nets_rows += new
            layer_rows += [r for r in new_layers if r["net"] in keep]
            for p, rows in ((nets_p, nets_rows), (layers_p, layer_rows)):
                if rows:
                    with open(p, "w", newline="") as fh:
                        wr = csv.DictWriter(fh, list(rows[0]))
                        wr.writeheader()
                        wr.writerows(rows)
            with open(dp, "a") as fh:
                fh.write(traj + "\n")
        print(traj, len(variants), "varianti,", len(new), "nuove,", round(time.time() - t0), "s", flush=True)


def measure():
    """Misura sul Pi solo le reti scelte da fit_eval.roles (15 di train + 5 di test per architettura) e quelle gia' misurate.
    pi.b4r legge <dir>/results/nets.csv e scrive raw.csv: una cartella di lavoro con nets.csv filtrato e raw.csv = quello vero."""
    from src.lut import pi
    sys.path.insert(0, str(D))
    import fit_eval
    rows = list(csv.DictReader(open(RES / "nets.csv")))
    role = fit_eval.roles({r["net"]: r for r in rows})
    raw = RES / "raw.csv"
    done = {r["net"] for r in csv.DictReader(open(raw))} if raw.exists() else set()
    keep = [r for r in rows if role[r["net"]] != "extra" or r["net"] in done]
    w = D / "_misura"
    (w / "results").mkdir(parents=True, exist_ok=True)
    (w / "runs").mkdir(exist_ok=True)
    for link, target in ((w / "runs/nets", NETS), (w / "results/raw.csv", raw)):
        if not link.is_symlink():
            link.symlink_to(target)
    with open(w / "results/nets.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0]))
        wr.writeheader()
        wr.writerows(keep)
    print(len(keep), "reti da misurare o gia' misurate su", len(rows), flush=True)
    pi.b4r(w, [])


def l0():
    """Rimisura le reti del livello 0 (IR in livello0/runs/nets) in questa sessione: -> l0_remisura/results/check.csv.
    Serve a separare l'errore dello stimatore dalla deriva fra sessioni di misura."""
    import csv as _csv
    from src.lut import pi
    d = D / "l0_remisura"
    (d / "results").mkdir(parents=True, exist_ok=True)
    (d / "runs").mkdir(exist_ok=True)
    if not (d / "runs/nets").exists():
        (d / "runs/nets").symlink_to(L0 / "runs/nets")
    pi.b4r(d, [r["net"] for r in _csv.DictReader(open(L0 / "results/nets.csv"))])


if __name__ == "__main__":
    {"gen": lambda: gen(*map(int, sys.argv[2:4])), "measure": measure, "l0": l0, "fit": lambda: __import__("fit_eval").main()}[sys.argv[1]]()
