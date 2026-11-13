"""B4-R: random pruned YOLO26 of every architecture up to the nano (widths 0.0625-0.25 x D101/D126/D170,
4 classes) with yolopit's constraints, exported for the Pi as in the pipeline (OpenVINO IR fp16, 640, one-to-one).

  python -m src.lut.b4r <dir> [N]  # per architecture: N random nets (default N_NETS) + the unpruned one
                                   # (<arch>_full): <dir>/runs/nets/<net>/ (IR and .pt), <dir>/results/nets.csv
                                   # (one row per net: MACs, parameters, output elements of
                                   # Concat/Add/Interpolate/MaxPool), <dir>/results/layers.csv (one row per conv)
Measure: python -m src.lut.pi b4r <dir>; fit: python -m src.lut.fit. Parameters (ARCHS, N_NETS, LEVEL, ...) in src.lut.pi.
Each net is built with yolopit (build_pit_model, n = 2 = LUT step): its block masks already encode the
constraints (residual/concat masks shared, outputs feeding C2PSA/PSABlock/Detect frozen, C3k2 split in cv1a
and cv1b, one block always kept). Every prunable mask gets a random number of blocks: a level g ~ U(0.1, 1)
per net, then keep fraction g * lognormal(0, 0.5) per mask, clipped to [1 block, all]. So nets span the whole
range (from ~10% to the full nano) and families vary independently of each other, which the fit of the
a_f needs. Export: OpenVINOBackend.export, the same as the `export` stage (fp16 IR, one-to-one head,
640x640, batch 1). The conv list (outside the fx leaves) is read with forward hooks on the exported model:
kernel, stride, output map, c_in, c_out -> LUT family; every point must lie on the LUT grid.
"""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import pi as lut
from .pi import ARCHS, LEVEL, N_NETS, N_STEP, NC, SIGMA

NONCONV = ("Concat", "Add", "Interpolate", "MaxPool")   # data-movement nodes that pruning shrinks, in C


def pit_model(width: float, depth: str):
    from ultralytics.utils import YAML
    from yolopit.search import build_pit_model
    from src.models import arch
    d = Path(tempfile.mkdtemp())
    YAML.save(d / "arch.yaml", arch(width, depth, NC))
    return build_pit_model(str(d / "arch.yaml"), n=N_STEP, verbose=False)


def maskers(wrapper) -> list:
    from yolopit.masks import PITBlockFeaturesMasker
    seen = {}
    for m in wrapper.net.seed.modules():            # shared masks appear once per sharing layer
        if isinstance(m, PITBlockFeaturesMasker) and m.trainable:
            seen.setdefault(id(m), m)
    return list(seen.values())


def randomize(ms: list, seed: int) -> float:
    """Random kept blocks per mask (see module doc); seed < 0: keep everything. Returns the level g."""
    import torch
    rng = np.random.default_rng(max(seed, 0))
    g = 1.0 if seed < 0 else float(rng.uniform(*LEVEL))
    for m in ms:
        nb = len(m.sizes)
        ka = nb - 1 - (1 if m.ordered_remainder else 0)        # keep-alive: the last full block
        if seed < 0:
            keep = nb
        else:
            keep = int(np.clip(round(g * rng.lognormal(0, SIGMA) * nb), 1, nb))
        v = torch.zeros(nb)
        v[ka] = 1
        others = [i for i in range(nb) if i != ka]
        v[list(rng.choice(others, keep - 1, replace=False))] = 1
        with torch.no_grad():
            m.block.copy_(v)
    return g


def conv_layers(model) -> list[tuple]:
    """(name, family, c_in, c_out) of the convs outside C2PSA/PSABlock/Detect, in execution order."""
    import torch
    from yolopit.search import LEAVES
    inside = set()
    for name, mod in model.named_modules():
        if isinstance(mod, LEAVES):
            inside |= {f"{name}.{n}" if n else name for n, _ in mod.named_modules()}
    rows, hooks = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and name not in inside:
            assert mod.groups == 1, (name, mod.groups)
            hooks.append(mod.register_forward_hook(lambda mo, i, o, name=name: rows.append(
                (name, f"k{mo.kernel_size[0]}s{mo.stride[0]}_{o.shape[-1]}", mo.in_channels, mo.out_channels))))
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    for h in hooks:
        h.remove()
    return rows


def nonconv_elems(xml: Path) -> int:
    """Output elements of the NONCONV nodes of the IR: the gamma term of the fit, known before measuring."""
    import openvino as ov
    return sum(int(np.prod(list(op.get_output_shape(0)))) for op in ov.Core().read_model(xml).get_ordered_ops()
               if op.get_type_name() in NONCONV)


def main(d: Path, n: int = N_NETS):
    from src.backends import get_backend
    from src.remote.connection import LocalConnection
    from yolopit.search import PITYOLO
    cfg = lut.make_cfg()                             # rpi5, openvino, fp16, deploy 640: as the LUT
    backend = get_backend("openvino", cfg)
    grid = {(f, ci, co) for f, _, ci, co in lut.points()}
    res, nets = d / "results", d / "runs" / "nets"
    res.mkdir(parents=True, exist_ok=True)
    nets.mkdir(parents=True, exist_ok=True)
    nets_rows, layer_rows = [], []
    for a, (width, depth) in enumerate(ARCHS):
        for j in [-1, *range(n)]:                    # -1: the unpruned architecture, calibrates its C
            name = f"w{width}_{depth}_" + ("full" if j < 0 else f"r{j:03d}")
            seed = -1 if j < 0 else 1000 * a + j     # different masks for every architecture
            w = pit_model(width, depth)              # fresh model: masks of the previous net do not leak
            g = randomize(maskers(w), seed)
            costs = {m: float(v) for m, v in w.real_costs().items()}
            p = PITYOLO.__new__(PITYOLO)             # export_pruned without a search: only model/trainer used
            p.model, p.trainer = w, None
            tmp = Path(tempfile.mkdtemp())
            yo = p.export_pruned(tmp / f"{name}.pt")
            layers = conv_layers(yo.model.eval())
            off = [(f, ci, co) for _, f, ci, co in layers if (f, ci, co) not in grid]
            if off:                                  # D101/D170 and w0.09375 have shapes the nano LUT lacks
                print(name, "fuori dalla griglia della LUT:", off, flush=True)
            dst = nets / name
            shutil.rmtree(dst, ignore_errors=True)
            ir = backend.export(LocalConnection(), cfg, tmp / f"{name}.pt", tmp / "ov")
            shutil.copytree(ir, dst)
            shutil.copy(tmp / f"{name}.pt", dst / f"{name}.pt")
            shutil.rmtree(tmp, ignore_errors=True)
            nets_rows.append({"net": name, "arch": f"w{width}_{depth}", "width": width, "depth": depth,
                              "seed": seed, "level": round(g, 4), "layers": len(layers),
                              **{f"cost_{m}": v for m, v in costs.items()},
                              "nonconv_elems": nonconv_elems(next(dst.glob("*.xml"))),
                              "off_grid": " ".join(f"{f}:{ci}x{co}" for f, ci, co in sorted(set(off)))})
            layer_rows += [{"net": name, "order": i, "layer": ln, "family": f, "c_in": ci, "c_out": co}
                           for i, (ln, f, ci, co) in enumerate(layers)]
            print(name, f"g={g:.2f}", len(layers), "strati", {m: f"{v:.3g}" for m, v in costs.items()}, flush=True)
    for path, rows in ((res / "nets.csv", nets_rows), (res / "layers.csv", layer_rows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), *map(int, sys.argv[2:3]))
