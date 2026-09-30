"""Complessita' di un modello per il roofline: MAC e byte spostati, dal grafo ONNX.

- MAC: Conv (uscita x Cin/groups x kh x kw) e MatMul/Gemm (M x N x K). Il resto (attivazioni,
  somme, concat, resize) non conta come calcolo.
- Byte: il traffico obbligatorio di ogni layer di calcolo, cioe' pesi + input + output di ogni
  Conv/MatMul/Gemm, assumendo che le operazioni elementwise siano fuse nel layer (come fanno
  i runtime). Non conta il riuso fra layer in cache: e' una stima per eccesso del traffico
  verso la memoria, quindi l'intensita' aritmetica e' una stima per difetto.
- La precisione scala i byte: fp32 = 4, fp16 = 2, int8 = 1 byte per peso e per attivazione.

Si calcola una volta per pesi e risoluzione da un ONNX fp32 esportato sulla workstation, e
vale per ogni backend: la precisione entra solo nei byte.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BYTES = {"fp32": 4, "fp16": 2, "int8": 1}
COMPUTE_OPS = ("Conv", "MatMul", "Gemm")


def _shapes(model) -> dict:
    from onnx import shape_inference

    g = shape_inference.infer_shapes(model).graph
    out = {}
    for v in list(g.input) + list(g.value_info) + list(g.output):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in v.type.tensor_type.shape.dim]
        out[v.name] = dims
    return out


def onnx_complexity(path: str | Path) -> dict:
    """{macs, weight_elems, act_elems, params, compute_layers} di un ONNX a batch 1."""
    import onnx

    model = onnx.load(str(path))
    shp = _shapes(model)
    inits = {i.name: list(i.dims) for i in model.graph.initializer}
    macs = weight_elems = act_elems = 0
    layers = 0
    for n in model.graph.node:
        if n.op_type not in COMPUTE_OPS:
            continue
        layers += 1
        out = shp.get(n.output[0])
        if out is None:
            raise ValueError(f"forma sconosciuta per l'uscita di {n.name or n.op_type}")
        if n.op_type == "Conv":
            w = inits.get(n.input[1]) or shp.get(n.input[1])
            macs += int(np.prod(out)) * int(np.prod(w[1:]))     # Cin/groups * kh * kw
        else:                                                  # MatMul / Gemm
            a = shp.get(n.input[0]) or inits.get(n.input[0])
            k = a[-1]
            if n.op_type == "Gemm":
                trans_a = next((x.i for x in n.attribute if x.name == "transA"), 0)
                k = a[0] if trans_a else a[-1]
            macs += int(np.prod(out)) * int(k)
        for name in n.input:
            if not name:
                continue
            if name in inits:
                weight_elems += int(np.prod(inits[name]))
            elif name in shp:
                act_elems += int(np.prod(shp[name]))
        act_elems += int(np.prod(out))
    params = sum(int(np.prod(d)) for d in inits.values())
    return {"macs": macs, "weight_elems": weight_elems, "act_elems": act_elems,
            "params": params, "compute_layers": layers}


def bytes_moved(cx: dict, precision: str) -> int:
    b = BYTES.get(precision, 4)
    return int((cx["weight_elems"] + cx["act_elems"]) * b)


def with_precision(cx: dict, precision: str) -> dict:
    """Complessita' + byte e intensita' (MAC/byte) alla precisione indicata."""
    out = dict(cx)
    out["precision"] = precision
    out["bytes"] = bytes_moved(cx, precision)
    out["intensity_mac_per_byte"] = cx["macs"] / out["bytes"] if out["bytes"] else None
    return out


def ensure_complexity(cfg, src_pt: Path, key: str) -> dict:
    """Complessita' dei pesi `src_pt` a model.imgsz, con cache in artifacts/complexity/.

    Esporta un ONNX fp32 con Ultralytics (le stesse impostazioni dell'export ONNX del
    backend) solo la prima volta per quella chiave dei pesi e risoluzione."""
    from ..jsonio import atomic_write_json, read_json

    imgsz = int(cfg.model.imgsz)
    root = Path(cfg.artifacts_dir) / "complexity"
    path = root / f"{key}_{imgsz}.json"
    cached = read_json(path)
    if cached:
        return cached
    from ultralytics import YOLO

    work = root / "_onnx" / f"{key}_{imgsz}"
    work.mkdir(parents=True, exist_ok=True)
    local_pt = work / Path(src_pt).name
    if not local_pt.exists():
        import shutil

        shutil.copy2(src_pt, local_pt)
    onnx_path = YOLO(str(local_pt), task="detect").export(
        format="onnx", imgsz=imgsz, batch=1, simplify=True, dynamic=False, device="cpu")
    cx = onnx_complexity(onnx_path)
    cx.update(imgsz=imgsz, weights_key=key, source="onnx fp32, Ultralytics export")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, cx)
    log.info("complessita' %s @%d: %.3f GMAC, %d layer di calcolo", key, imgsz,
             cx["macs"] / 1e9, cx["compute_layers"])
    return cx
