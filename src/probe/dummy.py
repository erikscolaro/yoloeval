"""Reti dummy in ONNX per il probe: L blocchi conv + SiLU + residual, tutti a C canali.

Una catena e non una conv isolata: con canali non allineati il costo vero spesso e' nelle
conversioni di layout FRA un layer e l'altro, che una conv da sola non mostra. Il grafo si
costruisce con `onnx.helper`, senza PyTorch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OPSET = 17


@dataclass(frozen=True)
class Shape:
    """Forma dei layer della rete dummy."""

    kernel: int = 3
    hw: int = 40                 # lato della feature map (40 = P4 di YOLO a 640)
    depthwise: bool = False
    layers: int = 8

    @property
    def name(self) -> str:
        kind = "dw" if self.depthwise else "conv"
        return f"{kind}{self.kernel}x{self.kernel}@{self.hw}x{self.layers}"


def costs(shape: Shape, c: int) -> dict:
    """MAC e parametri della rete dummy (stessa formula di PLiNIO, bias compreso)."""
    k2 = shape.kernel * shape.kernel
    per_out = (k2 if shape.depthwise else c * k2) + 1
    macs = shape.layers * c * per_out * shape.hw * shape.hw
    params = shape.layers * c * per_out
    return {"macs": int(macs), "params": int(params)}


def build(shape: Shape, c: int, seed: int = 0):
    """ModelProto fp32 con input [1, C, hw, hw]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    groups = c if shape.depthwise else 1
    fan_in = (1 if shape.depthwise else c) * shape.kernel * shape.kernel
    nodes, inits = [], []
    x = "input"
    pad = shape.kernel // 2
    for i in range(shape.layers):
        w = rng.normal(0, 1 / np.sqrt(fan_in), (c, c // groups, shape.kernel, shape.kernel))
        inits += [numpy_helper.from_array(w.astype(np.float32), f"w{i}"),
                  numpy_helper.from_array(np.zeros(c, np.float32), f"b{i}")]
        # nomi dei nodi espliciti: la conversione fp16 li usa per i Cast che aggiunge
        nodes += [
            helper.make_node("Conv", [x, f"w{i}", f"b{i}"], [f"c{i}"], name=f"conv{i}",
                             group=groups, kernel_shape=[shape.kernel] * 2, pads=[pad] * 4),
            helper.make_node("Sigmoid", [f"c{i}"], [f"s{i}"], name=f"sigmoid{i}"),
            helper.make_node("Mul", [f"c{i}", f"s{i}"], [f"a{i}"], name=f"silu{i}"),
            helper.make_node("Add", [f"a{i}", x], [f"y{i}"], name=f"residual{i}"),
        ]
        x = f"y{i}"
    nodes.append(helper.make_node("Identity", [x], ["output"], name="output"))
    dims = [1, c, shape.hw, shape.hw]
    graph = helper.make_graph(
        nodes, f"probe_{shape.name}_c{c}", [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, dims)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, dims)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class _RandomReader:
    """Calibrazione int8 su input casuali: per la latenza conta la forma, non i valori."""

    def __init__(self, dims, n=8, seed=0):
        rng = np.random.default_rng(seed)
        self._data = [rng.normal(0, 1, dims).astype(np.float32) for _ in range(n)]
        self._it = iter(self._data)

    def get_next(self):
        x = next(self._it, None)
        return None if x is None else {"input": x}

    def rewind(self):
        self._it = iter(self._data)


def artifact(shape: Shape, c: int, precision: str, out_dir: Path, int8_args: dict | None = None
             ) -> Path:
    """ONNX della rete dummy alla precisione richiesta (cache per contenuto)."""
    import onnx

    spec = {"shape": asdict(shape), "c": c, "precision": precision, "int8": int8_args or {},
            "opset": OPSET}
    tag = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    path = Path(out_dir) / f"{shape.name}_c{c}_{precision}_{tag}.onnx"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build(shape, c)
    if precision == "fp32":
        onnx.save(model, str(path))
    elif precision == "fp16":
        from onnxconverter_common import float16

        onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), str(path))
    elif precision == "int8":
        from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                              quantize_static)

        args = dict(int8_args or {})
        fp32 = path.with_suffix(".fp32.onnx")
        onnx.save(model, str(fp32))
        quantize_static(
            str(fp32), str(path), _RandomReader([1, c, shape.hw, shape.hw]),
            quant_format=QuantFormat.QDQ if str(args.get("quant_format", "QDQ")).upper()
            == "QDQ" else QuantFormat.QOperator,
            per_channel=bool(args.get("per_channel", True)),
            reduce_range=bool(args.get("reduce_range", False)),
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax)
        fp32.unlink(missing_ok=True)
    else:
        raise ValueError(f"precisione non supportata dal probe: {precision}")
    return path
