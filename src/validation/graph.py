"""Ispezione dell'artefatto esportato: quale testa contiene davvero.

YOLO26 ha un'architettura a doppia testa: una one-to-one per l'inferenza
end-to-end senza NMS e una one-to-many tradizionale che richiede NMS. Il
percorso end-to-end viene **disabilitato automaticamente** da alcune
combinazioni di runtime e quantizzazione, con fallback silenzioso sulla testa
one-to-many.

Ignorarlo significa attribuire alla quantizzazione una differenza di latenza
che dipende invece dal post-processing. Quindi l'artefatto si ispeziona, non si
assume.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

#: op che indicano NMS dentro il grafo
NMS_OPS = ("NonMaxSuppression", "EfficientNMS", "NMS_TRT", "BatchedNMS")

#: combinazioni note che disabilitano il percorso end-to-end
KNOWN_E2E_FALLBACKS = (
    ("tensorrt", lambda env, cfg: _trt_lt(env.get("tensorrt"), (8, 5, 0)),
     "trt < 8.5.0"),
    ("tensorrt", lambda env, cfg: (
        cfg.quantization.precision == "int8"
        and _trt_ge(env.get("tensorrt"), (10, 3, 0))
        and str(env.get("jetpack", "")).startswith(("6", "36"))),
     "trt 10.3 + int8 + jetpack6"),
    ("executorch", lambda env, cfg: cfg.quantization.precision in ("int8", "w8a16"),
     "litert 8 bit"),
)


def _version_tuple(value) -> tuple[int, ...] | None:
    if not value:
        return None
    parts = re.findall(r"\d+", str(value))
    return tuple(int(p) for p in parts[:3]) if parts else None


def _trt_lt(value, ref) -> bool:
    v = _version_tuple(value)
    return bool(v and v < ref)


def _trt_ge(value, ref) -> bool:
    v = _version_tuple(value)
    return bool(v and v >= ref)


def e2e_fallback_reason(cfg, env: dict) -> str | None:
    """Motivo noto per cui l'export potrebbe aver perso il percorso e2e.

    E' documentazione del risultato, non una previsione: la verita' resta
    quella letta dal grafo.
    """
    for backend, predicate, reason in KNOWN_E2E_FALLBACKS:
        if cfg.backend.name != backend:
            continue
        try:
            if predicate(env or {}, cfg):
                return reason
        except Exception:  # un env incompleto non deve far fallire l'export
            log.debug("predicato e2e non valutabile", exc_info=True)
    return None


def classify_head(output_shapes: list[tuple], nms_in_graph: bool,
                  nc: int | None = None) -> tuple[str, bool]:
    """Dalla forma degli output alla testa effettivamente esportata.

    * one-to-one (e2e):  [1, max_det, 6]  — box, score, classe gia' filtrati
    * one-to-many:       [1, 4+nc, anchors] — richiede NMS a valle
    """
    if nms_in_graph:
        return "one_to_many+nms", False
    for shape in output_shapes:
        dims = [d for d in shape if isinstance(d, int)]
        if len(shape) == 3 and dims and dims[-1] == 6:
            return "one_to_one", True
        if len(shape) == 3 and nc and shape[1] == 4 + nc:
            return "one_to_many", False
    if len(output_shapes) == 1 and len(output_shapes[0]) == 3:
        # forma inattesa: non si indovina, si dichiara sconosciuta
        return "unknown", False
    return "unknown", False


def inspect_onnx(path) -> dict:
    """{'actual_e2e', 'precision', 'input_shape', 'output_names', ...}."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    op_types = {n.op_type for n in graph.node}
    nms_in_graph = any(any(k in op for k in NMS_OPS) for op in op_types)

    def shape_of(value_info):
        dims = []
        for d in value_info.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or -1))
        return tuple(dims)

    input_shape = shape_of(graph.input[0]) if graph.input else ()
    output_shapes = [shape_of(o) for o in graph.output]
    nc = None
    for shape in output_shapes:
        if len(shape) == 3 and isinstance(shape[1], int) and shape[1] > 4:
            nc = shape[1] - 4
            break

    if "QuantizeLinear" in op_types or "QLinearConv" in op_types:
        precision = "int8"
    elif any(
        init.data_type == onnx.TensorProto.FLOAT16 for init in graph.initializer
    ):
        precision = "fp16"
    else:
        precision = "fp32"

    head, actual_e2e = classify_head(output_shapes, nms_in_graph, nc)
    return {
        "actual_e2e": actual_e2e,
        "head": head,
        "nms_in_graph": nms_in_graph,
        "precision": precision,
        "input_shape": list(input_shape),
        "output_names": [o.name for o in graph.output],
        "output_shapes": [list(s) for s in output_shapes],
        "opset": model.opset_import[0].version if model.opset_import else None,
        "source": "onnx",
    }


_BINDING_RE = re.compile(
    r"^\[.\]\s+(?P<io>Input|Output)\(s\)s?:\s*(?P<body>.+)$", re.MULTILINE
)
_TENSOR_RE = re.compile(
    r"(?P<name>[\w./:-]+)\s*\((?P<dtype>\w+)\s*\[(?P<dims>[\d,x\s]+)\]\)"
)


def parse_trtexec_bindings(stdout: str) -> dict:
    """Legge input e output dell'engine dal verbose di `trtexec --loadEngine`.

    Un engine non e' ispezionabile sulla workstation: e' compilato per una
    specifica GPU e versione di libreria, quindi l'ispezione avviene sulla
    board e il risultato viaggia come sidecar JSON.
    """
    inputs, outputs = [], []
    for m in _BINDING_RE.finditer(stdout or ""):
        target = inputs if m.group("io") == "Input" else outputs
        for t in _TENSOR_RE.finditer(m.group("body")):
            dims = tuple(
                int(d) for d in re.split(r"[,x]", t.group("dims")) if d.strip().isdigit()
            )
            target.append({"name": t.group("name"), "dtype": t.group("dtype"),
                           "shape": dims})
    return {"inputs": inputs, "outputs": outputs}


def inspect_from_bindings(bindings: dict, nc: int | None = None) -> dict:
    outputs = bindings.get("outputs", [])
    shapes = [tuple(o["shape"]) for o in outputs]
    dtypes = {str(o.get("dtype", "")).lower() for o in outputs}
    precision = (
        "int8" if "int8" in dtypes else "fp16" if "half" in dtypes or "fp16" in dtypes
        else "fp32"
    )
    head, actual_e2e = classify_head(shapes, nms_in_graph=False, nc=nc)
    return {
        "actual_e2e": actual_e2e,
        "head": head,
        "nms_in_graph": False,
        "precision": precision,
        "input_shape": list(bindings["inputs"][0]["shape"]) if bindings.get("inputs")
        else [],
        "output_names": [o["name"] for o in outputs],
        "output_shapes": [list(s) for s in shapes],
        "source": "trtexec_bindings",
    }
