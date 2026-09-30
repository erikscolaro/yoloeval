"""Calibration set e quantizzazione post-training.

Il calibration set deve essere **identico per ogni artefatto**: variarlo fra
quantizzazioni significa attribuire alla quantizzazione differenze che
dipendono dalle immagini scelte. La lista e' generata da uno shuffle con seed
fisso (`quantization.calibration.shuffle_seed`) e salvata accanto all'export,
cosi' e' ispezionabile e riproducibile.

Solo PTQ: la quantization-aware training e' fuori scope.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import yaml

from ..jsonio import atomic_write_json

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _split_dirs(cfg, split: str) -> list[Path]:
    """Directory delle immagini per uno split, dal data yaml di Ultralytics."""
    spec = yaml.safe_load(Path(cfg.dataset.yaml).read_text(encoding="utf-8"))
    root = Path(spec.get("path") or cfg.dataset.local_path)
    entry = spec.get(split)
    if entry is None:
        # Nessun ripiego su `val`: un nome sbagliato calibrerebbe sui dati
        # di valutazione senza che nessuno se ne accorga.
        raise KeyError(f"split {split!r} assente in {cfg.dataset.yaml}")
    entries = entry if isinstance(entry, list) else [entry]
    return [root / e if not Path(e).is_absolute() else Path(e) for e in entries]


def calibration_files(cfg) -> list[Path]:
    """Lista deterministica di immagini di calibrazione.

    Ordinata prima dello shuffle: l'ordine di `glob` dipende dal filesystem,
    quindi senza `sorted` lo stesso seed produrrebbe set diversi su macchine
    diverse.
    """
    calib = cfg.quantization.get("calibration")
    if not calib:
        return []

    images: list[Path] = []
    for d in _split_dirs(cfg, calib.get("source", "val")):
        if d.is_file():
            images.append(d)
            continue
        images.extend(
            p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        raise FileNotFoundError(
            f"nessuna immagine di calibrazione sotto {cfg.dataset.yaml} "
            f"(split {calib.get('source', 'val')})"
        )

    # Il seed della calibrazione e' fisso per costruzione; se il gruppo non lo
    # dichiara si usa quello globale, mai un default diverso per chiamata.
    rng = random.Random(calib.get("shuffle_seed") or cfg.get("seed", 42))
    rng.shuffle(images)
    n = int(calib.get("n_samples", 512))
    return images[:n]


def write_manifest(dst_dir: Path, files: list[Path]) -> Path:
    """Manifesto del calibration set, accanto all'artefatto."""
    payload = {
        "n_samples": len(files),
        "files": [str(f) for f in files],
    }
    return atomic_write_json(Path(dst_dir) / "calibration.json", payload)


def preprocess(path: Path, imgsz: int):
    """Letterbox + normalizzazione, come fa Ultralytics in inferenza.

    Usare un preprocessing diverso da quello di inferenza produce statistiche
    di calibrazione sbagliate e quindi un modello INT8 peggiore del necessario.
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"immagine non leggibile: {path}")
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype("float32") / 255.0
    return np.ascontiguousarray(x)[None]


class CalibrationReader:
    """`CalibrationDataReader` di onnxruntime.quantization.

    Implementato per composizione invece che per ereditarieta': cosi' il
    modulo si importa anche dove `onnxruntime.quantization` non c'e' (sulle
    board, dove non si quantizza).
    """

    def __init__(self, files: list[Path], input_name: str, imgsz: int):
        self.files = list(files)
        self.input_name = input_name
        self.imgsz = imgsz
        self._it = iter(self.files)

    def get_next(self):
        path = next(self._it, None)
        if path is None:
            return None
        return {self.input_name: preprocess(path, self.imgsz)}

    def rewind(self):
        self._it = iter(self.files)


def _unquantize_scalars(path: Path) -> int:
    """Riporta in float le costanti scalari quantizzate (DQ su tensori 0-d).

    onnxruntime quantizza anche le costanti scalari, per esempio la scala
    dell'attention. TensorRT le rifiuta due volte: il parser applica l'asse di
    default 1 a un tensore 0-d ("Axis must be in the range [0, nbDims (0)]") e
    il builder non ammette costanti quantizzate fuori dal pattern peso -> DQ
    di una conv. Si sostituisce il DequantizeLinear con il valore float che
    produrrebbe: e' un moltiplicatore, non un tensore su cui gira un kernel,
    quindi tenerlo in float non toglie niente alla quantizzazione del grafo.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}

    def scalar(name):
        init = inits.get(name)
        return numpy_helper.to_array(init) if init is not None and not init.dims else None

    folded = []
    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        q = scalar(node.input[0])
        scale = scalar(node.input[1]) if len(node.input) > 1 else None
        if q is None or scale is None:
            continue
        zp = scalar(node.input[2]) if len(node.input) > 2 and node.input[2] else 0
        value = (q.astype(np.float32) - np.float32(zp)) * scale.astype(np.float32)
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, dtype=scale.dtype), node.output[0])
        )
        folded.append(node)
    for node in folded:
        graph.node.remove(node)
    if folded:
        # gli input quantizzati rimasti senza consumatori
        used = {i for n in graph.node for i in n.input}
        orphans = [i for i in graph.initializer
                   if i.name not in used and i.name not in {o.name for o in graph.output}
                   and any(i.name in n.input for n in folded)]
        for init in orphans:
            graph.initializer.remove(init)
        onnx.save(model, str(path))
        log.info("%d costanti scalari quantizzate riportate in float", len(folded))
    return len(folded)


def quantize_onnx_static(src: Path, dst: Path, cfg, symmetric: bool = False) -> Path:
    """PTQ statica su ONNX, formato QDQ.

    QDQ (e non QOperator) perche' e' il formato che ONNX Runtime, TensorRT e
    OpenVINO sanno tutti consumare: lo stesso file resta confrontabile fra
    backend.

    `symmetric=True` produce il QDQ che accetta TensorRT: attivazioni INT8
    con zero point nullo (con UINT8 il parser si ferma con "unsupported input
    type of UINT8") e bias non quantizzati. Immagini e metodo di calibrazione
    restano gli stessi, cambia solo la rappresentazione.
    """
    import onnx
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    args = dict(cfg.quantization.backend_args.get("onnxruntime", {}) or {})
    files = calibration_files(cfg)
    if not files:
        raise ValueError("calibration set vuoto: controllare quantization.calibration")

    model = onnx.load(str(src))
    input_name = model.graph.input[0].name
    reader = CalibrationReader(files, input_name, int(cfg.model.imgsz))

    prepped = dst.with_suffix(".prep.onnx")
    quant_pre_process(str(src), str(prepped), skip_symbolic_shape=False)

    fmt = QuantFormat.QDQ if str(args.get("quant_format", "QDQ")).upper() == "QDQ" \
        else QuantFormat.QOperator
    quantize_static(
        str(prepped),
        str(dst),
        reader,
        quant_format=fmt,
        per_channel=bool(args.get("per_channel", True)),
        reduce_range=bool(args.get("reduce_range", False)),
        activation_type=QuantType.QInt8 if symmetric else QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        # Per TensorRT: bias in FP32, perche' il parser non accetta un
        # DequantizeLinear su bias INT32; e' TensorRT a fonderlo nella conv.
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QuantizeBias": False}
        if symmetric else None,
    )
    prepped.unlink(missing_ok=True)
    if symmetric:
        _unquantize_scalars(dst)
    write_manifest(dst.parent, files)
    log.info("quantizzazione INT8 completata su %d immagini", len(files))
    return dst
