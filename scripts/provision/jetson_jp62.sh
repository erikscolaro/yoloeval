#!/usr/bin/env bash
# Provisioning di una Jetson Orin con JetPack 6.2.
#
# Idempotente: se l'ambiente c'e' gia' ed e' coerente, i passi non fanno nulla.
# I wheel di PyTorch e ONNX Runtime da PyPI NON funzionano su aarch64 Tegra:
# servono i build NVIDIA per la specifica versione di JetPack. TensorRT e'
# preinstallato con JetPack — si usa quello, non si installa da pip.
set -euo pipefail

WORKDIR="${BENCH_WORKDIR:-$HOME/bench}"
REQUIREMENTS="${BENCH_REQUIREMENTS:-/tmp/requirements.txt}"
EXPECTED_JETPACK="${BENCH_JETPACK:-6.2}"
VENV="$WORKDIR/.venv"

say() { echo "[provision][jetson] $*"; }

# 1. La versione di JetPack deve corrispondere, e il controllo va fatto ORA:
#    un mismatch scoperto a meta' sweep si presenta come un import error.
[[ -f /etc/nv_tegra_release ]] || { echo "non sembra una Jetson" >&2; exit 1; }
RELEASE=$(head -1 /etc/nv_tegra_release)
say "$RELEASE"
case "$EXPECTED_JETPACK" in
  6.2) grep -q "R36" <<<"$RELEASE" || { echo "JetPack atteso 6.2 (L4T R36): trovato $RELEASE" >&2; exit 1; } ;;
  *) say "nessun controllo per JetPack $EXPECTED_JETPACK" ;;
esac

mkdir -p "$WORKDIR"

# 2. venv con --system-site-packages: serve a vedere il TensorRT di sistema.
if [[ ! -d "$VENV" ]]; then
  say "creo il venv in $VENV"
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# 3. Wheel NVIDIA di torch e torchvision per questa JetPack.
if ! python -c "import torch" 2>/dev/null; then
  say "installo torch dall'indice NVIDIA per JetPack ${EXPECTED_JETPACK}"
  python -m pip install --no-cache-dir \
    --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
    torch torchvision
fi

# 4. onnxruntime-gpu dal build NVIDIA per Jetson.
if ! python -c "import onnxruntime" 2>/dev/null; then
  say "installo onnxruntime-gpu per Jetson"
  python -m pip install --no-cache-dir \
    --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
    onnxruntime-gpu
fi

# 5. Requirements generici (senza torch/onnxruntime, vedi il file).
say "installo i requirements generici"
python -m pip install --no-cache-dir -r "$REQUIREMENTS"

# 6-7. Verifiche: senza queste il provisioning "riesce" e lo sweep fallisce.
python - <<'PY'
import sys
import torch
print(f"[provision][jetson] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
try:
    import tensorrt
    print(f"[provision][jetson] tensorrt {tensorrt.__version__}")
except ImportError:
    print("[provision][jetson] TensorRT non importabile dal venv: "
          "il venv e' stato creato senza --system-site-packages?", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("[provision][jetson] CUDA non disponibile", file=sys.stderr)
    sys.exit(1)
PY

if command -v trtexec >/dev/null 2>&1; then
  say "trtexec: $(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  say "trtexec: /usr/src/tensorrt/bin/trtexec (non in PATH)"
else
  echo "trtexec non trovato: atteso in /usr/src/tensorrt/bin" >&2
  exit 1
fi

# sudo senza password per i soli comandi di tuning, altrimenti lo sweep si
# blocca a ogni cambio di profilo.
sudo -n nvpmodel -q >/dev/null 2>&1 || \
  say "ATTENZIONE: sudo senza password non configurato per nvpmodel"

say "ok"
