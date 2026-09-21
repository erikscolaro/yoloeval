#!/usr/bin/env bash
# Provisioning di un Raspberry Pi 5 con acceleratore Axelera Metis.
#
# Il kernel driver vive sull'host, l'SDK dentro un container Ubuntu 22.04:
# Raspberry Pi OS non e' una piattaforma supportata da Axelera.
#
# Driver e SDK si aggiornano separatamente e **nulla verifica la coerenza al
# momento dell'installazione**: un driver sotto la versione minima lascia il
# device inaccessibile con un errore di connessione. Per questo `axdevice` e'
# il passo che non si puo' saltare.
set -euo pipefail

WORKDIR="${BENCH_WORKDIR:-$HOME/bench}"
REQUIREMENTS="${BENCH_REQUIREMENTS:-/tmp/requirements.txt}"
IMAGE="${BENCH_AXELERA_IMAGE:-yolo-bench-axelera:ubuntu22}"
SDK_VERSION="${BENCH_SDK_VERSION:-1.8.0}"
VENV="$WORKDIR/.venv"

say() { echo "[provision][rpi5] $*"; }

mkdir -p "$WORKDIR"

# 1. Repository apt di Axelera, con chiave GPG.
if [[ ! -f /etc/apt/sources.list.d/axelera.list ]]; then
  say "aggiungo il repository apt di Axelera"
  curl -fsSL https://software.axelera.ai/artifactory/api/gpg/key/public \
    | sudo gpg --dearmor -o /usr/share/keyrings/axelera.gpg
  . /etc/os-release
  echo "deb [signed-by=/usr/share/keyrings/axelera.gpg] " \
       "https://software.axelera.ai/artifactory/axelera-apt-source ${VERSION_CODENAME} main" \
    | sudo tee /etc/apt/sources.list.d/axelera.list >/dev/null
  sudo apt-get update
fi

# 2. Kernel module sull'host + libgl1 (richiesto da OpenCV, non incluso via pip).
say "installo metis-dkms e libgl1"
sudo apt-get install -y metis-dkms libgl1
sudo modprobe metis || true

# 3. Health check: se il device non si vede, fermarsi qui.
if ! axdevice >/dev/null 2>&1; then
  echo "axdevice non vede la scheda Metis." >&2
  echo "Verificare: metis-dkms >= 1.6.2 per SDK ${SDK_VERSION}, modprobe metis, " >&2
  echo "e che la scheda sia inserita nello slot PCIe." >&2
  exit 1
fi
say "axdevice: $(axdevice | head -3 | tr '\n' ' ')"

# 4. Container Ubuntu 22.04 con il Voyager SDK.
#    Il Dockerfile e i requirements arrivano nel workdir da ensure_support_files:
#    sulla board esiste solo questo script, non l'albero del repo.
DOCKERFILE="$WORKDIR/scripts/docker/axelera.Dockerfile"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  [[ -f "$DOCKERFILE" ]] || { echo "Dockerfile non trovato in $DOCKERFILE" >&2; exit 1; }
  say "costruisco l'immagine $IMAGE"
  docker build -t "$IMAGE" \
    --build-arg SDK_VERSION="$SDK_VERSION" \
    -f "$DOCKERFILE" \
    "$WORKDIR"
fi

# 5-6. Verifica end-to-end dentro il container.
DEVICE=$(ls /dev/metis* 2>/dev/null | head -1)
[[ -n "$DEVICE" ]] || { echo "nessun /dev/metis*" >&2; exit 1; }
say "verifica dentro il container (device $DEVICE)"
docker run --rm --device "$DEVICE" -v "$WORKDIR:/bench" --network host "$IMAGE" \
  bash -lc 'axdevice && python -c "import axelera; print(axelera.__version__)"'

# 7. venv host separato, per le celle CPU-only che non usano il container.
if [[ ! -d "$VENV" ]]; then
  say "creo il venv host in $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir -r "$REQUIREMENTS"

sudo -n true 2>/dev/null || \
  say "ATTENZIONE: sudo senza password non configurato (serve per cpufreq)"

say "ok"
