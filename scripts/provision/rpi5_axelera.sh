#!/usr/bin/env bash
# Provisioning della parte Axelera Metis di un Raspberry Pi 5 (hardware=rpi5_axelera):
# driver, SDK e container. Il resto (venv, onnxruntime_perf_test) lo fa rpi5.sh, che
# gira subito dopo (provision.script in conf/hardware/rpi5_axelera.yaml).
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
IMAGE="${BENCH_AXELERA_IMAGE:-yolo-bench-axelera:ubuntu22}"
SDK_VERSION="${BENCH_SDK_VERSION:-1.8.0}"

say() { echo "[provision][rpi5_axelera] $*"; }

mkdir -p "$WORKDIR"

# 1. Repository apt di Axelera, con chiave GPG.
#    Il repository non segue i codename di Debian (trixie non c'e'): ha stable, ubuntu22 e
#    ubuntu24. stable si ferma a metis-dkms 1.2.x, sotto il minimo per l'SDK; ubuntu22 e'
#    la base del container dell'SDK. metis-dkms e' arch all e si compila sul kernel del Pi.
APT_DIST="${BENCH_AXELERA_APT_DIST:-ubuntu22}"
if [[ ! -f /etc/apt/sources.list.d/axelera.list ]]; then
  say "aggiungo il repository apt di Axelera ($APT_DIST)"
  # --batch --yes: un tentativo fallito puo' aver lasciato il file, gpg non deve chiedere
  curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/axelera.gpg
  echo "deb [signed-by=/usr/share/keyrings/axelera.gpg] " \
       "https://software.axelera.ai/artifactory/axelera-apt-source ${APT_DIST} main" \
    | sudo tee /etc/apt/sources.list.d/axelera.list >/dev/null
  sudo apt-get update
fi

# 2. Kernel module sull'host.
say "installo metis-dkms"
sudo apt-get install -y metis-dkms
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

say "axelera ok"
