#!/usr/bin/env bash
# Provisioning della parte Axelera Metis di un Raspberry Pi 5 (hardware=rpi5_axelera):
# driver, docker, SDK e container. Il resto (venv, onnxruntime_perf_test) lo fa rpi5.sh, che
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
die() { echo "[provision][rpi5_axelera] $*" >&2; exit 1; }

# Il proxy della rete interna cade anche per minuti: ogni passo che scarica si riprova
# per RETRY_MINUTES (da stage.retry_minutes, default 30) prima di arrendersi.
retry() {
  local what=$1; shift
  local deadline=$(( $(date +%s) + ${RETRY_MINUTES:-30} * 60 )) attempt=1
  until "$@"; do
    (( $(date +%s) >= deadline )) && die "$what: fallito per ${RETRY_MINUTES:-30} minuti"
    say "$what fallito (tentativo $attempt), riprovo fra 30 s"
    attempt=$((attempt + 1))
    sleep 30
  done
}

# e dentro ogni tentativo apt riprova gia' da solo sulle cadute brevi
echo 'Acquire::Retries "10";' | sudo tee /etc/apt/apt.conf.d/80retries >/dev/null

mkdir -p "$WORKDIR"

# 1. Repository apt di Axelera, con chiave GPG.
#    Il repository non segue i codename di Debian (trixie non c'e'): ha stable, ubuntu22 e
#    ubuntu24. stable si ferma a metis-dkms 1.2.x, sotto il minimo per l'SDK; ubuntu22 e'
#    la base del container dell'SDK. metis-dkms e' arch all e si compila sul kernel del Pi.
APT_DIST="${BENCH_AXELERA_APT_DIST:-ubuntu22}"
if [[ ! -f /etc/apt/sources.list.d/axelera.list ]]; then
  say "aggiungo il repository apt di Axelera ($APT_DIST)"
  # --batch --yes: un tentativo fallito puo' aver lasciato il file, gpg non deve chiedere
  retry "chiave gpg Axelera" bash -c 'set -o pipefail
    curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public \
      | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/axelera.gpg'
  echo "deb [signed-by=/usr/share/keyrings/axelera.gpg] " \
       "https://software.axelera.ai/artifactory/axelera-apt-source ${APT_DIST} main" \
    | sudo tee /etc/apt/sources.list.d/axelera.list >/dev/null
  retry "apt update" sudo apt-get update
fi

# 2. Kernel module sull'host.
say "installo metis-dkms"
retry "apt metis-dkms" sudo apt-get install -y metis-dkms
sudo modprobe metis || true

# 3. Health check sull'host: se il device non c'e', fermarsi qui. `axdevice` non e' sull'host
#    (metis-dkms e i pacchetti axelera-runtime/device non lo installano): sta nell'SDK, dentro
#    il container, e lo esegue il passo 6.
if ! compgen -G "/dev/metis*" >/dev/null; then
  echo "nessun /dev/metis*: il driver non vede la scheda Metis." >&2
  echo "Verificare: metis-dkms >= 1.6.2 per SDK ${SDK_VERSION}, modprobe metis, " >&2
  echo "e che la scheda sia inserita nello slot PCIe." >&2
  exit 1
fi
say "device: $(ls /dev/metis* | head -3 | tr '\n' ' ')"

# 4. Docker sull'host. Il proxy del demone e' gia' in /etc/systemd/system/docker.service.d
#    (lo scrive provision.py prima di questo script), quindi parte con il proxy giusto.
if ! command -v docker >/dev/null; then
  say "installo docker"
  retry "apt docker.io" sudo apt-get install -y docker.io
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
# il gruppo appena aggiunto non vale in questa shell: `sg` lo applica, tenendo HOME e il
# ~/.docker/config.json col proxy (con `sudo docker` andrebbe perso)
docker() { sg docker -c "$(printf '%q ' docker "$@")"; }

# 5. Container Ubuntu 22.04 con il Voyager SDK.
#    Il Dockerfile e i requirements arrivano nel workdir da ensure_support_files:
#    sulla board esiste solo questo script, non l'albero del repo.
DOCKERFILE="$WORKDIR/scripts/docker/axelera.Dockerfile"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  [[ -f "$DOCKERFILE" ]] || { echo "Dockerfile non trovato in $DOCKERFILE" >&2; exit 1; }
  say "costruisco l'immagine $IMAGE"
  # un tentativo fallito a meta' riparte dalla cache dei layer gia' costruiti
  retry "docker build" docker build -t "$IMAGE" \
    --build-arg SDK_VERSION="$SDK_VERSION" \
    -f "$DOCKERFILE" \
    "$WORKDIR"
fi

# 6. Verifica end-to-end dentro il container.
DEVICE=$(ls /dev/metis* 2>/dev/null | head -1)
[[ -n "$DEVICE" ]] || { echo "nessun /dev/metis*" >&2; exit 1; }
say "verifica dentro il container (device $DEVICE)"
docker run --rm --device "$DEVICE" -v "$WORKDIR:/bench" --network host "$IMAGE" \
  bash -lc 'axdevice && python -c "import axelera; print(axelera.__version__)"'

say "axelera ok"
