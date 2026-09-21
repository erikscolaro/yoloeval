# Container per il Voyager SDK su Raspberry Pi 5.
#
# Raspberry Pi OS non e' una piattaforma supportata da Axelera, quindi l'SDK
# gira in un Ubuntu 22.04. Il kernel driver (metis-dkms) resta sull'host:
# e' un modulo kernel, nel container non ci puo' stare.
#
# Avvio con accesso al device PCIe e ai volumi di lavoro:
#   docker run --device /dev/metis0 -v <workdir>:/bench --network host
# Il nome effettivo del device va verificato con `ls /dev | grep -i metis`.
FROM ubuntu:22.04

ARG SDK_VERSION=1.8.0
ARG AXELERA_INDEX=https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

COPY scripts/requirements/axelera-container.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip wheel \
    && sed -i "s/==1\.8\.0/==${SDK_VERSION}/g" /tmp/requirements.txt \
    && python3 -m pip install --extra-index-url "${AXELERA_INDEX}" \
        -r /tmp/requirements.txt

WORKDIR /bench
CMD ["sleep", "infinity"]
