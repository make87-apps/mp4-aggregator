FROM ghcr.io/make87/debian:bookworm AS base-image

ARG VIRTUAL_ENV=/make87/venv

RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
        build-essential \
        python3-pip \
        python3 \
        libpython3-dev \
        python3-venv \
        ffmpeg \
    && python3 -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN ${VIRTUAL_ENV}/bin/python3 -m pip install -U pip \
    && ${VIRTUAL_ENV}/bin/python3 -m pip install .

ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]
