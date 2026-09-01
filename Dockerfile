FROM python:3.13-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG SPM12_RELEASE=7771

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SPM12_DIR=/opt/spm12

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates curl gzip libgomp1 libglib2.0-0 libgl1 octave liboctave-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/spm12 \
    && curl -fsSL --retry 5 "https://github.com/spm/spm12/archive/r${SPM12_RELEASE}.tar.gz" \
       | tar -xzC /opt/spm12 --strip-components=1 \
    && cd / \
    && curl -fsSL --retry 5 "https://raw.githubusercontent.com/spm/spm-octave/main/spm12_r${SPM12_RELEASE}.patch" \
       | patch -p0 \
    && make -C /opt/spm12/src PLATFORM=octave distclean \
    && make -C /opt/spm12/src PLATFORM=octave \
    && make -C /opt/spm12/src PLATFORM=octave install \
    && ln -sf /opt/spm12/bin/spm12-octave /usr/local/bin/spm12 \
    && rm -f /opt/spm12/src/*.{mex,o,a} \
    && apt-get purge -y build-essential curl liboctave-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

WORKDIR /opt/fastbrainage
COPY pyproject.toml README.md ./
COPY src ./src
COPY assets ./assets
COPY matlab ./matlab
COPY docker ./docker
COPY configs/exp2.json ./configs/exp2.json
COPY models/exp2.joblib models/exp2.json ./models/

RUN pip install --no-cache-dir . \
    && chmod +x /opt/fastbrainage/docker/*.sh

ENTRYPOINT ["/opt/fastbrainage/docker/entrypoint.sh"]
