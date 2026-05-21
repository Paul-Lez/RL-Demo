FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/workspace/.matplotlib \
    JUPYTER_TOKEN=reinforce

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# Install the CPU-only PyTorch wheel first, then the remaining notebook stack.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r /tmp/requirements.txt

RUN if ! getent group "${GID}" > /dev/null; then groupadd --gid "${GID}" app; fi \
    && useradd --uid "${UID}" --gid "${GID}" --create-home app \
    && mkdir -p /workspace \
    && chown -R "${UID}:${GID}" /workspace

USER app

EXPOSE 8888

ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=${JUPYTER_TOKEN:-reinforce}"]
