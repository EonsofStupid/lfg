FROM rust:1.95-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build --release -p lfg-server

FROM node:22-bookworm-slim AS ui-builder

WORKDIR /ui/master-clyffy-web

COPY ui/master-clyffy-web/package.json ui/master-clyffy-web/package-lock.json ./
RUN npm ci

COPY ui/master-clyffy-web ./
RUN npm run build

FROM debian:bookworm-slim AS runtime

ARG BUILD_VERSION=0.0.0-dev
ARG BUILD_REVISION=unknown
ARG BUILD_CREATED=unknown

LABEL org.opencontainers.image.title="lfg" \
      org.opencontainers.image.description="LFG runtime + UI + ML tooling" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.created="${BUILD_CREATED}"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 lfg \
    && useradd --system --uid 10001 --gid 10001 --home-dir /opt/lfg --create-home --shell /usr/sbin/nologin lfg

WORKDIR /opt/lfg

COPY --from=builder /app/target/release/lfg-server /usr/local/bin/lfg-server
COPY --from=ui-builder /ui/master-clyffy-web/dist ./ui/master-clyffy-web/dist
COPY ml/lfg-encoder ./ml/lfg-encoder

RUN python3 -m venv /opt/lfg/.venv \
    && /opt/lfg/.venv/bin/python -m pip install --no-cache-dir -r /opt/lfg/ml/lfg-encoder/requirements.txt \
    && chown -R lfg:lfg /opt/lfg

ENV PATH="/opt/lfg/.venv/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV MASTER_CLYFFY_UI_DIR="/opt/lfg/ui/master-clyffy-web/dist"

EXPOSE 4317

USER lfg:lfg

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import json,sys,urllib.request; d=json.loads(urllib.request.urlopen('http://127.0.0.1:4317/health', timeout=3).read().decode('utf-8')); sys.exit(0 if d.get('status') == 'ok' else 1)"

ENTRYPOINT ["lfg-server"]
CMD ["serve", "--bind", "0.0.0.0:4317"]
