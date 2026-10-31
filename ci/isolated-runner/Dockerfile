ARG RUNNER_BASE=ghcr.io/actions/actions-runner@sha256:759a1d94a15b05941aba5196974ff43fe15b4cbba29b694b35cbc32a1166b232
FROM ${RUNNER_BASE}

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential pkg-config libssl-dev gh \
    && gpasswd -d runner sudo \
    && gpasswd -d runner docker \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/runner/_work/_tool \
    && chown -R runner:runner /home/runner/_work
COPY entrypoint.sh /usr/local/bin/lfg-runner-entrypoint
USER runner
WORKDIR /home/runner
ENV RUNNER_TOOL_CACHE=/home/runner/_work/_tool
ENTRYPOINT ["/bin/sh", "/usr/local/bin/lfg-runner-entrypoint"]
