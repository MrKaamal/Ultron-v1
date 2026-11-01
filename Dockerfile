# ULTRON in a container: the safest place to run it unattended.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && rm -rf /build

# Non-root user; everything the agent may touch lives in /data
RUN useradd --create-home --uid 10001 ultron && mkdir -p /data && chown ultron /data
USER ultron
WORKDIR /data
ENV ULTRON_WORKSPACE=/data/workspace ULTRON_STATE_DIR=/data/state

ENTRYPOINT ["ultron"]
CMD ["--help"]
