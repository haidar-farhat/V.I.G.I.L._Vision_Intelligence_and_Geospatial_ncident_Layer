# Sentinel Vision — container image.
#
# Read this first, because a container is a strange shape for this product and
# the strangeness is deliberate.
#
# **The build stage uses the network. Nothing after it may.** Cargo resolves
# crates and pip resolves wheels here, at build time, exactly as they do on a
# developer's machine. The running image has no reason to reach anything and the
# compose file gives it `network_mode: none` to prove it — that is not a
# hardening option, it is the acceptance test. If the container needs the
# Internet to work, this product does not work.
#
# **What it is for.** Three things, in order of how much they are worth:
#
#   1. `sentinel run` — headless analysis over files or LAN cameras. This is the
#      real workload and the reason the image exists.
#   2. `python tasks.py check` with the network taken away — the offline
#      acceptance proof, reproducible on any machine with Docker rather than
#      only inside GitHub Actions.
#   3. A reproducible build environment, so "works on my machine" has an answer.
#
# **What it is not.** There is no server in here. This build has no control
# plane, no REST API and no daemon; see ROADMAP.md items 1.2 and 2.1. The
# container runs an analysis and exits. Nothing listens on a port, and if you
# find something that does, that is a bug.
#
# The operator console is a Qt desktop application and is not the point of this
# image, but it does run — see `docker-compose.yml`, which mounts an X11 socket
# for it. On Windows and macOS that needs a display server the container can
# reach, and running the packaged executable natively is easier and better.

# ----------------------------------------------------------------- build stage

FROM rust:1-slim-bookworm AS core

WORKDIR /build
COPY core/ ./core/

# Release, because the debug build is roughly forty times slower on the hot path
# and the whole reason this code is in Rust is the hot path.
RUN cargo build --release --manifest-path core/Cargo.toml


# --------------------------------------------------------------- runtime stage

FROM python:3.12-slim-bookworm AS runtime

# OpenCV needs these even in its headless build: libGL for the video path and
# libglib for the codec plumbing. Everything else Qt would want is deliberately
# absent — this image analyses, it does not display.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The engine's dependency set is installed before the source is copied, so
# editing a Python file does not re-resolve numpy.
COPY engine/pyproject.toml ./engine/pyproject.toml
COPY engine/sentinel/__init__.py ./engine/sentinel/__init__.py
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -e ./engine

COPY engine/ ./engine/
COPY apps/ ./apps/
COPY tools/ ./tools/
COPY docs/ ./docs/
# Tiny, and both scanned by the offline audit and asserted against by the tests.
# Leaving it out made the audit inside the container look at less than the whole
# product while still reporting "no route out".
COPY packaging/ ./packaging/
COPY tasks.py *.md ./
# The Rust source, even though it is already compiled into the .so above. The
# offline audit scans it, and an audit that runs over an incomplete tree returns
# "no route out" while having looked at less than the whole product.
COPY core/Cargo.toml core/Cargo.lock ./core/
COPY core/src/ ./core/src/

# Beside the `sentinel` package, because `core.py` looks in its own directory
# first — the same way the packaged executable finds it, so there is no
# container-specific path in the loader.
COPY --from=core /build/core/target/release/libsentinel_core.so \
                 ./engine/sentinel/libsentinel_core.so
# And where `cargo build` would have put it, so `tasks.py test` finds it too.
COPY --from=core /build/core/target/release/libsentinel_core.so \
                 ./core/target/release/libsentinel_core.so

# Data goes in a volume, not a layer. A container's filesystem is discarded when
# it stops, and evidence that disappears when a container restarts is not
# evidence.
ENV SENTINEL_DATA_DIR=/data
# The log is stdout. A second copy inside a layer that is thrown away is worse
# than useless, and `docker logs` is where anybody will actually look.
ENV SENTINEL_LOG_FILE=""
ENV PYTHONPATH=/app/engine:/app/apps/console
ENV PYTHONUNBUFFERED=1

# Not root. A process that parses hostile video should not be able to write to
# anything it does not own — "hostile camera" is in the threat model.
RUN useradd --create-home --uid 10001 sentinel \
 && mkdir -p /data /media /evidence \
 && chown -R sentinel:sentinel /data /evidence
USER sentinel

VOLUME ["/data"]

# No HEALTHCHECK. Nothing listens, so there is nothing to probe, and a health
# check that always passes is worse than none.

ENTRYPOINT ["python", "-m", "sentinel"]
CMD ["--help"]


# ------------------------------------------------------------------ test stage
#
# The offline acceptance proof, reproducible anywhere Docker runs rather than
# only inside GitHub Actions. Separate from the runtime image on purpose: the
# test fixtures need `onnx` to build a model locally — nothing may be
# downloaded — and a shipping image has no business carrying a graph compiler.
#
#   docker compose run --rm verify
#
# The Rust suite is not here. It needs a toolchain, and putting one in this
# image to run 57 tests would double its size; `python tasks.py check` runs
# them on a machine that has cargo.

FROM runtime AS test

USER root
RUN python -m pip install --no-cache-dir -e "./engine[dev]"
USER sentinel

# The image is read-only to the user that runs it, deliberately, and pytest's
# cache would be the one thing trying to write into it.
ENV PYTEST_ADDOPTS="-p no:cacheprovider"

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["python tools/offline_audit.py && python tools/docs_lint.py && cd engine && python -m pytest -q"]
