# 3DTongueQA construction pipeline — release image
#
# Self-contained: ArtiSynth (core + models, compiled) + Python + the pipeline
# code baked in at /workspace. Only the /data volume is external (pool,
# simulated shards, indexes, QA output).
#
# Build from the pipeline root (compose does this for you):
#   docker compose build
#
# Reproducibility: pin the ArtiSynth sources for a release —
#   docker compose build \
#     --build-arg ARTISYNTH_REF=<commit-or-tag> --build-arg MODELS_REF=<commit-or-tag>
# (default 'master' = latest; the released dataset should record the exact refs)

FROM eclipse-temurin:21-jdk-jammy

ARG ARTISYNTH_GIT=https://github.com/artisynth/artisynth_core.git
ARG ARTISYNTH_MODELS_GIT=https://github.com/artisynth/artisynth_models.git
ARG ARTISYNTH_REF=master
ARG MODELS_REF=master
ARG ARTISYNTH_HOME=/opt/artisynth/artisynth_core
ARG ARTISYNTH_MODELS=/opt/artisynth/artisynth_models

# Python for steps 1/2/5 + native/GL libs ArtiSynth's solvers and mesh I/O expect
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python-is-python3 git make ca-certificates \
        libgomp1 libxrender1 libgl1 libglib2.0-0 libsm6 libxext6 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Pipeline's actual Python dependencies only (no dev/training extras)
RUN pip3 install --no-cache-dir numpy pandas pyyaml

# ArtiSynth core + models at pinned refs, then native libs + compile
RUN git clone "${ARTISYNTH_GIT}" "${ARTISYNTH_HOME}" && \
    git -C "${ARTISYNTH_HOME}" checkout --detach "${ARTISYNTH_REF}" && \
    git clone "${ARTISYNTH_MODELS_GIT}" "${ARTISYNTH_MODELS}" && \
    git -C "${ARTISYNTH_MODELS}" checkout --detach "${MODELS_REF}" && \
    ( git -C "${ARTISYNTH_HOME}" rev-parse HEAD; \
      git -C "${ARTISYNTH_MODELS}" rev-parse HEAD ) > /opt/artisynth/REFS && \
    cp -r "${ARTISYNTH_MODELS}/src/." "${ARTISYNTH_HOME}/src/"

RUN cd "${ARTISYNTH_HOME}" && \
    java -cp "lib/vfs2.jar:bin/libraryInstaller.jar" \
        artisynth.core.driver.LibraryInstaller -updateLibs \
        -remoteSource https://www.artisynth.org/files/lib/ && \
    make

# Bake the pipeline in (build context = pipeline root; see docker-compose.yml)
COPY . /workspace
RUN sed -i 's/\r$//' /workspace/docker-entrypoint.sh /workspace/scripts/*.sh && \
    chmod +x /workspace/docker-entrypoint.sh /workspace/scripts/*.sh

ENV ARTISYNTH_HOME=${ARTISYNTH_HOME} \
    TONGUE_MODEL=artisynth.models.tongue3d.StableFemMuscleTongueDemo \
    PIPELINE_CONFIG=/workspace/settings/config.docker.json \
    PYTHONUNBUFFERED=1 \
    JVM_XMX=6g

WORKDIR /workspace
ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
CMD ["bash"]
