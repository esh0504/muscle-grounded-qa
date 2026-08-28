# MeshLLM — baseline training stack (release image)
#
# Stage-1 SpiralNet++ muscle regression + Stage-2 mesh-conditioned LLM
# (frozen mesh encoder -> bridge -> Qwen3-8B LoRA), evaluation suite included.
#
#   docker compose build
#   docker compose run --rm train ours_en          # = bash scripts/train_s2.sh ours_en
#
# The image bakes the code at /workspace; DATA/, outputs/ and .cache/ are volumes
# (see docker-compose.yml). Qwen3-8B is downloaded on first run into .cache/hf.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git ffmpeg build-essential cmake \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# openmesh (SpiralNet++ spiral extraction) ships no cp311 wheel and its sdist
# only builds with setuptools<70 — build it in one pinned layer.
RUN pip install --no-cache-dir "setuptools<70" wheel \
    && pip install --no-cache-dir openmesh==1.2.1

# Interactive shells also create world-writable files (dirs 777 / files 666)
RUN echo "umask 000" >> /etc/bash.bashrc

COPY . /workspace
WORKDIR /workspace

CMD ["bash"]
