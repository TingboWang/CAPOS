# Uses the built-in BuildKit frontend (no docker/dockerfile pull needed).
# =============================================================================
# CAPOS -- two images from one Dockerfile.
#
#   light   CPU-only inference from precomputed features. ~2 GB. This is the
#           out-of-the-box example the response points at: mount the .h5 files
#           and the checkpoint, get predictions in minutes on a laptop.
#
#   full    GPU end-to-end: FASTA -> Chai-1 -> ESM-S/SaProt -> CAPOS. Large,
#           needs an NVIDIA runtime. Each heavy dependency keeps its upstream
#           licence -- check the upstream license files before redistributing.
#
# Build from the repository root:
#   docker build -f Dockerfile -t capos:light --target light .
#   docker build -f Dockerfile -t capos:full  --target full  .
#
# Run:
#   docker run --rm -v $PWD:/data capos:light \
#       predict --seq-feat /data/esms.h5 --struct-feat /data/saprot.h5 \
#               --weights /data/capos_model.pth --out /data/preds.csv
#
#   docker run --rm --gpus all -v $PWD:/data capos:full \
#       run /data/queries.fasta --work-dir /data/out
#
# The build context is the repository root; .dockerignore keeps data, images,
# and supplements out of the image. The full tier also ships the helper scripts
# under tools/.
# =============================================================================


# ---------------------------------------------------------------- light -----
FROM python:3.10-slim-bookworm AS light

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /opt/capos

# CPU wheel of torch; the light tier never touches a GPU.
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" "pandas>=1.5" "h5py>=3.8" \
        --index-url https://pypi.org/simple \
 && pip install --no-cache-dir "typing_extensions>=4.10" --index-url https://pypi.org/simple \
 && pip install --no-cache-dir "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY capos/ ./capos/
COPY weights/ ./weights/
RUN pip install --no-cache-dir --no-deps .

# The CAPOS checkpoints are baked in; mount a different file and pass
# --weights (or set CAPOS_WEIGHTS) to override.
RUN mkdir -p /opt/capos/weights /data
ENV CAPOS_WEIGHTS=/opt/capos/weights/capos_model.pth

WORKDIR /data
ENTRYPOINT ["capos"]
CMD ["--help"]


# ----------------------------------------------------------------- full -----
# CUDA 12.1 comes from the pinned pip wheels (torch 2.3.1+cu121 plus the
# nvidia-*-cu12 packages). The base is plain Ubuntu 22.04 so the image does
# not carry a second copy of the CUDA runtime; the NVIDIA container runtime
# injects the driver for `docker run --gpus all`.
#
# chai-lab 0.6.0 pins torch~=2.3.1, so the full image cannot use torch 2.1.2:
# pip would upgrade torch and leave torch-scatter/sparse/cluster built for the
# wrong ABI. The pins below are the combination exercised on the cluster.
#
# ENVIRONMENT SPLIT. The image mirrors the two environments used for the
# paper instead of flattening them into one Python:
#   /opt/venv/capos   torch + torchdrug + ESM-S + SaProt -> featurize/predict
#   /opt/venv/chai1   chai-lab 0.6.0 + transformers 4.44.2 -> fold
# chai-lab 0.6.0 requires transformers~=4.44, while the CAPOS stack uses
# 4.48.1. Separate venvs keep both resolvers clean; `pip check` is run on
# both below. CAPOS_CHAI1_CMD points at the chai1 venv.
FROM ubuntu:22.04 AS full

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    CAPOS_HOME=/opt/capos

# Override these with --build-arg to use regional mirrors; the defaults are
# the official sources used for the published image.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_FIND_LINKS=
ARG PYG_FIND_LINKS=https://data.pyg.org/whl/torch-2.3.1+cu121.html
ARG APT_MIRROR=
ARG RUSTUP_DIST_SERVER=https://static.rust-lang.org
ARG RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    RUSTUP_DIST_SERVER=${RUSTUP_DIST_SERVER} \
    RUSTUP_UPDATE_ROOT=${RUSTUP_UPDATE_ROOT}

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu#${APT_MIRROR}#g" /etc/apt/sources.list; \
    fi \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git curl ca-certificates \
        build-essential cmake pkg-config zlib1g-dev libbz2-dev \
        libatomic1 libgomp1 \
 && ln -sf /usr/bin/python3.10 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# Foldseek's build uses Rust (via corrosion), even with ProstT5 disabled.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal \
 && /root/.cargo/bin/rustc --version
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/capos

COPY pyproject.toml README.md LICENSE ./
COPY capos/ ./capos/
COPY weights/ ./weights/
# The CLI shells out to these helpers; CAPOS_REPO_ROOT points at them below.
COPY tools/ ./tools/

# Two isolated environments, matching environment.yml / environment_chai1.yml.
RUN python3.10 -m venv /opt/venv/capos /opt/venv/chai1

# ---- capos venv ------------------------------------------------------------
RUN /opt/venv/capos/bin/pip install --no-cache-dir --upgrade \
        pip wheel "setuptools<70" \
 && /opt/venv/capos/bin/pip install --no-cache-dir \
        numpy==1.26.4 pandas==2.3.3 h5py==3.11.0 scipy==1.15.2 \
        scikit-learn==1.7.2 biopython==1.86 matplotlib==3.10.8 \
        seaborn==0.13.2 tqdm==4.67.3 pyyaml==6.0.3 \
        sentencepiece==0.2.1 networkx==3.4.2

RUN if [ -n "$TORCH_FIND_LINKS" ]; then \
        /opt/venv/capos/bin/pip install --no-cache-dir \
            torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
            --find-links "$TORCH_FIND_LINKS" \
            --extra-index-url "$PIP_INDEX_URL" \
            --extra-index-url "$TORCH_INDEX_URL"; \
    else \
        /opt/venv/capos/bin/pip install --no-cache-dir \
            torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
            --index-url "$TORCH_INDEX_URL" \
            --extra-index-url "$PIP_INDEX_URL"; \
    fi

# NVIDIA CUDA libraries for torch 2.3.1+cu121
# MUST use cu12 series (not cu129) to match libcudnn.so.8 requirement
RUN /opt/venv/capos/bin/pip install --no-cache-dir \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cudnn-cu12==8.9.2.26 \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nccl-cu12==2.20.5 \
        nvidia-nvtx-cu12==12.1.105

# torchdrug pulls torch-scatter and friends, which must match the torch build.
RUN /opt/venv/capos/bin/pip install --no-cache-dir \
        torch-scatter==2.1.2+pt23cu121 \
        torch-sparse==0.6.18+pt23cu121 \
        torch-cluster==1.6.3+pt23cu121 \
        torch-spline-conv==1.2.2+pt23cu121 \
        -f ${PYG_FIND_LINKS}

RUN /opt/venv/capos/bin/pip install --no-cache-dir \
        torchdrug==0.2.1 fair-esm==2.0.0 \
        decorator lmdb ninja rdkit-pypi \
        pytorch-lightning==2.2.5

RUN /opt/venv/capos/bin/pip install --no-cache-dir \
        transformers==4.48.1 tokenizers==0.21.4 safetensors==0.7.0 \
        accelerate==1.13.0 huggingface-hub==0.36.2

RUN /opt/venv/capos/bin/pip install --no-cache-dir --no-deps .

# ---- chai1 venv ------------------------------------------------------------
RUN /opt/venv/chai1/bin/pip install --no-cache-dir --upgrade pip wheel \
 && /opt/venv/chai1/bin/pip install --no-cache-dir \
        numpy==1.26.4 biopython==1.85

RUN if [ -n "$TORCH_FIND_LINKS" ]; then \
        /opt/venv/chai1/bin/pip install --no-cache-dir \
            torch==2.3.1 \
            --find-links "$TORCH_FIND_LINKS" \
            --extra-index-url "$PIP_INDEX_URL" \
            --extra-index-url "$TORCH_INDEX_URL"; \
    else \
        /opt/venv/chai1/bin/pip install --no-cache-dir \
            torch==2.3.1 \
            --index-url "$TORCH_INDEX_URL" \
            --extra-index-url "$PIP_INDEX_URL"; \
    fi

RUN /opt/venv/chai1/bin/pip install --no-cache-dir \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cudnn-cu12==8.9.2.26 \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nccl-cu12==2.20.5 \
        nvidia-nvtx-cu12==12.1.105

RUN /opt/venv/chai1/bin/pip install --no-cache-dir \
        chai-lab==0.6.0 transformers==4.44.2 tokenizers==0.19.1 \
        "safetensors>=0.6" "huggingface-hub>=0.35"

# rdkit (imported by torchdrug) links against X render libraries even when
# used headlessly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libsm6 libice6 libx11-6 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Fail the build if either environment has unresolved requirements.
RUN /opt/venv/capos/bin/pip check \
 && /opt/venv/chai1/bin/pip check \
 && /opt/venv/capos/bin/python -c "import torch, torchdrug, transformers; print('capos venv:', torch.__version__, transformers.__version__)" \
 && /opt/venv/chai1/bin/python -c "import torch, transformers; from importlib.metadata import version; print('chai1 venv:', torch.__version__, transformers.__version__, version('chai-lab'))"

# --- third-party tools the full pipeline shells out to ---------------------
# Foldseek: build the exact commit used for the paper. There is no tagged
# static binary for c9fa0e8, so this follows the upstream source build and
# pins the commit explicitly. ProstT5 is disabled because CAPOS only uses
# Foldseek's 3Di extraction and structure search.
ARG FOLDSEEK_COMMIT=c9fa0e8288f46c640cfd7f5102b5d741dfaf45dd
RUN mkdir -p /opt/foldseek-src \
 && curl -fsSL "https://github.com/steineggerlab/foldseek/archive/${FOLDSEEK_COMMIT}.tar.gz" \
        -o /tmp/foldseek.tar.gz \
 && tar -xzf /tmp/foldseek.tar.gz -C /opt/foldseek-src --strip-components=1 \
 && cd /opt/foldseek-src \
 && mkdir build && cd build \
 && cmake -DHAVE_AVX2=1 -DHAVE_MPI=0 -DHAVE_TESTS=0 \
          -DENABLE_PROSTT5=0 -DCMAKE_BUILD_TYPE=Release \
          -DVERSION_OVERRIDE=${FOLDSEEK_COMMIT} .. \
 && make -j"$(nproc)" \
 && install -D -m 0755 src/foldseek /opt/foldseek/bin/foldseek \
 && mkdir -p /opt/foldseek/data \
 && cp -a ../data/. /opt/foldseek/data/ \
 && cd / && rm -rf /opt/foldseek-src /tmp/foldseek.tar.gz
ENV PATH="/opt/foldseek/bin:${PATH}" \
    CAPOS_FOLDSEEK_BIN=/opt/foldseek/bin/foldseek \
    FOLDSEEK_DATA=/opt/foldseek/data

# Chai-1 lives in /opt/venv/chai1 (installed above). It pulls its own weights
# and an ESM-2 checkpoint from its asset host / HuggingFace on first use; bake
# them in if you need offline builds. CAPOS_CHAI1_CMD points at that venv.
ENV CAPOS_CHAI1_CMD=/opt/venv/chai1/bin/chai-lab

# Pretrained encoder and Chai-1 weights are not baked in. Create them with
#   ./download_weights.sh --components esm-s,saprot,prott5,chai1
# and mount pretrained_weights/ at /opt/capos/pretrained_weights.
ENV PATH="/opt/venv/capos/bin:/opt/foldseek/bin:${PATH}" \
    PIP_INDEX_URL=https://pypi.org/simple \
    CAPOS_HOME=/opt/capos \
    CAPOS_REPO_ROOT=/opt/capos \
    CAPOS_WEIGHTS=/opt/capos/weights/capos_model.pth \
    CAPOS_SAPROT_MODEL=/opt/capos/pretrained_weights/saprot/SaProt_650M_PDB \
    CAPOS_SAPROT_REPO=/opt/capos/pretrained_weights/saprot/SaProt \
    CAPOS_ESMS_MODEL=/opt/capos/pretrained_weights/esm-s \
    CAPOS_ESMS_CKPT=/opt/capos/pretrained_weights/esm-s/esm_650m_s.pth \
    CAPOS_PROTT5_MODEL=/opt/capos/pretrained_weights/prott5/prot_t5_xl_half_uniref50-enc \
    CHAI_DOWNLOADS_DIR=/opt/capos/pretrained_weights/chai1
RUN mkdir -p /opt/capos/weights /opt/capos/pretrained_weights /data

WORKDIR /data
ENTRYPOINT ["capos"]
CMD ["--help"]
