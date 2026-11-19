# For participants of the RLinf workshop: RL fine-tuning of the Lingbot-VLA robot policy with GRPO on RoboTwin 2.0, on 4 W7900D.
# The environment of the RLinf ROCm RoboTwin image (RLinf's docker/Dockerfile, BUILD_TARGET=embodied-robotwin,
# PLATFORM=amd, ROCM_ARCHS=gfx1100), reduced to what the workshop uses and rebuilt from pinned sources: the same base
# image and system packages, the Python packages at the versions of the validated image, RLinf, RoboTwin, Lingbot-VLA
# and LeRobot at pinned commits. The models and RoboTwin's assets are not in the image: they are read from the
# read-only volume mounted at /models/rlinf-workshop (see README.md for its layout and how to prepare it).
FROM docker.io/rocm/dev-ubuntu-22.04:7.2-complete@sha256:0333e72e373e5fcc5f2845c38677cb2d53984dcf80a32c8a5c6cba99d02fdc7e

# Each build argument is declared right before the step that uses it, so that changing one does not invalidate the
# layers before it.

# The GPU architecture settings of the RLinf image, and what RLinf's installer sets for RoboTwin on AMD: the RADV
# Vulkan driver for SAPIEN's rendering and flash-attention's Triton AMD backend. As ENV they reach every process,
# including Jupyter kernels.
ENV AMDGPU_TARGETS=gfx1100 \
    GPU_ARCHS=gfx1100 \
    HCC_AMDGPU_TARGET=gfx1100 \
    PYTORCH_ROCM_ARCH=gfx1100 \
    AMD_VISIBLE_DEVICES=all \
    RLINF_PLATFORM=amd \
    HF_HOME=/opt/.cache/huggingface \
    AMD_VULKAN_ICD=RADV \
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# 1. The system packages the RLinf image installs (build tools, OpenGL/EGL/Vulkan/Mesa for SAPIEN, video codecs, ...).
#    They come from the Ubuntu archive, so only the Ubuntu sources are used for this step (the base image's
#    repo.radeon.com sources are skipped and left in place).
ARG APT_MIRROR=mirrors.aliyun.com
COPY apt-packages.txt /tmp/rc/
RUN sed -i -e "s@//.*archive.ubuntu.com@//${APT_MIRROR}@g" -e "s@//security.ubuntu.com@//${APT_MIRROR}@g" /etc/apt/sources.list && \
    apt-get -o Dir::Etc::SourceParts=/dev/null update && \
    grep -v '^#' /tmp/rc/apt-packages.txt | DEBIAN_FRONTEND=noninteractive xargs apt-get -o Dir::Etc::SourceParts=/dev/null \
        install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt torch-wheels.txt fetch.py /tmp/rc/

# 2. uv and the Python 3.11.14 build that RLinf's installer uses (python-build-standalone 20260211, as uv 0.13.7
#    installs it), then the Lingbot-VLA environment with every package at the validated version (--no-deps).
#    The interpreter is downloaded and checked here (with retries and a GitHub fallback), then installed by uv from
#    that local copy, so it is byte-for-byte what uv installs.
# The build sets PIP_INDEX_URL to a PyPI mirror reachable from the build cluster; it is not kept in the image.
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
#    (uv settings for the build only; ARGs are not kept in the image)
ARG UV_PYTHON_INSTALL_DIR=/opt/venv/.python
ARG UV_LINK_MODE=copy
ARG UV_NO_CACHE=1
ARG PBS=cpython-3.11.14+20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
ARG PBS_URL=cpython-3.11.14%2B20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
RUN python3 -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" uv==0.13.7 && \
    mkdir -p /tmp/rc/pbs/20260211 && \
    python3 /tmp/rc/fetch.py file 1fb6a89daa750f7eff00eb65d901c3c236713812eb9c8c7e07ecdfcd78203c8e "/tmp/rc/pbs/20260211/${PBS}" \
        "https://releases.astral.sh/github/python-build-standalone/releases/download/20260211/${PBS_URL}" \
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260211/${PBS_URL}" && \
    UV_PYTHON_INSTALL_MIRROR=file:///tmp/rc/pbs uv python install 3.11.14 --no-bin && rm -rf /tmp/rc/pbs && \
    uv venv /opt/venv/lingbotvla --python 3.11.14 && \
    uv pip install --python /opt/venv/lingbotvla/bin/python --no-deps --index-url "${PIP_INDEX_URL}" \
        -r /tmp/rc/requirements.txt

# 3. The PyTorch ROCm 7.2 wheels (torch-wheels.txt: file names and sha256), from a mirror of download.pytorch.org
#    (default: NJU, which RLinf's installer also uses in China, ~40x faster there for the 6.2 GB torch wheel), then
#    download.pytorch.org itself. They are installed one at a time in the file's order, because three of them provide
#    the `triton` module: the validated image has triton 3.5.1 on top of pytorch-triton-rocm on top of triton-rocm.
#    The eight files removed afterwards belong to triton-rocm 3.6.0 only and are absent from the validated image (RLinf's
#    installer removed them when it replaced an earlier triton).
ARG TORCH_WHEELS_MIRROR=https://mirrors.nju.edu.cn/pytorch/whl
RUN mkdir -p /tmp/rc/wheels && \
    grep -v '^#' /tmp/rc/torch-wheels.txt | while read -r sha path; do \
        f=$(basename "$path"); u=$(printf '%s' "$path" | sed 's/+/%2B/g'); \
        python3 /tmp/rc/fetch.py file "$sha" "/tmp/rc/wheels/$f" "${TORCH_WHEELS_MIRROR}/$u" "https://download.pytorch.org/whl/$u" && \
        uv pip install --python /opt/venv/lingbotvla/bin/python --no-deps "/tmp/rc/wheels/$f" && \
        rm "/tmp/rc/wheels/$f" || exit 1; \
    done && \
    cd /opt/venv/lingbotvla/lib/python3.11/site-packages/triton && \
    rm _C/libtriton/linear_layout.pyi backends/amd/include/hip/linker_types.h \
        backends/amd/include/hipblas-common/hipblas-common.h backends/nvidia/bin/ptxas-blackwell \
        experimental/gluon/amd/__init__.py experimental/gluon/language/amd/_ops.py \
        experimental/gluon/language/amd/rdna3/__init__.py experimental/gluon/language/amd/rdna4/__init__.py && \
    rmdir _C/libtriton backends/amd/include/hipblas-common experimental/gluon/language/amd/rdna3 \
        experimental/gluon/language/amd/rdna4

# 4. The packages that are not on an index: flash-attn for ROCm 7.2 / PyTorch 2.11 (RLinf's prebuilt wheel), LeRobot
#    and Lingbot-VLA (editable, with its lingbot-depth and MoGe submodules) at pinned commits; then RLinf's patches for
#    RoboTwin on AMD to SAPIEN's URDF loader and mplib's planner.
ARG LEROBOT_COMMIT=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
ARG LINGBOT_VLA_COMMIT=a57f084679bccf976d90b2fe8b9a2372c81d86c2
ARG LINGBOT_DEPTH_COMMIT=64f964fbce4746bccd251b7ed5a10efa922b6675
ARG MOGE_COMMIT=07444410f1e33f402353b99d6ccd26bd31e469e8
# GitHub Releases downloads go through this proxy first (the one Radeon Cloud's own first-shell help suggests), then
# directly to github.com; the file is checked by sha256 either way. Set it empty to use github.com only.
ARG GITHUB_PROXY=https://gh-proxy.org/
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+rocm7.2torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
ARG FLASH_ATTN_WHEEL_URL=https://github.com/RLinf/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Brocm7.2torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
RUN V=/opt/venv/lingbotvla && L=$V/lingbot-vla && M=$L/lingbotvla/models/vla/vision_models && \
    python3 /tmp/rc/fetch.py file 5ca271152ccddb15d91abab09945e056fca5a3fa33f78827c95ccaf9b7e9813b /tmp/rc/${FLASH_ATTN_WHEEL} \
        ${GITHUB_PROXY:+"${GITHUB_PROXY}${FLASH_ATTN_WHEEL_URL}"} "${FLASH_ATTN_WHEEL_URL}" && \
    python3 /tmp/rc/fetch.py src huggingface/lerobot ${LEROBOT_COMMIT} /tmp/rc/lerobot && \
    python3 /tmp/rc/fetch.py src RLinf/lingbot-vla ${LINGBOT_VLA_COMMIT} $L && \
    python3 /tmp/rc/fetch.py src Robbyant/lingbot-depth ${LINGBOT_DEPTH_COMMIT} $M/lingbot-depth && \
    python3 /tmp/rc/fetch.py src microsoft/MoGe ${MOGE_COMMIT} $M/MoGe && \
    uv pip install --python $V/bin/python --no-deps --index-url "${PIP_INDEX_URL}" \
        /tmp/rc/${FLASH_ATTN_WHEEL} /tmp/rc/lerobot -e $L -e $M/lingbot-depth -e $M/MoGe && \
    S=$V/lib/python3.11/site-packages && \
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $S/sapien/wrapper/urdf_loader.py && \
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $S/mplib/planner.py && \
    rm -rf /tmp/rc/lerobot /tmp/rc/${FLASH_ATTN_WHEEL}

# 5. RLinf and RoboTwin at pinned commits. RoboTwin's assets (background_texture, objects, embodiments) are links into
#    the mounted volume; RoboTwin only reads them (the workshop uses the mplib planner, so no curobo configs are
#    generated into them).
ARG RLINF_COMMIT=58d1b9abd6c65dd2dce7fdf57afedfdd9273ef9c
ARG ROBOTWIN_COMMIT=dca9ec682688821c42944eacd7f5dd9bfb15396f
RUN python3 /tmp/rc/fetch.py src RLinf/RLinf ${RLINF_COMMIT} /opt/RLinf && \
    python3 /tmp/rc/fetch.py src RoboTwin-Platform/RoboTwin ${ROBOTWIN_COMMIT} /opt/robotwin && \
    for d in background_texture objects embodiments; do ln -s /models/rlinf-workshop/robotwin-assets/$d /opt/robotwin/assets/$d; done && \
    mkdir -p ${HF_HOME} /workspace && rm -rf /tmp/rc

# The Lingbot-VLA environment first on PATH: `python` and `jupyter` (JupyterLab 4.6.6, in requirements.txt) are that
# environment's, which passes Radeon Cloud's launch-time JupyterLab check. JupyterLab terminals start "$SHELL -l".
ENV VIRTUAL_ENV=/opt/venv/lingbotvla \
    PATH=/opt/venv/lingbotvla/bin:${PATH} \
    SHELL=/bin/bash
RUN command -v jupyter && python -c 'import importlib.metadata as m, sys; \
v = lambda n: tuple(map(int, m.version(n).split(".")[:2])); \
assert (3, 10) <= sys.version_info[:2] <= (3, 12) and (2, 14) <= v("jupyter_server") < (3, 0) and (4, 2) <= v("jupyterlab") < (5, 0); \
import torch, triton, sapien, mplib, flash_attn, lerobot, lingbotvla; \
assert torch.__version__ == "2.11.0+rocm7.2" and triton.__version__ == "3.5.1", (torch.__version__, triton.__version__)'

# No CMD/ENTRYPOINT of its own: the platform starts instances with its own command (the base image's CMD is bash).
WORKDIR /workspace
