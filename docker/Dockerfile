# Workshop image: the RLinf ROCm RoboTwin image plus everything the notebook needs except the model weights.
#
# Build context (this directory):
#   Dockerfile
#   rlinf_lingbot_place_shoe_grpo_workshop.ipynb   (copy from the repository root)
#   robotwin-assets/                              (RoboTwin assets, ~16 GB: background_texture, embodiments, objects, files)
#
#   docker build -t rlinf-ws:latest .
#
# Only the models are mounted at run time (see the README).

# RLinf ROCm RoboTwin image: docker/Dockerfile of RLinf with BUILD_TARGET=embodied-robotwin, PLATFORM=amd,
# ROCM_ARCHS=gfx1100 (ROCm 7.2, PyTorch 2.11, SAPIEN 3.0.1, /opt/venv/lingbotvla).
ARG BASE_IMAGE=rlinf-rocm:c8-robotwin
FROM ${BASE_IMAGE}

# Optional GitHub proxy prefix, e.g. https://cdn.gh-proxy.org/ (with the trailing slash)
ARG GITHUB_PREFIX=

# RLinf source, pinned to the commit the workshop was validated with
ARG RLINF_REF=58d1b9ab
RUN git clone ${GITHUB_PREFIX}https://github.com/RLinf/RLinf.git /workspace/RLinf && \
    git -C /workspace/RLinf checkout --detach ${RLINF_REF}

# RoboTwin, RLinf_support branch, pinned
ARG ROBOTWIN_REF=dca9ec6
RUN git clone -b RLinf_support ${GITHUB_PREFIX}https://github.com/RoboTwin-Platform/RoboTwin.git /opt/robotwin && \
    git -C /opt/robotwin checkout --detach ${ROBOTWIN_REF}

# RoboTwin assets; regenerate the embodiment configs for the /opt/robotwin location
COPY robotwin-assets/ /opt/robotwin/assets/
RUN cd /opt/robotwin && python3 script/update_embodiment_config_path.py

# JupyterLab in the Lingbot-VLA environment, and the workshop notebook.
# PYPI_INDEX overrides the package index (e.g. https://mirrors.aliyun.com/pypi/simple); empty = the image default.
ARG PYPI_INDEX=
RUN uv pip install --python /opt/venv/lingbotvla/bin/python ${PYPI_INDEX:+--default-index ${PYPI_INDEX}} \
        jupyterlab ipykernel
COPY rlinf_lingbot_place_shoe_grpo_workshop.ipynb /workspace/

WORKDIR /workspace
EXPOSE 8888
CMD ["/opt/venv/lingbotvla/bin/jupyter", "lab", "--notebook-dir=/workspace", "--ip=0.0.0.0", "--port=8888", \
     "--allow-root", "--no-browser"]
