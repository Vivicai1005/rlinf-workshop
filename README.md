# RLinf Workshop: RL Fine-Tuning of a VLA Robot Policy on AMD Radeon™ GPUs

A hands-on Jupyter notebook that trains the **Lingbot-VLA** policy with **GRPO** on the RoboTwin 2.0 `place_shoe` task using [RLinf](https://github.com/RLinf/RLinf), on 4 AMD Radeon™ PRO W7900D GPUs with ROCm.

| Notebook | What it covers |
| --- | --- |
| [`rlinf_lingbot_place_shoe_grpo_workshop.ipynb`](rlinf_lingbot_place_shoe_grpo_workshop.ipynb) | GRPO for VLA policies, how RLinf runs a training step, checking the GPUs and environment, the workshop configuration, launching a 40-step training run in the background, live progress monitoring, metrics, evaluation results and videos after every evaluation |

## What participants run

A demo-sized GRPO run: 16 parallel RoboTwin environments on 4 GPUs (GRPO group size 4), 40 training steps, with an evaluation on 32 held-out scenes (with videos) after steps 10, 20, 30 and 40. The live monitor stops at each new evaluation so participants can plot the curves and play the videos while training continues.

Timings on 4 × AMD Radeon PRO W7900D GPUs:

| | Time |
| --- | --- |
| Start-up (Ray, model loading, environments) | ~2.5 min |
| Step 1 | ~130 s |
| Every other step | 55–75 s |
| Evaluation (32 episodes) | ~90 s (up to ~3 min if other jobs load the CPUs) |
| **Launch → first evaluation result (after step 10)** | **16–18 min** (measured) |
| Each further 10 steps + evaluation | ~13 min (estimated) |
| Whole 40-step run | ~55 min (estimated) |

## Requirements

- 4 AMD Radeon PRO W7900D GPUs and ~200 GB of host RAM per participant.
- The workshop image (below), and the model weights on the host (~39 GB): `lingbot-vla-4b-posttrain-robotwin`, `Qwen2.5-VL-3B-Instruct` and `lingbot-vla-4b`.

## Workshop image

[`docker/Dockerfile`](docker/Dockerfile) builds the image the participants use. It starts from the RLinf ROCm RoboTwin image (`BUILD_TARGET=embodied-robotwin`, `PLATFORM=amd`, `ROCM_ARCHS=gfx1100`; see the [RLinf RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)) and adds everything the notebook needs except the model weights:

| Path in the image | Content |
| --- | --- |
| `/workspace/RLinf` | RLinf source, pinned to the validated commit (`58d1b9ab`) |
| `/opt/robotwin` | RoboTwin, `RLinf_support` branch (`dca9ec6`), with its assets in `/opt/robotwin/assets` (~16 GB) |
| `/opt/venv/lingbotvla` | Lingbot-VLA environment from the RLinf image, plus JupyterLab |
| `/workspace/rlinf_lingbot_place_shoe_grpo_workshop.ipynb` | The notebook |

Build it with the notebook and the RoboTwin assets (downloaded with RoboTwin's `script/_download_assets.sh`) in the build context:

```bash
cd docker
cp ../rlinf_lingbot_place_shoe_grpo_workshop.ipynb .
cp -r /path/to/robotwin-assets ./robotwin-assets     # background_texture, embodiments, objects, files
docker build -t rlinf-ws:latest .
# options: --build-arg BASE_IMAGE=<RLinf ROCm RoboTwin image>  --build-arg GITHUB_PREFIX=<GitHub proxy>
#          --build-arg PYPI_INDEX=<PyPI mirror for JupyterLab>
```

Start one container per participant, mounting only the models; JupyterLab serves the notebook on port 8888:

```bash
docker run -d --name rlinf-ws \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --init --security-opt seccomp=unconfined --ipc=host --shm-size 64g -p 8888:8888 \
  -v /path/to/models:/data/models:ro \
  rlinf-ws:latest \
  /opt/venv/lingbotvla/bin/jupyter lab --notebook-dir=/workspace --ip=0.0.0.0 --port=8888 \
      --allow-root --no-browser --ServerApp.token=<token>
```

The notebook's appendix explains the options.
