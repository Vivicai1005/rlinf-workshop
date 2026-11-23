# RLinf Workshop: RL Fine-Tuning of a VLA Robot Policy on AMD Radeon™ GPUs

A hands-on Jupyter notebook that trains the **Lingbot-VLA** policy with **GRPO** on the RoboTwin 2.0 `place_shoe` task using [RLinf](https://github.com/RLinf/RLinf), on 4 AMD Radeon™ PRO W7900D GPUs with ROCm.

| Notebook | What it covers |
| --- | --- |
| [`rlinf_lingbot_place_shoe_grpo_workshop.ipynb`](rlinf_lingbot_place_shoe_grpo_workshop.ipynb) (English) | GRPO for VLA policies, how RLinf runs a training step, checking the GPUs and environment, the workshop configuration, launching a 40-step training run in the background, live progress monitoring, metrics, evaluation results and videos after every evaluation |
| [`rlinf_lingbot_place_shoe_grpo_workshop_zh.ipynb`](rlinf_lingbot_place_shoe_grpo_workshop_zh.ipynb) (中文) | Chinese version of the same notebook, with identical code and settings |

## What participants run

A demo-sized GRPO run: 16 parallel RoboTwin environments on 4 GPUs (GRPO group size 4), 40 training steps, with an evaluation on 32 held-out scenes (with videos) after steps 10, 20, 30 and 40. The live monitor stops at each new evaluation so participants can play its videos while training continues.

Timings on 4 × AMD Radeon PRO W7900D GPUs:

| | Time |
| --- | --- |
| Start-up (Ray, model loading, environments) | ~2.5 min |
| Step 1 | ~130 s |
| Every other step | 55–75 s |
| Evaluation (32 episodes) | ~3 min the first time, then ~1.5 min (longer if other jobs load the CPUs) |
| **Launch → first evaluation result (after step 10)** | **~18 min** |
| Each further 10 steps + evaluation | ~12 min |
| Whole 40-step run | ~55 min |

Measured in four full 40-step runs.

## Requirements

- 4 AMD Radeon PRO W7900D GPUs and ~200 GB of host RAM per participant, with at least 8 GB of `/dev/shm`.
- The workshop image (below) and the workshop's **data volume** (below), mounted read-only at `/models/rlinf-workshop`.

## Data volume

The models and RoboTwin's simulation assets are not in the image. They live on one read-only volume (~33 GB), mounted at
**`/models/rlinf-workshop`**:

| Path | Source (Hugging Face, pinned revision) | Files | Size |
| --- | --- | --- | --- |
| `lingbot-vla-4b-posttrain-robotwin/` | [`robbyant/lingbot-vla-4b-posttrain-robotwin`](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin) @ `3e0c7c476bde3daaac00f79f3741a292a299f60a` | `model.safetensors`, `config.json`, `lingbotvla_cli.yaml` | 16.8 GB |
| `Qwen2.5-VL-3B-Instruct/` | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) @ `66285546d2b821cf421d4f5eb2576359d3770cd3` | `config.json`, `generation_config.json`, `chat_template.json`, `preprocessor_config.json`, `tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt` | 11 MB |
| `lingbot-vla-4b/` | [`robbyant/lingbot-vla-4b`](https://huggingface.co/robbyant/lingbot-vla-4b) @ `87712223fb463f0f6a6565f8feba64c35418e393` | `config.json` | 294 B |
| `robotwin-assets/` | dataset [`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) @ `981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042` | `background_texture.zip`, `objects.zip`, `embodiments.zip`, unpacked (without `__MACOSX`) | 16 GB |

All weights come from the post-trained checkpoint: RLinf builds the Qwen2.5-VL backbone from its configuration and then
loads the whole policy from `lingbot-vla-4b-posttrain-robotwin`; from the other two repositories it reads only the
configuration, tokenizer and processor. RoboTwin only reads its assets (the workshop uses the mplib planner).

[`mount/prepare_mount.sh`](mount/prepare_mount.sh) builds the volume on any machine with internet access; every file is
downloaded at its pinned revision and checked by sha256 (large files over parallel connections; `HF_ENDPOINT` selects
the Hugging Face endpoint, default hf-mirror.com):

```bash
bash mount/prepare_mount.sh /path/to/rlinf-workshop-data      # then mount it read-only at /models/rlinf-workshop
```

There is no training dataset: GRPO generates its data online in the simulator.

## Workshop image

[`docker/`](docker/) holds the image's build context (the same files as `images/rlinf-workshop/` of AMD-RC-TEAM/radeon-cloud-images).
The Dockerfile rebuilds the environment of the RLinf ROCm RoboTwin image (`BUILD_TARGET=embodied-robotwin`,
`PLATFORM=amd`, `ROCM_ARCHS=gfx1100`; see the [RLinf RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html))
that the workshop uses, from pinned sources, on the official `rocm/dev-ubuntu-22.04:7.2-complete` base (~41 GB):

| Path in the image | Content |
| --- | --- |
| `/opt/RLinf` | RLinf source, pinned to the validated commit (`58d1b9ab`) |
| `/opt/robotwin` | RoboTwin, `RLinf_support` branch (`dca9ec6`); `assets/{background_texture,objects,embodiments}` link into the data volume |
| `/opt/venv/lingbotvla` | Lingbot-VLA environment (Python 3.11, PyTorch 2.11 for ROCm 7.2, Triton 3.5.1, flash-attn, JupyterLab 4.6); first on `PATH`, so `python` and `jupyter` are this environment's |

Every package and source is pinned (`apt-packages.txt`, `requirements.txt`, `torch-wheels.txt` with sha256, commits),
and every downloaded file is checked by sha256. Nothing is installed under `/workspace`, because on Radeon Cloud
`/workspace` is the instance's workspace storage; the platform pulls the notebooks from this repository into it, and
the training runs write their logs and videos to `/workspace/rlinf_workshop`. The image has no `CMD` or `ENTRYPOINT`
of its own (the base image's `CMD` is `bash`).

```bash
cd docker && docker build -t rlinf-ws:latest .
# options: --build-arg PIP_INDEX_URL=<PyPI mirror>  --build-arg APT_MIRROR=<Ubuntu mirror host>
#          --build-arg TORCH_WHEELS_MIRROR=<mirror of download.pytorch.org/whl>  --build-arg GITHUB_PROXY=<GitHub proxy>
```

[`docker/smoke.sh`](docker/smoke.sh) is the image's smoke test (one GPU): JupyterLab, PyTorch and Triton on the GPU,
the asset links, and, with the data volume mounted, two RoboTwin `place_shoe` environments resetting and stepping.

### Radeon Cloud

The workshop runs as a Gallery notebook that launches the image on a 4-GPU size (~200 GB of memory, ≥8 GB of
`/dev/shm`) with the data volume mounted at `/models/rlinf-workshop`. The image passes Radeon Cloud's launch-time
JupyterLab check (`jupyter` on `PATH`, Python 3.10–3.12, `jupyter_server` >=2.14,<3, `jupyterlab` >=4.2,<5), so
instances start without installing JupyterLab, and the notebook kernel is the Lingbot-VLA environment.

### Your own machines

Start one container per participant, with the data volume at `/models/rlinf-workshop` and a clone of this repository
at `/workspace`; JupyterLab serves the notebooks on port 8888:

```bash
docker run -d --name rlinf-ws \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --init --security-opt seccomp=unconfined --ipc=host --shm-size 64g -p 8888:8888 \
  -v /path/to/rlinf-workshop-data:/models/rlinf-workshop:ro -v /path/to/rlinf-workshop:/workspace \
  rlinf-ws:latest \
  jupyter lab --notebook-dir=/workspace --ip=0.0.0.0 --port=8888 --allow-root --no-browser --ServerApp.token=<token>
```

`--init` lets finished background processes be reaped (the notebook launches training with `nohup setsid`). One run peaks at ~170 GB of host RAM, so the machine needs at least ~200 GB. If several participants share one machine, start each container on the default bridge network (not `--network host`) so that their Ray clusters do not collide on ports, and publish a different JupyterLab port for each.
