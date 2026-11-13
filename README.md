# RLinf Workshop: RL Fine-Tuning of a VLA Robot Policy on AMD Radeon™ GPUs

A hands-on Jupyter notebook that trains the **Lingbot-VLA** policy with **GRPO** on the RoboTwin 2.0 `place_shoe` task using [RLinf](https://github.com/RLinf/RLinf), on 4 AMD Radeon™ PRO W7900D GPUs with ROCm.

| Notebook | What it covers |
| --- | --- |
| [`rlinf_lingbot_place_shoe_grpo_workshop.ipynb`](rlinf_lingbot_place_shoe_grpo_workshop.ipynb) | GRPO for VLA policies, how RLinf runs a training step, checking the GPUs and environment, the workshop configuration, launching a 40-step training run in the background, live progress monitoring, metrics, evaluation results and videos after every evaluation |

## What participants run

A demo-sized GRPO run: 16 parallel RoboTwin environments on 4 GPUs (GRPO group size 4), 40 training steps, with an evaluation on 32 held-out scenes (with videos) after steps 10, 20, 30 and 40. The live monitor stops at each new evaluation so participants can plot the curves and play the videos while training continues.

Timings on an 8 × W7900D server with two participants training at the same time (GPUs 0-3 and 4-7):

| | Time |
| --- | --- |
| Start-up (Ray, model loading, environments) | ~2.5 min |
| Step 1 | ~130 s |
| Every other step | 55–75 s |
| Evaluation (32 episodes) | ~90 s alone, ~3 min when both runs evaluate together |
| **Launch → first evaluation result (after step 10)** | **17.5–17.6 min** (measured) |
| Each further 10 steps + evaluation | ~13 min (estimated) |
| Whole 40-step run | ~55 min (estimated) |

In a 40-minute session participants see the first two evaluations, and possibly the third. Ask them to run the notebook's stop cell at the end of the session if training is still running.

## Requirements

- An RLinf ROCm RoboTwin container image (`BUILD_TARGET=embodied-robotwin`, `PLATFORM=amd`, `ROCM_ARCHS=gfx1100`; see the [RLinf RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)) with JupyterLab added to `/opt/venv/lingbotvla`.
- 4 AMD Radeon PRO W7900D GPUs per participant (two participants per 8-GPU server).
- Mounted into the container: the RLinf source at `/workspace/RLinf`, RoboTwin (`RLinf_support` branch) with its assets at `/opt/robotwin`, and these models under `/data/models`: `lingbot-vla-4b-posttrain-robotwin`, `Qwen2.5-VL-3B-Instruct`, `lingbot-vla-4b`.

The notebook's appendix has the derived Dockerfile and the `docker run` command for each participant's container.
