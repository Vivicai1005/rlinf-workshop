#!/bin/bash
# Smoke for rlinf-workshop on ONE W7900D (gfx1100), within the smoke Job's limits (8 CPUs, 48 GiB, default /dev/shm).
# The workshop itself trains Lingbot-VLA with GRPO on 4 GPUs; this checks the parts it is built from: the JupyterLab
# launch fast path, PyTorch and Triton on the GPU, RLinf and RoboTwin importing, and, when the workshop's data volume
# is mounted at /models/rlinf-workshop, two RoboTwin place_shoe environments rendering and stepping in SAPIEN (Vulkan)
# as RLinf's env workers run them. Nothing is downloaded.
set -euo pipefail
. "$(dirname "$0")/smoke_lib.sh"

# 1. The launch fast path: JupyterLab is baked in (jupyter and python on PATH are the Lingbot-VLA environment's).
jupyterlab_fast_path
openssh_baked
# 2. The GPU: exactly one gfx1100, and a half-precision matmul in PyTorch.
echo "== rocminfo"; grep -m1 -E 'gfx1100' <<<"$(rocminfo)"
echo "== torch"
python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected exactly one visible GPU"
arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
assert arch.startswith("gfx1100"), f"expected gfx1100, got {arch!r}"
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
assert torch.isfinite(b).all(), "non-finite matmul result"
print("torch", torch.__version__, "hip", torch.version.hip, "arch", arch)
import os, triton
assert triton.__version__ == "3.5.1", triton.__version__
assert os.environ.get("AMD_VULKAN_ICD") == "RADV" and os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE") == "TRUE"
print("triton", triton.__version__, "| RADV and the flash-attention Triton AMD backend are set for every process")
PY

# amd-smi runs as typed (the environment's amdsmi bindings are ROCm 7.2's own), and HIP start-up finds amdgpu.ids
# (no "(null): No such file or directory" line, the GPU has its product name).
echo "== amd-smi"; amd-smi version
out=$(python -c 'import torch; torch.zeros(1, device="cuda"); print("GPU name:", torch.cuda.get_device_name(0))' 2>&1)
echo "${out}" | tail -1
if grep -q "(null): No such file" <<<"${out}"; then echo "libdrm did not find amdgpu.ids"; exit 1; fi

# 3. RLinf and RoboTwin: build the workshop's place_shoe training environment from RLinf's own config (2 envs),
#    reset it (scene setup + rendering of the head and wrist cameras) and step it once with a hold-still action chunk.
#    Expected and harmless (the training prints them too): "missing pytorch3d" with a curobo ImportError traceback
#    (RoboTwin then uses the mplib planner, which the workshop config selects). SAPIEN's CUDA-only denoiser is switched
#    off in the image, so its "Failed to initialize denoiser" errors must not appear.
echo "== RLinf + RoboTwin place_shoe"
for d in background_texture objects embodiments; do
    test "$(readlink /opt/robotwin/assets/$d)" = /models/rlinf-workshop/robotwin-assets/$d
done
grep -q 'set_ray_tracing_denoiser("none")' /opt/robotwin/envs/_base_task.py
export REPO_PATH=/opt/RLinf EMBODIED_PATH=/opt/RLinf/examples/embodiment ROBOTWIN_PATH=/opt/robotwin ROBOT_PLATFORM=ALOHA
export LINGBOT_VLA_PATH=/opt/venv/lingbotvla/lingbot-vla PYTHONPATH=/opt/RLinf:/opt/robotwin PYTHONWARNINGS=ignore
cd /opt/RLinf
python -c "from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv; print('RLinf and its RoboTwin environment import')"
if [ ! -d /models/rlinf-workshop/robotwin-assets/objects ]; then
    echo "The data volume is not mounted at /models/rlinf-workshop: RoboTwin environment check skipped"
else
timeout 1200 python - 2>&1 <<'PY' | tee /tmp/robotwin_env.log
import time
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

with initialize_config_dir(config_dir="/opt/RLinf/examples/embodiment/config", version_base="1.1"):
    cfg = compose("robotwin_place_shoe_grpo_lingbotvla", overrides=[
        "env.train.assets_path=/opt/robotwin", "env.train.video_cfg.save_video=False"])
env_cfg = cfg.env.train
with open_dict(env_cfg):
    env_cfg.group_size = 1
t0 = time.time()
env = RoboTwinEnv(env_cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None)
obs, _ = env.reset()
print(f"reset: {time.time() - t0:.0f} s, observation keys {sorted(obs)}")
images = [v for v in obs.values() if isinstance(v, torch.Tensor) and v.dim() >= 4]
assert images, "no camera images in the observation"
for img in images:
    assert img.shape[0] == 2 and img.float().std() > 0, f"blank or wrong-shaped image {tuple(img.shape)}"
state = obs.get("states")
action = state[:, :14].cpu().numpy() if isinstance(state, torch.Tensor) else np.zeros((2, 14), dtype=np.float32)
t0 = time.time()
obs, reward, terminated, truncated, info = env.step(np.repeat(action[:, None, :], 5, axis=1))
assert torch.isfinite(torch.as_tensor(reward)).all(), "non-finite reward"
print(f"step (5 actions): {time.time() - t0:.0f} s, reward {torch.as_tensor(reward).tolist()}")
env.venv.close() if hasattr(env.venv, "close") else None
PY
if grep -q "Failed to initialize denoiser" /tmp/robotwin_env.log; then echo "SAPIEN still tries its denoiser"; exit 1; fi
fi

# 4. The notebook server starts and answers.
echo "== notebook"
(jupyter lab --no-browser --ip=127.0.0.1 --port=18888 --ServerApp.token=smoke --allow-root >/tmp/jl.log 2>&1 &)
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:18888/api/status?token=smoke" >/dev/null && { echo "jupyter ok"; break; }; sleep 2; done
curl -sf "http://127.0.0.1:18888/api/status?token=smoke" >/dev/null || { tail -20 /tmp/jl.log; exit 1; }

echo "SMOKE PASS"
