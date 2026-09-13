"""Train Sample Factory algorithms on Isaac Lab (isaac-sim/IsaacLab) environments.

Bridges Isaac Lab's GPU-vectorized DirectRLEnv into Sample Factory's batched
(non-batched-per-agent) sampling mode, mirroring the IsaacGym integration's
shape: one SF "environment" object IS the whole vectorized sim, obs arrive as
a single GPU tensor per step.

Usage (laptop, Newton kit-less physics, no Isaac Sim app needed):
    python -m sf_examples.isaac_lab_examples.train_isaaclab \
        --algo=APPO --env=Isaac-Ant-Direct-v0 --experiment=il-ant \
        --batched_sampling=True --serial_mode --num_envs=512 --train_for_env_steps=2000000
"""

import argparse
import sys
from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch
from torch import Tensor

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

_ENV_SINGLETON: dict = {}


class IsaacLabVecEnv(gym.Env):
    """Adapts an Isaac Lab DirectRLEnv to SF's batched-env contract."""

    def __init__(self, env):
        self.env = env
        raw = env.unwrapped  # gymnasium OrderEnforcing wrapper hides Isaac Lab attrs
        self._device = raw.device
        self.num_agents = raw.num_envs
        self.action_space = convert_space(raw.single_action_space)
        self.observation_space = gym.spaces.Dict(dict(obs=convert_space(raw.single_observation_space["policy"])))

    def _obs(self, obs_dict) -> Dict[str, Tensor]:
        return {"obs": obs_dict["policy"]}

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        obs_dict, infos = self.env.reset()
        return self._obs(obs_dict), {}

    def step(self, actions) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        if not torch.is_tensor(actions):  # SF hands over numpy unless --env_gpu_actions
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        obs_dict, rew, terminated, truncated, infos = self.env.step(actions)
        return self._obs(obs_dict), rew, terminated, truncated, {}

    def render(self):
        pass


def make_isaaclab_env(full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None) -> gym.Env:
    """SF factory: builds one Isaac Lab vectorized env (newton preset, kit-less).

    The preset is resolved hydra-style with sys.argv swapped out (same trick as
    flash_rl's wrapper), so SF's own CLI args are invisible to hydra.
    """
    import os
    # SF re-fires RolloutWorker.init (once per inference worker), calling the
    # factory twice in one process; Isaac Lab allows a single sim context per
    # process, so serve a per-process singleton instead of rebuilding.
    key = (full_env_name, os.getpid())
    if key in _ENV_SINGLETON:
        print(f"[bridge] reusing cached env pid={os.getpid()}", flush=True)
        return _ENV_SINGLETON[key]
    physics = getattr(cfg, "il_physics", "newton_mjwarp") if cfg is not None else "newton_mjwarp"
    print(f"[bridge] make_isaaclab_env CALLED pid={os.getpid()} physics={physics}", flush=True)

    sf_device = getattr(cfg, "device", "gpu") if cfg is not None else "gpu"
    torch_device = "cuda:0" if "gpu" in sf_device else "cpu"

    if physics == "physx":
        # app-first boot: PhysX needs the full Isaac Sim app running BEFORE any
        # task-package import that touches pxr (dual-USD crash otherwise)
        import isaacsim  # noqa: F401

        from isaaclab.app import AppLauncher

        AppLauncher({"headless": True, "device": torch_device, "enable_cameras": False})

    import isaaclab_tasks  # noqa: F401  (task registration)
    from isaaclab_tasks.utils.hydra import resolve_task_config

    _argv = sys.argv
    sys.argv = [_argv[0], f"presets={physics}"]
    try:
        env_cfg, _agent_cfg = resolve_task_config(full_env_name, "rsl_rl_cfg_entry_point")
    finally:
        sys.argv = _argv

    env_cfg.sim.device = torch_device
    env_cfg.scene.num_envs = getattr(cfg, "il_num_envs", 512) if cfg is not None else 512
    env_cfg.seed = getattr(cfg, "seed", 0) if cfg is not None else 0

    env = gym.make(full_env_name, cfg=env_cfg)
    wrapped = IsaacLabVecEnv(env)
    _ENV_SINGLETON[key] = wrapped
    return wrapped


def add_extra_params_func(parser: argparse.ArgumentParser) -> None:
    p = parser
    # set_defaults BEFORE parse_full_cfg: values become true argparse defaults
    # and reach every component (post-parse cfg mutation does NOT propagate to
    # BufferMgr/sampler -- both num_workers pinning and rollout/batch sizing
    # were silently lost that way). batched mode: one SF env = the whole
    # GPU-vectorized sim, so exactly one env instance per worker; and SF
    # requires batch_size/rollout trajectories (4096/32=128) to divide
    # num_agents (512).
    p.set_defaults(
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        rollout=32,
        batch_size=4096,
        num_batches_per_epoch=2,
    )
    p.add_argument(
        "--il_physics",
        default="newton_mjwarp",
        type=str,
        choices=["newton_mjwarp", "physx"],
        help="Isaac Lab physics preset: newton_mjwarp (kit-less) or physx (full Isaac Sim, headless)",
    )
    p.add_argument(
        "--il_num_envs",
        default=512,
        type=int,
        help="Number of parallel Isaac Lab envs inside the single batched env",
    )


def custom_env_override_defaults(cfg) -> None:
    """Network/hyperparameter shaping (structural pins live in
    add_extra_params_func via parser.set_defaults -- see the note there)."""
    cfg.encoder_mlp_layers = [400, 200, 100]  # match the rsl-rl baseline sizing
    cfg.hidden_mlp_layers = [256, 128]
    cfg.learning_rate = 3e-4
    cfg.adam_eps = 1e-5


def register_isaaclab_envs() -> None:
    register_env("Isaac-Ant-Direct-v0", make_isaaclab_env)


def parse_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    cfg = parse_full_cfg(parser, argv=argv)
    custom_env_override_defaults(cfg)  # pins num_workers/num_envs_per_worker=1 etc.
    return cfg


def main() -> None:
    register_isaaclab_envs()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    main()
