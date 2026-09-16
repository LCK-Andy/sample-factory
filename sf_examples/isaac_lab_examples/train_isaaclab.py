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
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

_ENV_SINGLETON: dict = {}


class IsaacLabVecEnv(gym.Env):
    """Adapts an Isaac Lab vectorized env (DirectRLEnv or ManagerBasedRLEnv) to SF's batched-env contract."""

    def __init__(self, env, obs_keys=("policy",), framestack: int = 1):
        self.env = env
        raw = env.unwrapped  # gymnasium OrderEnforcing wrapper hides Isaac Lab attrs
        self._device = raw.device
        self.num_agents = raw.num_envs
        self.action_space = convert_space(raw.single_action_space)
        self._obs_keys = tuple(obs_keys)
        group_spaces = [convert_space(raw.single_observation_space[k]) for k in self._obs_keys]
        if len(group_spaces) == 1:
            obs_space = group_spaces[0]
        else:
            # manager-based tasks expose several obs groups (each flat, since the
            # task sets concatenate_terms=True); the agent consumes their concat
            low = np.concatenate([s.low for s in group_spaces])
            high = np.concatenate([s.high for s in group_spaces])
            obs_space = gym.spaces.Box(low, high, dtype=group_spaces[0].dtype)
        self.framestack = max(1, framestack)
        if self.framestack > 1:
            # sliding-window history (see _obs); the agent sees the K frames concat'd
            low = np.concatenate([obs_space.low] * self.framestack)
            high = np.concatenate([obs_space.high] * self.framestack)
            obs_space = gym.spaces.Box(low, high, dtype=obs_space.dtype)
        self.observation_space = gym.spaces.Dict(dict(obs=obs_space))
        self._hist: Optional[torch.Tensor] = None  # [N, K, D] on the sim device

    def _obs(self, obs_dict, done: Optional[torch.Tensor] = None) -> Dict[str, Tensor]:
        if len(self._obs_keys) == 1:
            x = obs_dict[self._obs_keys[0]]
        else:
            x = torch.cat([obs_dict[k] for k in self._obs_keys], dim=-1)
        if self.framestack > 1:
            # Sliding window on the GPU, oldest -> newest along dim 1. DirectRLEnv
            # resets done envs BEFORE computing the step's obs, so an obs that
            # arrives with done=True is the new episode's FIRST frame: re-fill that
            # row's whole window with it (repeat-fill -- standard frame-stack
            # treatment, no padded tokens for attention to trip over).
            if self._hist is None:
                self._hist = x.unsqueeze(1).repeat(1, self.framestack, 1)
            else:
                self._hist = torch.cat([self._hist[:, 1:], x.unsqueeze(1)], dim=1)
                if done is not None:
                    done_rows = done.to(device=x.device, dtype=torch.bool).reshape(-1)
                    if done_rows.any():
                        self._hist[done_rows] = x[done_rows].unsqueeze(1)
            x = self._hist.reshape(self.num_agents, -1)
        return {"obs": x}

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        obs_dict, infos = self.env.reset()
        return self._obs(obs_dict), {}

    def step(self, actions) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        if not torch.is_tensor(actions):  # SF hands over numpy unless --env_gpu_actions
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        obs_dict, rew, terminated, truncated, infos = self.env.step(actions)
        done = torch.logical_or(terminated, truncated)
        return self._obs(obs_dict, done), rew, terminated, truncated, {}

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

        # Kit sizes its tasking pool from os.cpu_count(), which reports the HOST
        # core count; inside a CPU-quota'd container that oversubscribes the
        # cgroup and the container spends its quota on context switches. Pass the
        # cap through sys.argv, the same transport Isaac Lab uses for distributed
        # runs (it strips the arg again once the app is up).
        cpu_threads = os.environ.get("IL_CPU_THREADS")
        if cpu_threads:
            sys.argv.append(f"--/plugins/carb.tasking.plugin/threadCount={cpu_threads}")
            print(f"[bridge] capping Kit tasking pool at {cpu_threads} threads", flush=True)

        # GUI playback: default is headless; IL_RENDER=1 boots the full Kit
        # visualizer so the env's own rendering draws the scene while it steps
        # (same AppLauncher posture as the flash_rl play path).
        headless = os.environ.get("IL_RENDER", "") != "1"
        kit_args = {"headless": headless, "device": torch_device, "enable_cameras": False}
        if not headless:
            kit_args["visualizer"] = ["kit"]
            kit_args["visualizer_explicit"] = True
        AppLauncher(kit_args)

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

    # Freeze the ADR curriculum at a fixed difficulty when asked. DifficultyScheduler
    # clamps to [min_difficulty, max_difficulty], so pinning init=min=max removes both
    # promotion and demotion and holds difficulty_frac constant, which fixes every
    # interpolation term. Needed for hyperparameter comparison: the curriculum ramps on
    # a policy-dependent schedule, so without this a *better* policy can score *lower*
    # simply because it was promoted to a harder setting. Fails loudly on tasks with no
    # ADR term rather than silently measuring the wrong thing.
    adr_freeze = os.environ.get("IL_ADR_DIFFICULTY")
    if adr_freeze is not None:
        adr_term = getattr(getattr(env_cfg, "curriculum", None), "adr", None)
        if adr_term is None or "init_difficulty" not in adr_term.params:
            raise RuntimeError(
                f"IL_ADR_DIFFICULTY={adr_freeze} was set but task {full_env_name!r} has no ADR curriculum term"
            )
        for key in ("init_difficulty", "min_difficulty", "max_difficulty"):
            adr_term.params[key] = int(adr_freeze)
        print(f"[bridge] ADR curriculum FROZEN at difficulty {adr_freeze}", flush=True)

    env = gym.make(full_env_name, cfg=env_cfg)
    obs_keys = tuple(getattr(cfg, "il_obs_groups", "policy").split(","))
    framestack = getattr(cfg, "il_framestack", 0) if cfg is not None else 0
    if framestack <= 0:  # auto: temporal models get a history window, everyone else none
        framestack = 8 if getattr(cfg, "il_model", "default") == "transformer" else 1
    wrapped = IsaacLabVecEnv(env, obs_keys=obs_keys, framestack=framestack)
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
    p.add_argument(
        "--il_obs_groups",
        default="policy",
        type=str,
        help="Comma-separated Isaac Lab obs groups concatenated into the agent obs "
        "(manager-based tasks, e.g. policy,proprio,perception for dexsuite)",
    )
    p.add_argument(
        "--il_model",
        default="default",
        type=str,
        choices=["default", "flashsac", "flashsac_shared", "transformer"],
        help="Model architecture: default (SF MLP + optional GRU); flashsac (FlashSAC's "
        "separate actor/critic residual-block nets, GRU on the actor path); flashsac_shared "
        "(the combination: APPO's shared-trunk + GRU topology built from FlashSAC's "
        "residual blocks, RMSNorm, ensemble 101-bin categorical value heads); transformer "
        "(causal pre-LN transformer over the il_framestack window, stateless -- pair with "
        "--use_rnn=False)",
    )
    p.add_argument(
        "--il_framestack",
        default=0,
        type=int,
        help="Sliding-window obs history K maintained on the GPU inside the bridge (obs dim "
        "becomes K x obs_dim, oldest first). 0 = auto: 8 for il_model=transformer, 1 (off) "
        "otherwise. Rows reset by repeat-fill at episode boundaries",
    )
    p.add_argument("--il_tr_d_model", default=256, type=int, help="transformer trunk width")
    p.add_argument("--il_tr_layers", default=2, type=int, help="transformer encoder layers")
    p.add_argument("--il_tr_heads", default=4, type=int, help="transformer attention heads (must divide il_tr_d_model)")


def custom_env_override_defaults(cfg) -> None:
    """Network/hyperparameter shaping (structural pins live in
    add_extra_params_func via parser.set_defaults -- see the note there).

    NOTE: runs AFTER CLI parsing, so anything set here trumps command-line flags.
    Hyperparameter pins are scoped strictly to the env they were matched for
    (Ant recipe, dexsuite sizing); every other env keeps CLI values -- an
    earlier version let the Ant lr/encoder pins fall through onto ALL tasks,
    which silently gave e.g. Repose-Cube MLP arms lr=3e-4 while custom-model
    arms fell through to SF's default 1e-4, confounding architecture
    comparisons run with "identical" flags."""
    if getattr(cfg, "il_model", "default") in ("flashsac", "flashsac_shared"):
        from sf_examples.isaac_lab_examples.flashsac_model import register_flashsac_model

        register_flashsac_model(shared=cfg.il_model == "flashsac_shared")
        return  # FlashSAC-based models ignore the SF encoder/decoder sizing knobs

    if getattr(cfg, "il_model", "default") == "transformer":
        from sf_examples.isaac_lab_examples.transformer_model import register_transformer_model

        register_transformer_model()
        # The stateless core rejects PackedSequence input with a readable error, but
        # the saved config of a transformer experiment already carries use_rnn=False;
        # a CLI --use_rnn=False also works for fresh launches.
        return

    cfg.hidden_mlp_layers = [256, 128]
    if "Dexsuite" in cfg.env:
        # match the task's rsl_rl cfg sizing ([512, 256, 128], elu)
        cfg.encoder_mlp_layers = [512, 256, 128]
    elif "Ant" in cfg.env:
        # Ant-only recipe (matched the rsl-rl IsaacGym baseline sizing)
        cfg.encoder_mlp_layers = [400, 200, 100]
        cfg.learning_rate = 3e-4
        cfg.adam_eps = 1e-5
    # everything else: no implicit pins, CLI values stand


def register_isaaclab_envs() -> None:
    register_env("Isaac-Ant-Direct-v0", make_isaaclab_env)
    register_env("Isaac-Repose-Cube-Allegro-Direct-v0", make_isaaclab_env)
    for env_id in (
        "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    ):
        register_env(env_id, make_isaaclab_env)


def parse_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    cfg = parse_full_cfg(parser, argv=argv)
    # SF loads train_dir/<exp>/config.json only LATER (inside run_rl/enjoy), but the
    # custom-model registration below must see the SAVED --il_model now (playback and
    # re-runs of a custom-arch experiment would otherwise build the default model and
    # crash on the checkpoint's state_dict). Loading here is idempotent and CLI values
    # keep precedence over the saved config.
    from sample_factory.cfg.arguments import maybe_load_from_checkpoint

    cfg = maybe_load_from_checkpoint(cfg)
    custom_env_override_defaults(cfg)  # pins num_workers/num_envs_per_worker=1 etc.
    return cfg


def main() -> None:
    register_isaaclab_envs()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    main()
