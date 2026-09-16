"""Unit tests for the stateless causal-transformer actor-critic.

Covers sf_examples/isaac_lab_examples/transformer_model.py, which is pure SF
(no Isaac Lab imports) and therefore testable everywhere: a tiny
continuous-action env whose obs IS a stacked window (K frames x D features)
exercises the full train + enjoy path, plus direct model-contract tests
(PackedSequence rejection under use_rnn=True, RoPE cross-window-size transfer).
"""

import shutil
from collections import deque
from os.path import isdir
from typing import Optional

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from sample_factory.algo.utils.context import reset_global_context
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir, log
from sf_examples.isaac_lab_examples.transformer_model import TransformerActorCritic, register_transformer_model

ENV_ID = "transformer_test_env_v1"
K, D = 4, 4  # window length and per-frame feature count; obs dim = K * D = 16
TINY_TRUNK = ["--il_tr_d_model=32", "--il_tr_layers=2", "--il_tr_heads=4"]


class StackedWindowEnv(gym.Env):
    """Obs is a K-frame window [oldest ... newest], each frame [n1, n2, n3, target].

    Reward = -|action - target_of_newest_frame|: the best policy reads the LAST
    frame's target feature. Targets persist across a few frames so the window's
    history carries (mildly) more information than the newest frame alone.
    """

    def __init__(self, full_env_name, cfg, _env_config=None, render_mode: Optional[str] = None):
        self.cfg = cfg
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (K * D,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.render_mode = render_mode
        self._targets: deque = deque(maxlen=K)
        self._steps = 0

    def _obs(self) -> np.ndarray:
        frames = [np.array([0.0, 0.0, 0.0, t], dtype=np.float32) for t in self._targets]
        return np.concatenate(frames)  # oldest first, newest last (bridge convention)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        target = float(np.clip(np.random.uniform(-0.8, 0.8), -0.8, 0.8))
        self._targets.clear()
        self._targets.extend([target] * K)
        return self._obs(), {}

    def step(self, action):
        # mild persistence: new target stays halfway close to the previous one
        fresh = float(np.clip(np.random.uniform(-0.8, 0.8), -0.8, 0.8))
        target = float(np.clip(0.5 * self._targets[-1] + 0.5 * fresh, -0.8, 0.8))
        self._targets.append(target)
        reward = -abs(float(np.asarray(action).reshape(-1)[0]) - target)
        self._steps += 1
        terminated, truncated = False, self._steps >= self.cfg.transformer_test_episode_len
        return self._obs(), reward, terminated, truncated, {}


def make_stacked_window_env(full_env_name, cfg=None, _env_config=None, render_mode=None):
    return StackedWindowEnv(full_env_name, cfg, render_mode=render_mode)


def add_extra_params(parser):
    parser.add_argument("--il_framestack", default=K, type=int, help="Window length K for the test env")
    parser.add_argument("--transformer_test_episode_len", default=50, type=int, help="Episode length")
    parser.add_argument("--il_tr_d_model", default=32, type=int, help="transformer trunk width")
    parser.add_argument("--il_tr_layers", default=2, type=int, help="transformer encoder layers")
    parser.add_argument("--il_tr_heads", default=4, type=int, help="attention heads")


def parse_test_args(argv=None, evaluation=False) -> Config:
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_extra_params(parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def register_test_components():
    register_env(ENV_ID, make_stacked_window_env)
    register_transformer_model()


def default_test_cfg():
    argv = ["--algo=APPO", f"--env={ENV_ID}", "--experiment=test_transformer_model", "--use_rnn=False"] + TINY_TRUNK
    cfg = parse_test_args(argv=argv)
    cfg.num_workers = 1
    cfg.num_envs_per_worker = 2
    cfg.train_for_env_steps = 128
    cfg.batch_size = 64
    cfg.batched_sampling = False
    cfg.async_rl = True
    cfg.save_every_sec = 4
    cfg.decorrelate_experience_max_seconds = 0
    cfg.decorrelate_envs_on_one_worker = False
    cfg.seed = 0
    cfg.device = "cpu"
    cfg.learning_rate = 1e-3
    cfg.normalize_input = True
    cfg.normalize_returns = True  # untrained value fn is far from -37/episode returns otherwise
    cfg.initial_stddev = 0.3  # sigma=1 exploration noise drowns the |action - target| signal

    eval_cfg = parse_test_args(argv=argv, evaluation=True)
    eval_cfg.device = "cpu"
    eval_cfg.max_num_frames = 1000
    eval_cfg.no_render = True
    return cfg, eval_cfg


def run_test_env(cfg: Config, eval_cfg: Config, expected_reward_at_least: float, expected_reward_at_most: float):
    register_test_components()

    directory = experiment_dir(cfg=cfg, mkdir=False)
    if isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)

    cfg, runner = make_runner(cfg)
    status = runner.init()
    assert status == ExperimentStatus.SUCCESS
    status = runner.run()
    assert status == ExperimentStatus.SUCCESS

    status, avg_reward = enjoy(eval_cfg)
    log.debug(f"Test reward: {avg_reward:.4f}")

    try:
        assert status == ExperimentStatus.SUCCESS
        assert avg_reward >= expected_reward_at_least
        assert avg_reward <= expected_reward_at_most
    finally:
        assert isdir(directory)
        shutil.rmtree(directory, ignore_errors=True)

    reset_global_context()


class TestTransformerModel:
    def test_sanity_train_and_enjoy(self):
        """Short train + enjoy roundtrip: nothing crashes, reward is finite."""
        cfg, eval_cfg = default_test_cfg()
        run_test_env(cfg, eval_cfg, expected_reward_at_least=-60.0, expected_reward_at_most=0.0)

    def test_full_run(self):
        """Actually train: policy learns to track the newest frame's target."""
        cfg, eval_cfg = default_test_cfg()
        cfg.train_for_env_steps = 100000
        cfg.batch_size = 64  # more updates per env step on this tiny task
        cfg.learning_rate = 3e-4
        cfg.lr_warmup_updates = 25  # also exercises the WarmupScheduler path
        # untrained deterministic episode sum is ~50 * -0.75 ~= -37; learned lands near -5
        run_test_env(cfg, eval_cfg, expected_reward_at_least=-20.0, expected_reward_at_most=0.0)

    def test_packed_sequence_rejected_with_readable_error(self):
        """The stateless core must fail loudly (not cryptically) if launched with use_rnn=True."""
        model, cfg = self._make_model(framestack=K)
        packed = pack_padded_sequence(torch.rand(2, 3, 8), lengths=[3, 2], enforce_sorted=False)
        with pytest.raises(RuntimeError, match="use_rnn=False"):
            model.forward_core(packed, rnn_states=torch.zeros(2, cfg.rnn_size))

    def test_rope_transfers_across_window_sizes(self):
        """No POLICY parameter may depend on K: a K=4 checkpoint's trunk+heads must load
        (matching shapes) into a K=8 model and both produce finite trunk outputs for
        their window lengths. Caveat captured here on purpose: the obs normalizer's
        running stats are [K*D]-shaped (slot-dependent), so a cross-K evaluation pass
        starts with fresh normalizer stats -- the policy net itself transfers."""
        model_k4, _ = self._make_model(framestack=4)
        model_k8, _ = self._make_model(framestack=8)

        out4 = model_k4.trunk(torch.rand(5, 4 * D))
        assert out4.shape == (5, 32) and torch.isfinite(out4).all()

        policy_weights = {k: v for k, v in model_k4.state_dict().items() if not k.startswith("obs_normalizer")}
        result = model_k8.load_state_dict(policy_weights, strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith("obs_normalizer") for k in result.missing_keys)  # only stats are K-shaped
        out8 = model_k8.trunk(torch.rand(5, 8 * D))
        assert out8.shape == (5, 32) and torch.isfinite(out8).all()

    def test_forward_contract(self):
        """Full forward returns actions/log-probs/values and passes rnn_states through unchanged."""
        model, cfg = self._make_model(framestack=K)
        rnn_states = torch.zeros(5, cfg.rnn_size)
        result = model({"obs": torch.rand(5, K * D)}, rnn_states)
        assert result["actions"].shape == (5,)
        assert result["values"].shape == (5,)
        assert torch.isfinite(result["log_prob_actions"]).all()
        assert torch.equal(result["new_rnn_states"], rnn_states)  # identity core

    @staticmethod
    def _make_model(framestack: int):
        argv = [
            "--algo=APPO",
            f"--env={ENV_ID}",
            "--experiment=transformer_model_contract",
            "--use_rnn=False",
            f"--il_framestack={framestack}",
        ] + TINY_TRUNK
        cfg = parse_test_args(argv=argv)
        obs_space = gym.spaces.Dict(dict(obs=gym.spaces.Box(-np.inf, np.inf, (framestack * D,), dtype=np.float32)))
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        return TransformerActorCritic(obs_space, action_space, cfg), cfg
