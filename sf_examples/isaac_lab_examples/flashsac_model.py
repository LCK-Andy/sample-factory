"""FlashSAC network architecture (Holiday-Robot, arXiv:2604.04539) ported to Sample
Factory's APPO/PPO trainer.

Ported layer-for-layer from the official flash_rl implementation
(flash_rl/agents/flashSAC/{network,layer}.py), at its published defaults:

    actor:  embedder(obs->128) -> 2 x FlashSACBlock(128, expansion 4) -> RMSNorm
            -> NormalTanhPolicy head (23-dim mean + state-dependent log-std,
               tanh-normalized into [-10, 2])
    critic: 2-member ensemble (clipped-double style), each
            embedder(obs->256) -> 2 x FlashSACBlock(256, expansion 4) -> RMSNorm
            -> 101-bin categorical value head (support linspace(-vmax, vmax, 101)),
            expectation used as the value estimate, min over the ensemble.

Deliberate adaptations for on-policy PPO (each is forced by the algorithm, not
preference -- PPO recomputes log-probs on minibatches and needs exact ratios):

1. BatchNorm outputs use RUNNING statistics always (stats still update during
   training forwards). FlashSAC's UnitBatchNorm uses batch statistics, which makes
   a sample's output depend on its neighbors; under PPO that shifts log-probs
   between rollout (full batch) and update (minibatch) and corrupts importance
   ratios. Running-stat outputs are per-sample deterministic.
2. The critic predicts V(s) from the observation alone (PPO value baseline), not
   Q(s, a) (no action input, no target networks, no min-of-double-Q target --
   the min over the 2 members is kept as the reported value for FlashSAC
   fidelity).
3. No tanh squashing on actions: SF's PPO computes exact Gaussian log-probs and
   clips samples to the action space; a tanh wrapper would need Jacobian
   corrections SF's distribution machinery does not implement. Mean/log-std
   heads are ported verbatim.
4. UnitLinear weight re-normalization after each optimizer step (FlashSAC
   normalize_parameters) is not ported: SF has no post-step hook. Orthogonal
   init (gain 1) is kept.
"""

from typing import Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace

# FlashSAC published defaults (configs/agent/flashSAC.yaml)
ACTOR_HIDDEN = 128
CRITIC_HIDDEN = 256
NUM_BLOCKS = 2
EXPANSION = 4
NUM_BINS = 101
NUM_QS = 2
VMAX = 10.0  # FlashSAC uses +-5 on normalized returns; widened for PPO slack
LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0


class UnitLinear(nn.Module):
    """Bias-free linear with orthogonal init (weight re-norm not ported, see module doc)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x):
        return self.w(x)


class RunningBatchNorm(nn.Module):
    """FlashSAC's UnitBatchNorm with per-sample-deterministic output (see module doc, item 1)."""

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                self.running_mean.lerp_(x.mean(dim=0), self.momentum)
                self.running_var.lerp_(x.var(dim=0, unbiased=False), self.momentum)
        # clone: F.batch_norm saves the stats tensors for backward; without the copy,
        # a later in-place lerp_ between a forward and its backward breaks autograd
        return F.batch_norm(x, self.running_mean.clone(), self.running_var.clone(), self.weight, self.bias, training=False)


class UnitRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)


class FlashSACBlock(nn.Module):
    """Residual block: w1 -> BN -> ReLU -> w2 -> BN -> ReLU -> + residual."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w1 = UnitLinear(hidden, hidden * EXPANSION)
        self.w2 = UnitLinear(hidden * EXPANSION, hidden)
        self.norm1 = RunningBatchNorm(hidden * EXPANSION)
        self.norm2 = RunningBatchNorm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.norm1(self.w1(x)))
        x = F.relu(self.norm2(self.w2(x)))
        return x + residual


class FlashSACTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.norm = RunningBatchNorm(in_dim)
        self.embed = UnitLinear(in_dim, hidden)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden) for _ in range(NUM_BLOCKS)])
        self.post_norm = UnitRMSNorm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(self.norm(x))
        for block in self.blocks:
            x = block(x)
        return self.post_norm(x)


class NormalPolicyHead(nn.Module):
    """FlashSAC NormalTanhPolicy without the tanh (module doc, item 3)."""

    def __init__(self, hidden: int, action_dim: int):
        super().__init__()
        self.mean_w = UnitLinear(hidden, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = UnitLinear(hidden, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = self.mean_w(x) + self.mean_bias
        raw_log_std = self.std_w(x) + self.std_bias
        # tanh-normalized log-std, verbatim from FlashSAC
        log_std = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * 0.5 * (1 + torch.tanh(raw_log_std))
        return torch.cat([mean, log_std], dim=1)


class CategoricalValueHead(nn.Module):
    """101-bin categorical head; expectation over the fixed support is the value."""

    def __init__(self, hidden: int, num_bins: int, vmin: float, vmax: float):
        super().__init__()
        self.w = UnitLinear(hidden, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_bins))
        self.register_buffer("bin_values", torch.linspace(vmin, vmax, num_bins))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.w(x) + self.bias
        return (F.softmax(logits, dim=-1) @ self.bin_values).squeeze(-1)


class FlashSACActorCritic(ActorCritic):
    """FlashSAC actor + ensemble categorical critic behind SF's ActorCritic interface."""

    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(obs_space, action_space, cfg)

        if isinstance(obs_space, gym.spaces.Dict):
            assert list(obs_space.spaces.keys()) == ["obs"], f"unexpected obs keys {obs_space.spaces}"
            obs_space = obs_space["obs"]
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.actor_trunk = FlashSACTrunk(obs_dim, ACTOR_HIDDEN)
        self.policy_head = NormalPolicyHead(ACTOR_HIDDEN, action_dim)

        self.critic_trunks = nn.ModuleList([FlashSACTrunk(obs_dim, CRITIC_HIDDEN) for _ in range(NUM_QS)])
        self.value_heads = nn.ModuleList([CategoricalValueHead(CRITIC_HIDDEN, NUM_BINS, -VMAX, VMAX) for _ in range(NUM_QS)])

        self._actor_slice = ACTOR_HIDDEN  # head-output layout: [actor | critic0 | critic1]

    def _values(self, x: Tensor) -> Tensor:
        member_values = torch.stack([head(trunk(x)) for trunk, head in zip(self.critic_trunks, self.value_heads)])
        return member_values.min(dim=0).values  # min over the 2 members (clipped-double style)

    # ---- SF three-stage API: the learner's PPO path calls these directly ----
    # head output = concat(actor trunk, critic trunks...); core = identity (feed-forward);
    # tail splits it back. Sizes are fixed at construction, so the split is exact.

    def _head(self, x: Tensor) -> Tensor:
        return torch.cat([self.actor_trunk(x)] + [trunk(x) for trunk in self.critic_trunks], dim=1)

    def _tail_values(self, core_output: Tensor) -> Tensor:
        feats = core_output[:, self._actor_slice :]
        member_values = torch.stack([head(f) for head, f in zip(self.value_heads, feats.chunk(NUM_QS, dim=1))])
        return member_values.min(dim=0).values

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        return self._head(normalized_obs_dict["obs"])

    def forward_core(self, head_output: Tensor, rnn_states: Tensor):
        return head_output, rnn_states  # feed-forward: identity

    def forward_tail(
        self,
        core_output: Tensor,
        values_only: bool = False,
        sample_actions: bool = True,
        action_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        if values_only:
            return TensorDict(values=self._tail_values(core_output))

        action_logits = self.policy_head(core_output[:, : self._actor_slice])
        self.last_action_distribution = get_action_distribution(self.action_space, raw_logits=action_logits)

        result = TensorDict()
        result["values"] = self._tail_values(core_output)
        result["action_logits"] = action_logits
        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self,
        normalized_obs_dict: Dict[str, Tensor],
        rnn_states: Tensor,
        values_only: bool = False,
        sample_actions: bool = True,
        action_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        head_output = self.forward_head(normalized_obs_dict)
        core_output, new_rnn_states = self.forward_core(head_output, rnn_states)
        result = self.forward_tail(core_output, values_only=values_only, sample_actions=sample_actions, action_mask=action_mask)
        result["new_rnn_states"] = new_rnn_states
        return result

    # the base class introspects encoders[0] for these; this model has no SF encoders
    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        from sample_factory.model.model_utils import model_device

        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


def make_flashsac_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    return FlashSACActorCritic(obs_space, action_space, cfg)


def register_flashsac_model() -> None:
    from sample_factory.algo.utils.context import global_model_factory

    global_model_factory().register_actor_critic_factory(make_flashsac_actor_critic)
    print(
        f"[flashsac-model] registered: actor {NUM_BLOCKS}x{ACTOR_HIDDEN} (exp {EXPANSION}), "
        f"critic {NUM_QS}x({NUM_BLOCKS}x{CRITIC_HIDDEN}, {NUM_BINS} bins, +-{VMAX})",
        flush=True,
    )
