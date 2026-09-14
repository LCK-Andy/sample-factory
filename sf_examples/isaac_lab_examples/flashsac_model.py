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
from torch.nn.utils.rnn import PackedSequence

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
    """FlashSAC actor + ensemble categorical critic behind SF's ActorCritic interface.

    Separate actor and critic networks (FlashSAC's own topology); GRU core on the
    actor path only (SF's recurrent posture -- see the note in __init__).
    """

    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(obs_space, action_space, cfg)

        if isinstance(obs_space, gym.spaces.Dict):
            assert list(obs_space.spaces.keys()) == ["obs"], f"unexpected obs keys {obs_space.spaces}"
            obs_space = obs_space["obs"]
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.actor_trunk = FlashSACTrunk(obs_dim, ACTOR_HIDDEN)
        # Recurrent memory on the actor path (SF's GRU core posture): the trunks are
        # FlashSAC's, but this task's tactile inference needs history beyond the obs's
        # built-in 5-step window -- the screen measured rnn-off at -2.22 vs +0.13 baseline.
        # The critic stays feed-forward (value estimation from the obs history suffices),
        # which also keeps SF's single-rnn-state-per-agent contract intact.
        self.rnn_size = cfg.rnn_size
        self.actor_gru = nn.GRU(ACTOR_HIDDEN, self.rnn_size, cfg.rnn_num_layers)
        self.policy_head = NormalPolicyHead(self.rnn_size, action_dim)

        self.critic_trunks = nn.ModuleList([FlashSACTrunk(obs_dim, CRITIC_HIDDEN) for _ in range(NUM_QS)])
        self.value_heads = nn.ModuleList([CategoricalValueHead(CRITIC_HIDDEN, NUM_BINS, -VMAX, VMAX) for _ in range(NUM_QS)])

        # head-output layout: [gru output | critic0 | critic1]
        self._actor_slice = self.rnn_size

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

    def forward_core(self, head_output, rnn_states):
        """GRU over the actor slice; critic features pass through unchanged.

        Handles both plain [B, D] tensors (rollout inference) and PackedSequences
        (learner BPTT). build_rnn_inputs returns rnn_states already ordered to
        match the packed sequences (indexed by rollout starts) -- the same
        contract SF's own ModelCoreRNN relies on.
        """
        is_seq = not torch.is_tensor(head_output)
        h0 = rnn_states.unsqueeze(0).contiguous()

        if is_seq:
            data, batch_sizes, sorted_indices, unsorted_indices = head_output
            actor_feat = PackedSequence(data[:, :ACTOR_HIDDEN], batch_sizes, sorted_indices, unsorted_indices)
            critic_feat = data[:, ACTOR_HIDDEN:]
            gru_out, new_states = self.actor_gru(actor_feat, h0)
            out_data = torch.cat([gru_out.data, critic_feat], dim=1)
            return PackedSequence(out_data, batch_sizes, sorted_indices, unsorted_indices), new_states.squeeze(0)

        actor_feat = head_output[:, :ACTOR_HIDDEN]
        critic_feat = head_output[:, ACTOR_HIDDEN:]
        gru_out, new_states = self.actor_gru(actor_feat.unsqueeze(0), h0)
        return torch.cat([gru_out.squeeze(0), critic_feat], dim=1), new_states.squeeze(0)

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


def run_gru(gru: nn.GRU, x, rnn_states: Tensor):
    """GRU over a plain [B, D] tensor or a PackedSequence (learner BPTT).

    build_rnn_inputs returns rnn_states already ordered to match the packed
    sequences (indexed by rollout starts) -- the same contract SF's own
    ModelCoreRNN relies on.
    """
    h0 = rnn_states.unsqueeze(0).contiguous()
    if torch.is_tensor(x):
        x = x.unsqueeze(0)
        out, new_states = gru(x, h0)
        return out.squeeze(0), new_states.squeeze(0)
    out, new_states = gru(x, h0)
    return out, new_states.squeeze(0)


# ---- the COMBINED architecture: APPO topology, FlashSAC vocabulary ----
SHARED_HIDDEN = 256  # trunk width; the GRU still widens to cfg.rnn_size (512)


class FlashSACSharedActorCritic(ActorCritic):
    """FlashSAC components arranged in SF-APPO's topology.

    APPO contributes the SHAPE (one shared trunk for actor and critic, a GRU core
    between trunk and heads, obs/returns normalizers -- the same topology as the
    model that trained the 1B dexlift policy). FlashSAC contributes the PARTS:
    UnitLinear embedder, residual blocks with expansion 4, RMSNorm, a
    state-dependent tanh-normalized log-std policy head, and an ensemble of
    101-bin categorical value heads (min over 2 members). The ensemble here is
    head-level on the shared features rather than network-level, since the
    trunk is shared.
    """

    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(obs_space, action_space, cfg)

        if isinstance(obs_space, gym.spaces.Dict):
            assert list(obs_space.spaces.keys()) == ["obs"], f"unexpected obs keys {obs_space.spaces}"
            obs_space = obs_space["obs"]
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.shared_trunk = FlashSACTrunk(obs_dim, SHARED_HIDDEN)
        self.rnn_size = cfg.rnn_size
        self.core_gru = nn.GRU(SHARED_HIDDEN, self.rnn_size, cfg.rnn_num_layers)
        self.policy_head = NormalPolicyHead(self.rnn_size, action_dim)
        self.value_heads = nn.ModuleList(
            [CategoricalValueHead(self.rnn_size, NUM_BINS, -VMAX, VMAX) for _ in range(NUM_QS)]
        )

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        return self.shared_trunk(normalized_obs_dict["obs"])

    def forward_core(self, head_output, rnn_states):
        return run_gru(self.core_gru, head_output, rnn_states)

    def _tail_values(self, core_output: Tensor) -> Tensor:
        member_values = torch.stack([head(core_output) for head in self.value_heads])
        return member_values.min(dim=0).values

    def forward_tail(
        self,
        core_output: Tensor,
        values_only: bool = False,
        sample_actions: bool = True,
        action_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        if values_only:
            return TensorDict(values=self._tail_values(core_output))

        action_logits = self.policy_head(core_output)
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


def make_flashsac_shared_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    return FlashSACSharedActorCritic(obs_space, action_space, cfg)


def register_flashsac_model(shared: bool = False) -> None:
    from sample_factory.algo.utils.context import global_model_factory

    if shared:
        global_model_factory().register_actor_critic_factory(make_flashsac_shared_actor_critic)
        print(
            f"[flashsac-model] registered SHARED variant (APPO topology): trunk {NUM_BLOCKS}x{SHARED_HIDDEN} "
            f"(exp {EXPANSION}) -> GRU -> cfg.rnn_size -> policy head + {NUM_QS}x {NUM_BINS}-bin value heads (+-{VMAX})",
            flush=True,
        )
    else:
        global_model_factory().register_actor_critic_factory(make_flashsac_actor_critic)
        print(
            f"[flashsac-model] registered (separate nets): actor {NUM_BLOCKS}x{ACTOR_HIDDEN} (exp {EXPANSION}) "
            f"+ GRU -> policy head, critic {NUM_QS}x({NUM_BLOCKS}x{CRITIC_HIDDEN}, {NUM_BINS} bins, +-{VMAX})",
            flush=True,
        )
