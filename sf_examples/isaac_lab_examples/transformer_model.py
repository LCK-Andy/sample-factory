"""GPT-style causal transformer policy for SF-APPO on Isaac Lab proprioceptive tasks.

Architecture (sliding-window attention, stateless):

    obs [B, K*D]  --reshape--> tokens [B, K, D]  --Linear embed-->
    [B, K, d_model]  --L x pre-LN transformer blocks (RoPE relative positions, causal)-->
    last-token output [B, d_model]  --policy head / value head

The temporal context comes from the BRIDGE-side frame stacker (`--il_framestack=K`,
a GPU history buffer inside IsaacLabVecEnv), not from SF recurrent state. The model
is therefore stateless:

* PPO minibatching needs no BPTT -- every rollout step is a plain [B, K*D] forward,
  identical between rollout and training (no importance-ratio drift from recurrence).
* Episode-boundary semantics live in exactly one place, the wrapper: DirectRLEnv
  resets done envs *before* computing the step's obs, so the obs arriving with a
  done is the new episode's first frame; the wrapper re-fills that row's whole
  window with it (repeat-fill -- standard frame-stack treatment, sidesteps the
  padded-token problem attention would otherwise face at episode starts).
* Positions are RELATIVE (RoPE on q/k; no absolute embedding table): no parameter
  depends on the window length, so a policy trained at one K evaluates at another
  by just launching with a different --il_framestack (the checkpoint loads as-is).
* Requires `--use_rnn=False` on the CLI (SF's use_rnn default is True, and the
  value cannot be flipped post-parse -- BufferMgr/sampler never see the change).
  forward_core asserts this with a readable message rather than crashing on a
  PackedSequence.

Head semantics deliberately match SF's default model (isolate the trunk effect):
mean = Linear(d_model -> A); log_std = one learned vector (state-independent),
forwarded through SF's get_action_distribution as raw [mu | log_std] logits.
"""

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from torch import Tensor

from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace

LOG_STD_CLAMP = (-10.0, 2.0)  # same band FlashSAC's head uses


class ScalarLogStd(nn.Module):
    """State-independent learned log-std, wrapped in a module.

    Must be a child MODULE, not a bare nn.Parameter on the ActorCritic itself:
    ActorCritic.model_to_device moves children only, so a bare parameter would
    silently stay on CPU while the rest of the model moves to the GPU.

    Initialized from cfg.initial_stddev to match the default model's
    ActionParameterizationContinuousNonAdaptiveStddev semantics.
    """

    def __init__(self, action_dim: int, initial_stddev: float = 1.0):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(initial_stddev)))


def rope_freqs(seq_len: int, head_dim: int, device: torch.device, theta: float = 10000.0) -> Tuple[Tensor, Tensor]:
    """RoPE rotation tables for seq_len relative positions.

    Computed for the ACTUAL sequence length at every forward, which is what makes
    the trunk transferable across window sizes -- there is no absolute position
    table to go stale when K changes between training and evaluation.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [L, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each consecutive feature pair of x [B, H, L, D] per position."""
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]  # broadcast over B, H
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to q and k (v stays unrotated)."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} not divisible by {num_heads} heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0, "RoPE rotates feature pairs, head_dim must be even"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        b, length, _ = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, L, head_dim]

        cos, sin = rope_freqs(length, self.head_dim, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        # causality makes the last row a full-window average of v; read it via last token below
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, length, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block: RoPE attention + GELU MLP, both residual."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTransformerTrunk(nn.Module):
    """Pre-LN causal self-attention over a sliding window of proprioceptive frames.

    Window slot k-1 is the newest frame; the output is the LAST token's hidden
    state (the only position that, by causality, has seen the entire window).
    RoPE positions make the trunk length-agnostic: `window` below is the nominal
    training window, but forward() works for any sequence length the input reshapes into.
    """

    def __init__(self, obs_dim: int, window: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.window = window  # nominal; RoPE means forward is not tied to it
        self.embed = nn.Linear(obs_dim, d_model)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.zeros_(self.embed.bias)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)  # final block-end LN, GPT-style
        for block in self.blocks:
            for lin in block.modules():
                if isinstance(lin, nn.Linear):
                    nn.init.orthogonal_(lin.weight, gain=1.0)
                    nn.init.zeros_(lin.bias)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        length = x.shape[1] // self.obs_dim  # actual window (may differ from nominal at eval)
        tokens = self.embed(x.view(b, length, self.obs_dim))
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden, causal_mask)
        return self.out_norm(hidden[:, -1])


class TransformerActorCritic(ActorCritic):
    """Causal-transformer trunk behind SF's three-stage head/core/tail API.

    Core is the identity (stateless; see module doc) -- history is already inside
    the observation via the bridge frame stacker.
    """

    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(obs_space, action_space, cfg)

        if isinstance(obs_space, gym.spaces.Dict):
            assert list(obs_space.spaces.keys()) == ["obs"], f"unexpected obs keys {obs_space.spaces}"
            obs_space = obs_space["obs"]
        obs_flat = obs_space.shape[0]
        window = max(1, getattr(cfg, "il_framestack", 1))
        assert obs_flat % window == 0, f"stacked obs dim {obs_flat} not divisible by window {window}"
        self.obs_dim = obs_flat // window
        self.window = window
        action_dim = action_space.shape[0]

        self.trunk = CausalTransformerTrunk(
            self.obs_dim,
            window,
            getattr(cfg, "il_tr_d_model", 256),
            getattr(cfg, "il_tr_layers", 2),
            getattr(cfg, "il_tr_heads", 4),
        )
        self.policy_mean = nn.Linear(getattr(cfg, "il_tr_d_model", 256), action_dim)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)  # near-deterministic start
        nn.init.zeros_(self.policy_mean.bias)
        self.actor_log_std = ScalarLogStd(action_dim, getattr(cfg, "initial_stddev", 1.0))
        self.value_head = nn.Linear(getattr(cfg, "il_tr_d_model", 256), 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.5)

        # kept for SF plumbing introspection; the core never uses it
        self.rnn_size = cfg.rnn_size

    # ---- SF three-stage API (mirrors flashsac_model; core = identity) ----

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        return self.trunk(normalized_obs_dict["obs"])

    def forward_core(self, head_output, rnn_states):
        if not torch.is_tensor(head_output):
            raise RuntimeError(
                "TransformerActorCritic got sequence (PackedSequence) input -- launch with --use_rnn=False "
                "(the model is stateless: history arrives via --il_framestack, not recurrent state)"
            )
        return head_output, rnn_states

    def _action_logits(self, core_output: Tensor) -> Tensor:
        mean = self.policy_mean(core_output)
        log_std = self.actor_log_std.log_std.clamp(*LOG_STD_CLAMP).expand_as(mean)
        return torch.cat([mean, log_std], dim=1)

    def forward_tail(
        self,
        core_output: Tensor,
        values_only: bool = False,
        sample_actions: bool = True,
        action_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        if values_only:
            return TensorDict(values=self.value_head(core_output).squeeze(-1))

        action_logits = self._action_logits(core_output)
        self.last_action_distribution = get_action_distribution(self.action_space, raw_logits=action_logits)

        result = TensorDict()
        result["values"] = self.value_head(core_output).squeeze(-1)
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
        result = self.forward_tail(
            core_output, values_only=values_only, sample_actions=sample_actions, action_mask=action_mask
        )
        result["new_rnn_states"] = new_rnn_states
        return result

    # the base class introspects encoders[0] for these; this model has no SF encoders
    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        from sample_factory.model.model_utils import model_device

        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


def make_transformer_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    return TransformerActorCritic(obs_space, action_space, cfg)


def register_transformer_model() -> None:
    from sample_factory.algo.utils.context import global_model_factory

    global_model_factory().register_actor_critic_factory(make_transformer_actor_critic)
    print(
        "[transformer-model] registered: frame-stack obs -> causal pre-LN transformer (RoPE) -> last token -> heads",
        flush=True,
    )
