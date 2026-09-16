# Sample Factory × Isaac Lab — Integration & Benchmark Record

**Date:** 2026-09-14 · **Machine:** laptop (RTX 5070 Ti Laptop, i9 ~5GHz) · **Task:** `Isaac-Ant-Direct-v0` · **Backend:** Newton kit-less (`newton_mjwarp`), unless noted

Sample Factory 2.1.1 bridged to Isaac Lab 3.0 (source clone `~/Documents/projects/sample-factory`, editable into `env_isaaclab`). Entry point:

```bash
python -m sf_examples.isaac_lab_examples.train_isaaclab \
  --algo=APPO --env=Isaac-Ant-Direct-v0 --experiment=<name> \
  --batched_sampling=True --il_physics=newton_mjwarp --il_num_envs=2048 \
  --device=gpu --env_gpu_actions=True --env_gpu_observations=True \
  <official-recipe flags below>
```

## Headline finding (the reason this doc exists)

**At 2048 envs / 20M steps, SF-APPO with the official IsaacGym recipe BEATS rsl_rl PPO on final return (+10%) despite 2.2× lower throughput — while rsl_rl wins at 512 envs. The two engines have opposite sweet spots in (num_envs × budget), and the LEAP tasks (4096+ envs, large budgets) land squarely in SF's.**

## Benchmark table (all 20M env steps unless noted)

| Config | rsl_rl final reward | SF final reward | rsl_rl steps/s | SF steps/s |
|---|---|---|---|---|
| 512 envs, 2M steps | **3,502** | 1,119 (default hp) | 113,388 | 34,221 |
| 512 envs, 2M steps | 3,502 | 1,280 (recipe) | 113,388 | 89,953 |
| 512 envs, 20M | **8,145** | 7,525 | 112,988 | 132,523 |
| **2048 envs, 20M** | 8,988 | **9,887** (best 9,970) | **343,860** | 157,281 |

## Curves (mean reward vs env steps)

**512 envs, 20M:**

| steps | rsl_rl | SF recipe |
|---|---|---|
| 2M | 2,258 | 552 |
| 5M | 6,063 | 3,250 |
| 10M | 7,510 | 5,603 |
| 20M | 8,145 | 7,525 |

**2048 envs, 20M (crossover at 5M):**

| steps | rsl_rl | SF recipe |
|---|---|---|
| 2M | 927 | 82 |
| 5M | 2,481 | **3,102** ← SF takes the lead |
| 10M | 4,896 | 7,039 |
| 15M | 7,666 | 8,410 |
| 20M | 8,988 | **9,887** |

## Why the crossover happens

- SF's official recipe uses **batch 32,768**: at 512 envs, 20M steps = only ~305 full-quality updates (slow warmup, 2M-budget scores are meaningless — `save_best_after=5M` in the recipe says as much). At 2048 envs every batch is saturated immediately; SF overtakes by 5M steps.
- rsl_rl updates on 12k-sample batches ~4× more often — better at small budgets/env counts, slightly update-starved at 2048 (407 iterations for 20M).
- Throughput: rsl_rl amortizes better with env count (113k→344k steps/s, 3.0×) because its step is dominated by env physics + small nets; SF's learner time (32k-sample PPO epochs ×4) doesn't amortize (133k→157k, 1.2×).
- Recipe ingredients that mattered: input+return normalization, value_bootstrap, KL-adaptive LR, ELU, reward_scale 0.01, serial_mode + `env_gpu_actions/observations` (34k→90k steps/s alone at 512).

## Engine selection rule (this stack, this hardware class)

| Regime | Pick |
|---|---|
| ≤5M steps or ≤512 envs | rsl_rl |
| ≥2048 envs, large budget, care about final return | **SF-APPO** |
| Same, care about wall-clock | rsl_rl (2.2× throughput) |
| LEAP-style tasks (4096+ envs, 100M+ budgets) | SF's sweet spot — candidate engine switch |

## Integration notes (what it took; commits in the clone)

1. `setup.py`: gymnasium pin lifted `<1.0` → `<2.0` (Isaac Lab 3.0 needs ≥1.1.1; SF API is 1.x-compatible). **Upstreamable.**
2. `make_env.py`: **SF bug** — `BatchedListToDictWrapper` dropped `num_agents`/`is_multiagent`, collapsing GPU-vectorized envs (512 agents) to 1 → buffer-shape crash. Fixed by propagating through the wrapper. **Upstreamable.**
3. Bridge `sf_examples/isaac_lab_examples/train_isaaclab.py`: app-first Kit boot for PhysX (`--il_physics=physx`), per-process env singleton (SF re-fires `RolloutWorker.init`), numpy→torch action conversion, `--il_physics/--il_num_envs` flags.
4. Config plumbing traps: post-parse `cfg` mutation does NOT reach BufferMgr/sampler — structural pins must go through `parser.set_defaults` BEFORE `parse_full_cfg`; and SF **reloads `train_dir/<experiment>/config.json`** for existing experiment names, silently overriding new flags (delete the dir or rename the experiment when changing config).
5. SF booleans need `--flag=True` syntax; SF device names (`gpu`) must map to Isaac Lab's (`cuda:0`).

## PhysX backend (also validated)

`--il_physics=physx` boots a headless Isaac Sim per SF process (app-first order). 512 envs: **6.1k steps/s** vs Newton's 8.7k (default hp smoke). Same numeric recipe not re-tuned for PhysX.

## Artifacts

- SF runs: `~/Documents/projects/sample-factory/train_dir/il-ant-2m{,-tuned}/`, `il-ant-20m-tuned/`, `il-ant-20m-2048/` (best ckpt `checkpoint_p0/best_*.pt`; TB in `.summary/0`)
- rsl_rl runs: `~/IsaacLab/logs/rsl_rl/ant_direct/2026-09-14_0*` (playable `model_*.pt`)
- Clone commits: `2060b57` (integration + 2 core fixes), `060f5cc` (PhysX), `1f4d488` (set_defaults + config-cache note)

---

# Architecture comparison v2 — clean re-run (2026-09-16)

**Task:** `Isaac-Repose-Cube-Allegro-Direct-v0` · 512 envs · Newton · 10M steps · 3 seeds · lr 3e-4 (all arms, Stage-A-screened) · warmup 200 updates · bf16 AMP (learner) · ELU + [512,256,128] on default-model arms · TR = RoPE causal transformer, K=16.

## Why v2 exists: the v1 architecture comparison was confounded

The bridge's Ant-recipe pin (`lr=3e-4`, encoder `[400,200,100]`) leaked onto **all** non-Dexsuite default-model envs, while custom-model arms (`return` early) fell through to SF's default `1e-4`. So v1's MLP/GRU arms (best 2.14/2.29) ran 3× the LR of the transformer arm (best 6.64) despite the script's "identical flags" claim. The pin is now scoped Ant-only (`custom_env_override_defaults`); v2 pins every arm's LR/sizing on its own command line (`run_allegro_compare.sh`).

## Headline results

**Training best-checkpoint reward (scaled units, max over episodes — noisy):**

| arch | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| MLP | 1.32 | 1.75 | 1.04 | **1.37** |
| GRU | 1.52 | 1.56 | 0.08 | **1.05** |
| **Transformer (RoPE, K=16)** | **9.22** | **9.10** | **8.98** | **9.10** |

**Deterministic eval (per-episode reward, raw env scale, mean ± std over ≥130k episodes, best ckpt):**

| arch | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| MLP | 11.96 ± 57.6 | −0.04 ± 46.5 | 16.59 ± 63.8 | **~9.5** (huge seed spread) |
| GRU | −1.13 ± 36.1 | −3.67 ± 26.9 | −1.03 ± 38.8 | **~−1.9** |
| **Transformer** | **18.66 ± 63.3** | **24.67 ± 70.3** | **21.34 ± 67.0** | **~21.6** |

best vs latest checkpoints agree everywhere (endpoints are stable). **The transformer wins decisively and consistently: ~2× MLP, GRU never leaves negative territory — and its seed variance is the smallest of the three.** v1's "transformer wins" finding replicates *stronger* at matched LR.

Throughput (10M, s0): MLP 22.6k · GRU 19.1k · TR 14.6k steps/s — TR pays ~24% vs MLP, keeps 1.5× headroom over the v1 10.9k (RoPE trunk + bf16).

## Findings beyond the ranking

1. **Warmup unlocks 3e-4 for the transformer** (Stage A, 2M): TR at 1e-4 → −0.63; at 3e-4 → 2.53. All three archs selected 3e-4, so the decisive runs are uniform-LR — seeds carry the statistics.
2. **Best-checkpoint reward mis-ranks arms.** GRU's training peaks (1.5/1.6/0.08) suggested 2nd place; deterministic eval puts GRU last (~−2 everywhere). Its rare success episodes don't transfer to eval. The eval sweep is the headline metric from now on.
3. **K-sweep (2M, best-reward): K=8 → 0.66, K=16 → 1.91, K=32 → 0.64.** Non-monotone with a peak at 16: no evidence longer context helps on this task family. **Direction B (native in-SF sequence transformer core) shelved** — the stateless frame-stack + RoPE design is the right endpoint here.
4. **Caveats:** eval episode counts differ ~5× across arms (GRU 767k vs TR 134k in the same frame budget) — GRU's episodes are much shorter, which flatters its per-episode sums slightly; and same-seed 2M reruns vary ±0.6 best-reward (v2-tr-2m 2.53 vs k-sweep k16 1.91, identical config), so single 2M numbers are screens, not results.

## Reproduce

```bash
./run_allegro_compare.sh stage-a                          # LR screen (6 × 2M)
./run_allegro_compare.sh stage-b mlp:3e-4 gru:3e-4 tr:3e-4  # 10M × 3 seeds
./run_allegro_compare.sh stage-c <experiment names>       # deterministic eval sweep
./run_allegro_compare.sh k-sweep tr:3e-4                  # K ∈ {8,16,32}
```

New SF flags this round: `--lr_warmup_updates`, `--use_amp=bf16` (learner-only), `--verbose` (per-episode eval lines). Transformer trunk now RoPE (policy weights transfer across K; only obs-normalizer stats are K-shaped). Unit tests: `tests/test_transformer_model.py`.
