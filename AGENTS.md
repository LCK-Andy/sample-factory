# AGENTS.md — Sample Factory fork (Isaac Lab bridge work)

Fork of `alex-petrenko/sample-factory` (upstream remote: `origin`; user's fork: `fork`). Sample Factory is a
high-throughput RL library (APPO/PPO). All local work on top of upstream base commit `1dc7f635` is the
**Isaac Lab integration**: a bridge that runs SF-APPO on Isaac Lab's GPU-vectorized envs, plus custom policy
architectures. Read `ISAACLAB_SF_BENCHMARKS.md` before touching the bridge or comparing against rsl_rl —
it records the integration traps and the benchmark/recipe findings.

## Layout

- `sample_factory/` — core library: `algo/` (sampling, learning, runners), `cfg/` (arguments/config), `envs/`,
  `model/`, `utils/`. Avoid gratuitous edits here; the intentional upstream diffs are listed below.
- `sf_examples/` — per-domain integrations (vizdoom, atari, dmlab, mujoco, envpool, brax, nethack,
  isaacgym_examples).
- `sf_examples/isaac_lab_examples/` — **the active custom area**:
  - `train_isaaclab.py` — the bridge (`IsaacLabVecEnv`, env factory, `--il_*` flags, entry point)
  - `enjoy_isaaclab.py` — playback (GUI via `IL_RENDER=1`)
  - `eval_sweep_isaaclab.py` — deterministic eval sweep (per-episode mean±std over best+latest ckpts; runs enjoy as subprocesses because the bridge env is a per-process singleton)
  - `flashsac_model.py` / `transformer_model.py` — custom actor-critics registered via
    `global_model_factory().register_actor_critic_factory(...)`; the transformer trunk uses RoPE (no parameter depends on the window K, so policy weights transfer across `--il_framestack` values; only the obs normalizer stats are K-shaped)
- `tests/`, `docs/` (mkdocs, served via `make docs-serve`), `train_dir/` (experiment outputs, not tracked).

## Commands

- Dev env: editable install into `~/env_isaaclab` — run things with `~/env_isaaclab/bin/python`
  (`run_allegro_compare.sh` shows the pattern). CI tests Python 3.9–3.11; setup.py pins `gymnasium>=0.27,<2.0`.
- Format: `make format`; style check: `make check-codestyle` (black + isort, line length 120, py38 target;
  flake8 ignores E501,F401,E203,W503,E126,E722,E704).
- Tests: `make test` (= `pytest -s --maxfail=2 -rA`). Full suite pulls in optional env deps (VizDoom, onnxruntime, etc.);
  prefer a focused subset, e.g. `pytest -s tests/test_transformer_model.py -x` (the stateless-transformer policy is
  fully unit-tested without Isaac Lab) or `tests/examples/`. The Isaac Lab bridge itself has no unit tests — smoke it
  with a short training run (`--train_for_env_steps` small).
- Architecture comparisons: `./run_allegro_compare.sh` (stage-a LR screen → stage-b 10M×3 seeds → stage-c deterministic
  eval sweep → k-sweep). Do NOT trust best-checkpoint reward comparisons across arms run before 2026-09-16: the bridge's
  Ant-recipe pin leaked lr=3e-4 onto non-Ant default-model arms while custom models fell through to SF's 1e-4 default.
  The pin is now scoped Ant-only; sizing/LR must be explicit on each arm's command line.
- Training: `python -m sf_examples.isaac_lab_examples.train_isaaclab --algo=APPO --env=Isaac-Ant-Direct-v0
  --experiment=<name> --batched_sampling=True --il_physics=newton_mjwarp --il_num_envs=2048 --device=gpu
  --env_gpu_actions=True --env_gpu_observations=True` (+ recipe flags in the benchmarks doc).

## SF config-system gotchas (the ones that bite)

- **Structural settings** (num_workers, rollout/batch sizes) must be pinned via `parser.set_defaults(...)` in
  `add_extra_params_func` BEFORE `parse_full_cfg`. Mutating `cfg` after parsing does NOT propagate to
  BufferMgr/sampler — such changes are silently lost.
- **SF reloads `train_dir/<experiment>/config.json`** for an existing experiment name and overrides your CLI
  flags. Delete the experiment dir (or rename) when changing config; re-runs of custom-arch experiments must
  have the model factory registered before `run_rl`/`enjoy` reload the saved config (the bridge's `parse_args`
  calls `maybe_load_from_checkpoint` + `custom_env_override_defaults` in the right order — keep it that way).
- SF boolean CLI flags need `--flag=True` / `--flag=False` syntax. SF device names (`gpu`) must be mapped to
  Isaac Lab's (`cuda:0`).
- Batched mode contract: one SF "env" object IS the whole GPU-vectorized sim → `num_workers=1`,
  `num_envs_per_worker=1`; `batch_size/rollout` trajectories must divide the agent count
  (4096/32=128 divides 512).
- The bridge env is a **per-process singleton** (`_ENV_SINGLETON`): SF re-fires `RolloutWorker.init` per
  inference worker, but Isaac Lab allows one sim context per process.
- Physics backends: `--il_physics=newton_mjwarp` (default, kit-less) vs `physx` (needs app-first Kit boot —
  `import isaacsim` before any task import that touches pxr). Env-var knobs: `IL_RENDER=1` (GUI),
  `IL_ADR_DIFFICULTY=N` (freeze ADR curriculum for comparable evals), `IL_CPU_THREADS` (cap Kit tasking pool
  to the cgroup quota).

## Intentional upstream diffs (do not revert)

- `setup.py`: gymnasium pin lifted `<1.0` → `>=0.27,<2.0` (Isaac Lab 3.0 needs ≥1.1.1). Upstreamable.
- `sample_factory/algo/utils/make_env.py`: `BatchedListToDictWrapper.__init__` propagates
  `num_agents`/`is_multiagent` (upstream bug: GPU-vectorized envs collapsed to 1 agent → buffer-shape crash).
- `sample_factory/cfg/arguments.py`: `load_from_checkpoint` uses `cfg.items()` for AttrDict (a second reload
  dropped every non-JSON key).
- `sample_factory/algo/utils/env_info.py`: env-info logging bumped to INFO.
- Fork-local SF additions (2026-09-16): `--lr_warmup_updates` (`WarmupScheduler` in `learner.py` wraps any
  `lr_schedule`, linear ramp then hands over; seeded so the FIRST optimizer step is warmed; fast-forwarded on
  checkpoint resume), `--use_amp` (bf16 autocast around the learner's loss forward only — backward stays
  outside; rollout/inference remains fp32), `--verbose` evaluation flag (per-episode lines parsed by the eval
  sweep harness), and the transformer-model `ScalarLogStd` honors `--initial_stddev`.
- `sf_examples/isaac_lab_examples/` itself.

## Conventions

- Commits follow conventional-commit style with scopes: `feat(bridge): ...`, `fix(cfg): ...`, `docs: ...`.
- The codebase comments heavily at constraint sites (why-not-the-obvious-thing notes in the bridge and model
  files are load-bearing — read them before refactoring those areas).
