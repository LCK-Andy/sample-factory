"""Deterministic eval sweep over Isaac Lab experiment checkpoints.

For each experiment name, runs enjoy_isaaclab as a SUBPROCESS once per checkpoint
kind (best + latest) with --eval_deterministic=True --verbose=True, parses the
per-episode reward lines, and reports mean +/- std over all finished episodes.

Subprocess per evaluation is deliberate: the bridge serves a per-process env
singleton keyed by (env, pid), so in-process sequential enjoy() calls would
silently reuse the first env (e.g. ignore a different --il_num_envs).

Architecture flags (il_model, il_framestack, use_rnn, ...) come from each
experiment's saved config.json via the bridge's load-order fix, so the sweep
works for mixed-architecture experiment lists unchanged.

Usage:
    python -m sf_examples.isaac_lab_examples.eval_sweep_isaaclab \
        --experiments=allegro-mlp-10m-s0,allegro-tr-10m-s0 \
        --env=Isaac-Repose-Cube-Allegro-Direct-v0 \
        --il_num_envs=64 --max_num_frames=25000
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
from typing import List, Optional

# matches: "Episode finished for agent 12 at 340 frames. Reward: -3.512, true_objective: -351.200"
EPISODE_RE = re.compile(r"Reward: ([-+0-9.eE]+), true_objective: ([-+0-9.eE]+)")
NO_CHECKPOINT = "No checkpoints found"


def run_single_eval(
    experiment: str, env: str, train_dir: str, kind: str, num_envs: int, physics: str, max_num_frames: int
) -> Optional[List[float]]:
    """One enjoy subprocess; returns per-episode rewards or None if no checkpoint."""
    cmd = [
        sys.executable,
        "-m",
        "sf_examples.isaac_lab_examples.enjoy_isaaclab",
        "--algo=APPO",
        f"--env={env}",
        f"--experiment={experiment}",
        f"--train_dir={os.path.expanduser(train_dir)}",
        "--device=gpu",
        "--batched_sampling=True",
        f"--il_physics={physics}",
        f"--il_num_envs={num_envs}",
        f"--load_checkpoint_kind={kind}",
        "--eval_deterministic=True",
        "--verbose=True",
        "--no_render",
        f"--max_num_frames={max_num_frames}",
    ]
    print(f"[eval-sweep] {experiment} [{kind}]: running...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    output = proc.stdout + proc.stderr

    if NO_CHECKPOINT in output:
        print(f"[eval-sweep] {experiment} [{kind}]: NO CHECKPOINT of this kind", flush=True)
        return None

    rewards = [float(m.group(1)) for m in EPISODE_RE.finditer(output)]
    if not rewards:
        print(f"[eval-sweep] {experiment} [{kind}]: no episodes parsed (exit {proc.returncode})", flush=True)
        print(output[-2000:], flush=True)  # tail for diagnosis
        return None
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", required=True, type=str, help="Comma-separated experiment names")
    parser.add_argument("--env", default="Isaac-Repose-Cube-Allegro-Direct-v0", type=str)
    parser.add_argument("--train_dir", default=None, type=str, help="Default: SF default train dir")
    parser.add_argument("--il_num_envs", default=64, type=int, help="Env count for evaluation")
    parser.add_argument("--il_physics", default="newton_mjwarp", type=str, choices=["newton_mjwarp", "physx"])
    parser.add_argument("--max_num_frames", default=25000, type=int, help="Batched frames (x num_envs episodes' worth)")
    parser.add_argument("--kinds", default="best,latest", type=str, help="Checkpoint kinds to evaluate")
    args = parser.parse_args()

    train_dir = args.train_dir
    if train_dir is None:
        train_dir = "~/Documents/projects/sample-factory/train_dir"  # this repo's convention

    rows = []
    header = f"{'experiment':40s} {'kind':7s} {'episodes':>8s} {'mean':>10s} {'std':>10s}"
    print(header, flush=True)
    for experiment in args.experiments.split(","):
        for kind in args.kinds.split(","):
            rewards = run_single_eval(
                experiment, args.env, train_dir, kind, args.il_num_envs, args.il_physics, args.max_num_frames
            )
            if rewards is None:
                rows.append(f"{experiment:40s} {kind:7s} {'-':>8s} {'-':>10s} {'-':>10s}")
            else:
                mean = statistics.mean(rewards)
                std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
                rows.append(f"{experiment:40s} {kind:7s} {len(rewards):8d} {mean:10.3f} {std:10.3f}")
            print(rows[-1], flush=True)

    print("\n===== EVAL SWEEP SUMMARY =====", flush=True)
    print(header, flush=True)
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
