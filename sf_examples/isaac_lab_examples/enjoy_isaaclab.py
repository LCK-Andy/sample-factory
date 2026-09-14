"""Play a trained Sample Factory policy on an Isaac Lab environment.

Mirrors sf_examples/isaacgym_examples/enjoy_isaacgym.py: register the custom envs,
parse evaluation args, and hand off to sample_factory.enjoy.

The saved train_dir config is reloaded and CLI values override it, so a 4096-env
training run can be played with fewer envs:

    IL_RENDER=1 python -m sf_examples.isaac_lab_examples.enjoy_isaaclab \
        --algo=APPO --env=Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
        --experiment=dexlift-sf-4096-1b --train_dir=~/Documents/projects/sample-factory/train_dir \
        --batched_sampling=True --il_physics=physx --il_num_envs=4 \
        --load_checkpoint_kind=best --max_num_frames=100000000

IL_RENDER=1 boots the Kit GUI (default is headless). Checkpoint selection follows
SF's --load_checkpoint_kind: best (best_*.pth) or latest (checkpoint_*.pth).
"""

import sys

from sample_factory.enjoy import enjoy

from sf_examples.isaac_lab_examples.train_isaaclab import parse_args, register_isaaclab_envs


def main():
    """Script entry point."""
    register_isaaclab_envs()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
