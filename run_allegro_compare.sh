#!/bin/bash
# Small-scale comparison: transformer vs MLP vs GRU on Isaac-Repose-Cube-Allegro-Direct-v0
# All arms share identical SF recipe flags; only the architecture differs.
set -u
cd ~/Documents/projects/sample-factory
PY=~/env_isaaclab/bin/python
COMMON="--algo=APPO --env=Isaac-Repose-Cube-Allegro-Direct-v0 --batched_sampling=True \
  --il_physics=newton_mjwarp --il_num_envs=512 --device=gpu --env_gpu_actions=True \
  --env_gpu_observations=True --serial_mode=True --value_bootstrap=True --reward_scale=0.01 \
  --train_for_env_steps=10000000 --seed=0 --save_best_after=100000 \
  --train_dir=$HOME/Documents/projects/sample-factory/train_dir"

run () {  # $1=name $2=arch flags
  local name=$1; shift
  rm -rf "$HOME/Documents/projects/sample-factory/train_dir/$name"
  echo "=== $name: $* ==="
  $PY -m sf_examples.isaac_lab_examples.train_isaaclab $COMMON --experiment="$name" "$@" \
    > "/tmp/train_${name}.log" 2>&1
  echo "=== $name EXIT=$? ==="
}

export PYTHONPATH=/home/andy/Documents/projects/sample-factory
run allegro-mlp-10m  --il_model=default --use_rnn=False
run allegro-gru-10m  --il_model=default --use_rnn=True
run allegro-tr-10m   --il_model=transformer --use_rnn=False --il_framestack=16
echo "ALL RUNS DONE"
