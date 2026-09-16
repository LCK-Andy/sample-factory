#!/bin/bash
# Clean architecture comparison on Isaac-Repose-Cube-Allegro-Direct-v0 (v2).
#
# v1 lesson (2026-09-16): the bridge's Ant-recipe pin leaked lr=3e-4 onto the
# MLP/GRU arms while the transformer fell through to SF's default 1e-4, so v1's
# "identical flags" was false in effect. v2 pins nothing implicitly: every arm's
# LR and sizing is on its own command line, all arms share warmup + bf16 AMP,
# and the decisive numbers come from a deterministic eval sweep, not from
# best-checkpoint filenames (a max over noisy training rewards).
#
# Usage:
#   ./run_allegro_compare.sh stage-a              # 2M screen: {mlp,gru,tr} x {1e-4,3e-4}
#   ./run_allegro_compare.sh stage-b tr:1e-4 ...  # 10M x 3 seeds at per-arch best LR (read off stage-a)
#   ./run_allegro_compare.sh stage-c exp1 exp2... # deterministic eval sweep (best+latest, mean±std)
#   ./run_allegro_compare.sh k-sweep tr:1e-4      # transformer K in {8,16,32} at 2M (RoPE trunk)
set -u
cd ~/Documents/projects/sample-factory
PY=~/env_isaaclab/bin/python
TRAIN_DIR=$HOME/Documents/projects/sample-factory/train_dir
ENV=Isaac-Repose-Cube-Allegro-Direct-v0

# Flags shared by EVERY arm. --nonlinearity=elu + [512,256,128] match the rsl_rl
# Allegro baseline sizing for the default-model arms; the transformer carries
# its own trunk sizing (sizing is part of an architecture).
COMMON="--algo=APPO --env=$ENV --batched_sampling=True \
  --il_physics=newton_mjwarp --il_num_envs=512 --device=gpu --env_gpu_actions=True \
  --env_gpu_observations=True --serial_mode=True --value_bootstrap=True --reward_scale=0.01 \
  --nonlinearity=elu --use_amp=bf16 --lr_warmup_updates=200 --seed=0 \
  --train_dir=$TRAIN_DIR"

arch_flags () {  # $1 = arch
  case "$1" in
    mlp) echo "--il_model=default --use_rnn=False --encoder_mlp_layers 512 256 128" ;;
    gru) echo "--il_model=default --use_rnn=True  --encoder_mlp_layers 512 256 128" ;;
    tr)  echo "--il_model=transformer --use_rnn=False --il_framestack=16" ;;
    *) echo "unknown arch $1" >&2; exit 2 ;;
  esac
}

run () {  # $1=name, rest = flags
  local name=$1; shift
  rm -rf "$TRAIN_DIR/$name"
  echo "=== $name: $* ==="
  $PY -m sf_examples.isaac_lab_examples.train_isaaclab $COMMON --experiment="$name" "$@" \
    > "/tmp/train_${name}.log" 2>&1
  echo "=== $name EXIT=$? ==="
}

best_rewards () {  # $1 = experiment name: print best-checkpoint reward from filename
  ls "$TRAIN_DIR/$1/checkpoint_p0/" 2>/dev/null | grep -oP 'best_.*_reward_\K[-0-9.]+' | tail -1
}

stage_a () {  # 2M screen: pick each arch's best LR
  local arch lr name
  for arch in mlp gru tr; do
    for lr in 1e-4 3e-4; do
      name="v2-$arch-2m-lr${lr}"
      run "$name" --learning_rate=$lr --train_for_env_steps=2000000 $(arch_flags "$arch")
      echo "--- $name best reward: $(best_rewards "$name") ---"
    done
  done
  echo "STAGE A DONE -- read the six best-reward lines above, then:"
  echo "  ./run_allegro_compare.sh stage-b mlp:<lr> gru:<lr> tr:<lr>"
}

stage_b () {  # 10M x 3 seeds at per-arch best LR: ./run_allegro_compare.sh stage-b tr:1e-4 mlp:3e-4
  local spec arch lr seed name
  for spec in "$@"; do
    arch="${spec%%:*}"; lr="${spec##*:}"
    for seed in 0 1 2; do
      name="v2-$arch-10m-lr${lr}-s$seed"
      run "$name" --learning_rate=$lr --train_for_env_steps=10000000 --seed=$seed $(arch_flags "$arch")
      echo "--- $name best reward: $(best_rewards "$name") ---"
    done
  done
  echo "STAGE B DONE -- deterministic eval:"
  echo "  ./run_allegro_compare.sh stage-c <experiment names from above>"
}

stage_c () {  # deterministic eval sweep: ./run_allegro_compare.sh stage-c v2-tr-10m-lr1e-4-s0 ...
  $PY -m sf_examples.isaac_lab_examples.eval_sweep_isaaclab \
    --experiments="$(IFS=,; echo "$*")" --env=$ENV --il_num_envs=64 --max_num_frames=25000
}

k_sweep () {  # transformer context length at 2M: ./run_allegro_compare.sh k-sweep tr:1e-4
  local spec lr k
  for spec in "$@"; do
    lr="${spec##*:}"
    for k in 8 16 32; do
      name="v2-tr-k${k}-2m-lr${lr}"
      run "$name" --learning_rate=$lr --train_for_env_steps=2000000 --seed=0 \
        --il_model=transformer --use_rnn=False --il_framestack=$k
      echo "--- $name best reward: $(best_rewards "$name") ---"
    done
  done
  echo "K-SWEEP DONE -- if return still climbs at K=32, Direction B (native sequence core) has a case."
}

case "${1:-}" in
  stage-a) shift; stage_a "$@" ;;
  stage-b) shift; stage_b "$@" ;;
  stage-c) shift; stage_c "$@" ;;
  k-sweep) shift; k_sweep "$@" ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20; exit 1 ;;
esac
