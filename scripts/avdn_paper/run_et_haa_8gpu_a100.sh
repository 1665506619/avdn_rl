#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-./datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-output/et_stop_8gpu}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
SEED="${SEED:-0}"
DEBUG_CUDA="${DEBUG_CUDA:-0}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/launcher_logs}"

# Training schedule
ITERS="${ITERS:-300000}"
LOG_EVERY="${LOG_EVERY:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_ACTION_LEN="${MAX_ACTION_LEN:-15}"
MAX_INSTR_LEN="${MAX_INSTR_LEN:-80}"

# Optimization
LR="${LR:-1e-5}"
OPTIM="${OPTIM:-adamW}"
FEEDBACK="${FEEDBACK:-student}"

# Stop head
STOP_TH="${STOP_TH:-0.5}"
STOP_IOU_TH="${STOP_IOU_TH:-0.5}"
STOP_LOSS_W="${STOP_LOSS_W:-1.0}"

# PPO stage
USE_PPO="${USE_PPO:-0}"
RL_LR="${RL_LR:-5e-6}"
PPO_ITERS="${PPO_ITERS:-200}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
PPO_CLIP="${PPO_CLIP:-0.2}"
PPO_GAMMA="${PPO_GAMMA:-0.99}"
PPO_LAM="${PPO_LAM:-0.95}"
PPO_VALUE_W="${PPO_VALUE_W:-0.5}"
PPO_ENTROPY_W="${PPO_ENTROPY_W:-0.01}"
PPO_IL_W="${PPO_IL_W:-0.05}"
PPO_BATCH_EPISODES="${PPO_BATCH_EPISODES:-8}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-32}"
PPO_ACTION_STD_INIT="${PPO_ACTION_STD_INIT:-0.2}"

# Reward shaping
STEP_PENALTY="${STEP_PENALTY:--0.01}"
DELTA_IOU_REWARD="${DELTA_IOU_REWARD:-1.0}"
SUCCESS_STOP_REWARD="${SUCCESS_STOP_REWARD:-2.0}"
FALSE_STOP_PENALTY="${FALSE_STOP_PENALTY:--1.0}"
OVERSHOOT_PENALTY="${OVERSHOOT_PENALTY:--0.5}"
MAX_STEP_PENALTY="${MAX_STEP_PENALTY:--1.0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_DIR}"

DARKNET_CFG="${DARKNET_CFG:-${ROOT_DIR}/AVDN/pretrain_weights/yolo_v3.cfg}"
DARKNET_WEIGHT="${DARKNET_WEIGHT:-${ROOT_DIR}/AVDN/pretrain_weights/best.pt}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

if [[ "${DEBUG_CUDA}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_DEBUG=WARN
  echo "Debug mode enabled: CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=WARN"
fi

ARGS=(
  --root_dir "${ROOT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --world_size "${GPUS}"
  --seed "${SEED}"
  --feedback "${FEEDBACK}"
  --max_action_len "${MAX_ACTION_LEN}"
  --max_instr_len "${MAX_INSTR_LEN}"
  --lr "${LR}"
  --rl_lr "${RL_LR}"
  --iters "${ITERS}"
  --log_every "${LOG_EVERY}"
  --batch_size "${BATCH_SIZE}"
  --optim "${OPTIM}"
  --stop_th "${STOP_TH}"
  --stop_iou_th "${STOP_IOU_TH}"
  --stop_loss_w "${STOP_LOSS_W}"
  --darknet_model_file "${DARKNET_CFG}"
  --darknet_weight_file "${DARKNET_WEIGHT}"
)

if [[ "${USE_PPO}" == "1" ]]; then
  ARGS+=(
    --use_ppo
    --ppo_iters "${PPO_ITERS}"
    --ppo_epochs "${PPO_EPOCHS}"
    --ppo_clip "${PPO_CLIP}"
    --ppo_gamma "${PPO_GAMMA}"
    --ppo_lam "${PPO_LAM}"
    --ppo_value_w "${PPO_VALUE_W}"
    --ppo_entropy_w "${PPO_ENTROPY_W}"
    --ppo_il_w "${PPO_IL_W}"
    --ppo_batch_episodes "${PPO_BATCH_EPISODES}"
    --ppo_minibatch_size "${PPO_MINIBATCH_SIZE}"
    --ppo_action_std_init "${PPO_ACTION_STD_INIT}"
    --step_penalty "${STEP_PENALTY}"
    --delta_iou_reward "${DELTA_IOU_REWARD}"
    --success_stop_reward "${SUCCESS_STOP_REWARD}"
    --false_stop_penalty "${FALSE_STOP_PENALTY}"
    --overshoot_penalty "${OVERSHOOT_PENALTY}"
    --max_step_penalty "${MAX_STEP_PENALTY}"
  )
fi

echo "Launching 8-GPU training from ${REPO_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Root dir: ${ROOT_DIR}"
echo "Use PPO: ${USE_PPO}"
echo "Log file: ${LOG_FILE}"

{
  echo "========== Launch Info =========="
  date
  echo "REPO_DIR=${REPO_DIR}"
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "GPUS=${GPUS}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  echo "USE_PPO=${USE_PPO}"
  echo "DEBUG_CUDA=${DEBUG_CUDA}"
  echo "========== Command =========="
  printf '%q ' torchrun --standalone --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" src/xview_et/main.py "${ARGS[@]}"
  echo
  echo "=============================="

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  PYTHONUNBUFFERED=1 \
  torchrun \
    --standalone \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    src/xview_et/main.py "${ARGS[@]}"
} 2>&1 | tee "${LOG_FILE}"
