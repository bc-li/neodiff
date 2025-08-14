#!/bin/bash

# Configuration parameters
SCHEDULER=${1:-"poisson"}                    # Scheduling method: "poisson" or "det" (deterministic)
DEVICE=${2:-"7"}                              # GPU device to use
EXPERIMENT_NAME=${3:-"default"}             # Experiment name suffix
GPUS=${4:-"1"}                              # Number of GPUs to use

# Model hyperparameters
TRANS_SCHEDULE="constant_same"              # Transition schedule
NUM_TAU=100                                 # Number of tau values for time predictor
TAU_TYPE="normal"                          # Tau distribution type
TAU_LAYERS=1                               # Number of tau predictor layers

# Noise configuration
NOISE_SCHEDULE="sqrt"                      # Noise schedule: "linear", "sqrt", etc.
NOISE_FACTOR=6.5                          # Noise scaling factor

# Training hyperparameters
LR="5e-4"                                  # Learning rate
MAX_TOKENS=8000                            # Maximum tokens per batch
MAX_UPDATE=300000                          # Maximum number of updates
DROPOUT=0.2                                # Dropout rate

# Dataset configuration
DATASET="iwslt14_de_en"

# Model naming based on configuration (simplified naming like difformer script)
if [ "$SCHEDULER" = "det" ]; then
    # Deterministic scheduler configuration
    MODEL_NAME="neodiff_${SCHEDULER}_tau${NUM_TAU}_trans${TRANS_SCHEDULE}_noise${NOISE_SCHEDULE}_f${NOISE_FACTOR}_lr${LR}_bsz${MAX_TOKENS}${TAU_TYPE:+_$TAU_TYPE}_drop${DROPOUT}_taulayer${TAU_LAYERS}_${EXPERIMENT_NAME}"
else
    # Poisson (non-simultaneous) scheduler configuration  
    MODEL_NAME="neodiff_${SCHEDULER}_tau${NUM_TAU}_trans${TRANS_SCHEDULE}_noise${NOISE_SCHEDULE}_f${NOISE_FACTOR}_lr${LR}_bsz${MAX_TOKENS}${TAU_TYPE:+_$TAU_TYPE}_drop${DROPOUT}_taulayer${TAU_LAYERS}_taulsanchor_taumse_${EXPERIMENT_NAME}_GPUS${GPUS}"
fi

MODEL_DIR="models/${DATASET}/${MODEL_NAME}"

echo "Training NeoDiff model: ${MODEL_NAME}"
echo "Using device: ${DEVICE}"
echo "Model directory: ${MODEL_DIR}"

# Create directories
mkdir -p $MODEL_DIR/tb
mkdir -p $MODEL_DIR/logs

# Training command
CUDA_VISIBLE_DEVICES=$DEVICE fairseq-train \
    data-bin/${DATASET} \
    --save-dir $MODEL_DIR/ckpts \
    --ddp-backend no_c10d \
    --user-dir neodiff \
    --task neodiff \
    --criterion nat_loss \
    --arch neodiff_iwslt_de_en \
    --share-all-embeddings \
    --scheduler $SCHEDULER \
    --num-tau $NUM_TAU \
    --tau-predictor-layers $TAU_LAYERS \
    --tau-loss-anchor \
    --tau-mse \
    --mse-loss-weight none \
    --anchor-loss-weight none \
    ${TAU_TYPE:+--tau-type $TAU_TYPE} \
    --trans-schedule $TRANS_SCHEDULE \
    --noise-schedule $NOISE_SCHEDULE \
    --forward-coeff-type "sqrt" \
    --noise-factor $NOISE_FACTOR \
    --self-cond \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr $LR --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --clip-norm 1.0 \
    --dropout ${DROPOUT} --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'json' --log-interval 100 \
    --tensorboard-logdir $MODEL_DIR/tb \
    --fixed-validation-seed 7 \
    --decoding-steps 20 \
    --eval-bleu \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --validate-interval 5 \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --max-tokens $MAX_TOKENS \
    --max-update $MAX_UPDATE \
    --keep-last-epochs 10 \
    --keep-best-checkpoints 5 \
    --num-workers 20 \
    --store-ema \
    --ema-fp32 \
    2>&1 | tee $MODEL_DIR/train.log

echo "Training completed. Logs saved to: $MODEL_DIR/train.log"

# Automatically run evaluation after training
echo "Starting evaluation..."
bash scripts/iwslt14_de_en/evaluate_neodiff.sh $MODEL_NAME 10 normal $DEVICE 