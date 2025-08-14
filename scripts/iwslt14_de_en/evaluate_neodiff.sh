#!/bin/bash

MODEL_NAME=${1}
STEPS=${2:-20}                              # Number of decoding steps
METHOD=${3:-"normal"}                       # Decoding method: "normal", etc.
DEVICE=${4:-0}                              # GPU device to use
EARLY_STOPPING=${5:-0}                      # Early stopping flag
CUSTOM_TIMESTEPS=${6:-""}                  # Optional custom timesteps (comma-separated)

if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is required"
    echo "Usage: bash evaluate_neodiff.sh <MODEL_NAME> [STEPS] [METHOD] [DEVICE] [EARLY_STOPPING] [CUSTOM_TIMESTEPS]"
    exit 1
fi

# Evaluation configuration
SCHEDULER="poisson"                         # Use Poisson scheduler for evaluation
NF=1                                       # Noise factor for evaluation
LENGTH_BEAM=9                              # Length beam size
NOISE_BEAM=1                               # Noise beam size

# Select task depending on whether custom timesteps are provided
TASK="neodiff"
if [ -n "$CUSTOM_TIMESTEPS" ]; then
    TASK="neodiff_tuning_global_t"
fi

# Dataset and paths
DATASET="iwslt14_de_en"
MODEL_DIR="models/${DATASET}/${MODEL_NAME}"

# Output configuration
OUTPUT_NAME="evaluate_step${STEPS}_beam${LENGTH_BEAM}x${NOISE_BEAM}_${METHOD}_stop${EARLY_STOPPING}_nf${NF}"
if [ -n "$CUSTOM_TIMESTEPS" ]; then
    OUTPUT_NAME="${OUTPUT_NAME}_custom"
    echo "Using custom timesteps: ${CUSTOM_TIMESTEPS}"
fi
OUTPUT_DIR=$MODEL_DIR/evaluate/$OUTPUT_NAME

echo "Evaluating NeoDiff model: ${MODEL_NAME}"
echo "Decoding steps: ${STEPS}, Method: ${METHOD}, Device: ${DEVICE}"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Average last epoch checkpoints
if [ ! -f $MODEL_DIR/ckpts/checkpoint_last_avg.pt ]; then
    echo "Creating averaged checkpoint from last epochs..."
    python scripts/average_checkpoints.py \
        --inputs $(ls $MODEL_DIR/ckpts/checkpoint[0-9]*) \
        --output $MODEL_DIR/ckpts/checkpoint_last_avg.pt
fi

# Average best checkpoints
if [ ! -f $MODEL_DIR/ckpts/checkpoint_best_avg.pt ]; then
    echo "Creating averaged checkpoint from best checkpoints..."
    python scripts/average_checkpoints.py \
        --inputs $(ls $MODEL_DIR/ckpts/checkpoint.*) \
        --output $MODEL_DIR/ckpts/checkpoint_best_avg.pt
fi

# Evaluation checkpoint list
CKPT_LIST=(best_avg)

# Run evaluation for each checkpoint
for CKPT in ${CKPT_LIST[@]}; do
    echo "Evaluating checkpoint: ${CKPT}"
    
    CUDA_VISIBLE_DEVICES=$DEVICE fairseq-generate \
        data-bin/${DATASET} \
        --gen-subset test \
        --user-dir neodiff \
        --task $TASK \
        --path $MODEL_DIR/ckpts/checkpoint_$CKPT.pt \
        --decoding-steps $STEPS \
        --decoding-early-stopping $EARLY_STOPPING \
        --decoding-scheduler $SCHEDULER \
        --decoding-method $METHOD \
        ${NF:+--decoding-nf $NF} \
        --length-beam-size $LENGTH_BEAM \
        --noise-beam-size $NOISE_BEAM \
        --retain-iter-history \
        --retain-z-0-hat \
        --bleu-mbr \
        --remove-bpe \
        --batch-size 50 \
        --beam 9 \
        --uses-ema \
        ${CUSTOM_TIMESTEPS:+--custom-timesteps $CUSTOM_TIMESTEPS} \
        > $OUTPUT_DIR/$CKPT.txt
    
    # Extract and save BLEU score
    echo $CKPT >> $OUTPUT_DIR/bleu.txt
    tail -n 1 $OUTPUT_DIR/$CKPT.txt >> $OUTPUT_DIR/bleu.txt
    echo >> $OUTPUT_DIR/bleu.txt
done

echo "Evaluation completed. Results:"
cat $OUTPUT_DIR/bleu.txt

# Also generate SacreBLEU results
echo "Generating SacreBLEU results..."
mkdir -p $OUTPUT_DIR/sacrebleu

for CKPT in ${CKPT_LIST[@]}; do
    CUDA_VISIBLE_DEVICES=$DEVICE fairseq-generate \
        data-bin/${DATASET} \
        --gen-subset test \
        --user-dir neodiff \
        --task $TASK \
        --path $MODEL_DIR/ckpts/checkpoint_$CKPT.pt \
        --decoding-steps $STEPS \
        --decoding-early-stopping $EARLY_STOPPING \
        --decoding-scheduler $SCHEDULER \
        --decoding-method $METHOD \
        ${NF:+--decoding-nf $NF} \
        --length-beam-size $LENGTH_BEAM \
        --noise-beam-size $NOISE_BEAM \
        --retain-iter-history \
        --retain-z-0-hat \
        --bleu-mbr \
        --remove-bpe \
        --batch-size 50 \
        --beam 9 \
        --uses-ema \
        ${CUSTOM_TIMESTEPS:+--custom-timesteps $CUSTOM_TIMESTEPS} \
        --results-path $OUTPUT_DIR/sacrebleu
done

# Calculate SacreBLEU scores
echo "Computing SacreBLEU scores..."
for CKPT in ${CKPT_LIST[@]}; do
    echo "Computing SacreBLEU for ${CKPT}..."
    python scripts/iwslt14_de_en/sacre_bleu.py \
        $OUTPUT_DIR/sacrebleu/generate-test.txt \
        en > $OUTPUT_DIR/sacrebleu/${CKPT}_sacrebleu.txt
    
    # Extract and save SacreBLEU score to summary file
    echo $CKPT >> $OUTPUT_DIR/sacrebleu_scores.txt
    cat $OUTPUT_DIR/sacrebleu/${CKPT}_sacrebleu.txt >> $OUTPUT_DIR/sacrebleu_scores.txt
    echo >> $OUTPUT_DIR/sacrebleu_scores.txt
done

echo "SacreBLEU evaluation completed."
echo "SacreBLEU results:"
cat $OUTPUT_DIR/sacrebleu_scores.txt
echo "All evaluation results saved to: ${OUTPUT_DIR}" 