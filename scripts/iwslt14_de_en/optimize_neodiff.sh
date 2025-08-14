#!/bin/bash

# NeoDiff Bayesian Optimization Script for IWSLT14 DE-EN Translation
# This script runs Bayesian optimization to find optimal time schedules and evaluates results


# Function to list available models
list_models() {
    echo "Available NeoDiff models in iwslt14_de_en:"
    echo "========================================"
    if [ -d "models/iwslt14_de_en" ]; then
        ls -1 models/iwslt14_de_en/ | grep -E '^neodiff_' | nl
    else
        echo "No models directory found!"
        exit 1
    fi
}

# Function to validate model exists
validate_model() {
    local model_name=$1
    if [ ! -d "models/iwslt14_de_en/$model_name" ]; then
        echo "Error: Model directory not found: models/iwslt14_de_en/$model_name"
        echo ""
        list_models
        exit 1
    fi
    
    if [ ! -f "models/iwslt14_de_en/$model_name/ckpts/checkpoint_best_avg.pt" ]; then
        echo "Error: best_avg checkpoint not found for model: $model_name"
        echo "Please ensure the model has been trained and evaluated first."
        exit 1
    fi
}

# Parse command line arguments
MODEL_NAME=${1}
TIMESTEPS=${2:-20}                          # Number of timesteps to optimize
DEVICE=${3:-0}                              # GPU device to use
INIT_POINTS=${4:-20}                        # Initial random points for Bayesian optimization
N_ITER=${5:-100}                            # Number of Bayesian optimization iterations
MODE=${6:-"optimize"}                       # Mode: "optimize", "eval", or "list"
BATCH_SIZE=${7:-100}                         # Batch size for evaluation
CUSTOM_TIMESTEPS=${8:-""}                  # Optional custom timesteps (comma-separated)

# Handle different modes
case $MODE in
    "list")
        list_models
        exit 0
        ;;
    "eval")
        # Evaluation only mode with predefined schedule
        if [ -z "$MODEL_NAME" ]; then
            echo "Error: MODEL_NAME is required for evaluation mode"
            echo "Usage: bash optimize_neodiff.sh <MODEL_NAME> [TIMESTEPS] [DEVICE] [INIT_POINTS] [N_ITER] eval [BATCH_SIZE] [CUSTOM_TIMESTEPS]"
            exit 1
        fi
        ;;
    "optimize")
        # Full optimization mode (default)
        if [ -z "$MODEL_NAME" ]; then
            echo "Usage: bash optimize_neodiff.sh <MODEL_NAME> [TIMESTEPS] [DEVICE] [INIT_POINTS] [N_ITER] [MODE] [BATCH_SIZE] [CUSTOM_TIMESTEPS]"
            echo ""
            echo "Arguments:"
            echo "  MODEL_NAME    : Name of the trained NeoDiff model (required)"
            echo "  TIMESTEPS     : Number of timesteps to optimize (default: 20)"
            echo "  DEVICE        : GPU device to use (default: 0)"
            echo "  INIT_POINTS   : Initial random points for Bayesian optimization (default: 40)"
            echo "  N_ITER        : Number of optimization iterations (default: 200)"
            echo "  MODE          : Operation mode - 'optimize', 'eval', or 'list' (default: optimize)"
            echo "  BATCH_SIZE    : Batch size for fairseq-generate evaluation (default: 50)"
            echo "  CUSTOM_TIMESTEPS : Optional custom timesteps for direct evaluation"
            echo ""
            echo "Examples:"
            echo "  bash optimize_neodiff.sh neodiff_poisson_tau100_...  # Full optimization"
            echo "  bash optimize_neodiff.sh MODEL_NAME 20 0 40 200 eval  # Evaluation only"
            echo "  bash optimize_neodiff.sh MODEL_NAME 20 0 40 200 optimize 25  # With custom batch size"
            echo "  bash optimize_neodiff.sh \"\" \"\" \"\" \"\" \"\" list  # List available models"
            echo ""
            list_models
            exit 1
        fi
        ;;
    *)
        echo "Error: Unknown mode '$MODE'. Use 'optimize', 'eval', or 'list'"
        exit 1
        ;;
esac

# Validate model exists
validate_model $MODEL_NAME

# Dataset configuration
DATASET="iwslt14_de_en"
MODEL_DIR="models/${DATASET}/${MODEL_NAME}"

echo "NeoDiff Bayesian Optimization"
echo "=============================="
echo "Model: ${MODEL_NAME}"
echo "Timesteps: ${TIMESTEPS}"
echo "Device: GPU ${DEVICE}"
echo "Mode: ${MODE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Model directory: ${MODEL_DIR}"
echo ""

# Create optimization results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OPT_DIR="${MODEL_DIR}/optimization_${TIMESTAMP}"
mkdir -p $OPT_DIR

# Run Bayesian optimization
if [ "$MODE" = "optimize" ]; then
    echo "Starting Bayesian optimization..."
    echo "Configuration:"
    echo "   - Initial points: ${INIT_POINTS}"
    echo "   - Iterations: ${N_ITER}"
    echo "   - Total evaluations: $((INIT_POINTS + N_ITER))"
    echo "   - Batch size: ${BATCH_SIZE}"
    echo "==============================================="
    
    python scripts/iwslt14_de_en/optimize_neodiff_schedule.py \
        --model-name $MODEL_NAME \
        --timesteps $TIMESTEPS \
        --device $DEVICE \
        --init-points $INIT_POINTS \
        --n-iter $N_ITER \
        --batch-size $BATCH_SIZE \
        2>&1 | tee $OPT_DIR/optimization.log
    
    # Check if optimization completed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Optimization failed. Check logs in $OPT_DIR/optimization.log"
        exit 1
    fi
    
    echo "Optimization completed. Logs saved to: $OPT_DIR/optimization.log"
    
elif [ "$MODE" = "eval" ]; then
    echo "Running evaluation with provided or stored schedule..."

    # Determine custom timesteps to use
    BEST_TIMESTEPS="${CUSTOM_TIMESTEPS}"

    if [ -z "$BEST_TIMESTEPS" ] && [ -f "$MODEL_DIR/best_timesteps.txt" ]; then
        BEST_TIMESTEPS=$(cat "$MODEL_DIR/best_timesteps.txt" | tr -d '[:space:]')
    fi

    if [ -z "$BEST_TIMESTEPS" ]; then
        # Try to read from the latest optimization log under the model directory
        LATEST_OPT_DIR=$(find "$MODEL_DIR" -maxdepth 1 -type d -name "optimization_*" 2>/dev/null | sort -r | head -1)
        if [ -n "$LATEST_OPT_DIR" ] && [ -f "$LATEST_OPT_DIR/optimization.log" ]; then
            BEST_TIMESTEPS_LINE=$(grep -E "Best timesteps:" "$LATEST_OPT_DIR/optimization.log" | tail -n 1)
            echo "Found best timesteps line: $BEST_TIMESTEPS_LINE"
            # Use same robust parsing logic as in optimization mode
            if [[ "$BEST_TIMESTEPS_LINE" =~ \[([^\]]+)\] ]]; then
                BEST_TIMESTEPS=$(echo "${BASH_REMATCH[1]}" | tr -d ' ' | tr -d '"' | tr -d "'")
            else
                BEST_TIMESTEPS=$(echo "$BEST_TIMESTEPS_LINE" | sed -E 's/.*Best timesteps: *([0-9.,\- ]+).*/\1/' | tr -d ' ')
            fi
            echo "Extracted timesteps: $BEST_TIMESTEPS"
        fi
    fi

    if [ -z "$BEST_TIMESTEPS" ]; then
        echo "Error: Could not determine custom timesteps. Provide as 8th arg or ensure best_timesteps.txt exists."
        exit 1
    fi

    echo "Using custom timesteps: $BEST_TIMESTEPS"
    echo "$BEST_TIMESTEPS" > "$MODEL_DIR/best_timesteps.txt"
    mkdir -p "$OPT_DIR"
    echo "$BEST_TIMESTEPS" > "$OPT_DIR/best_timesteps.txt"

    # Run test evaluation with custom timesteps
    bash scripts/iwslt14_de_en/evaluate_neodiff.sh "$MODEL_NAME" "$TIMESTEPS" normal "$DEVICE" 0 "$BEST_TIMESTEPS" 2>&1 | tee "$OPT_DIR/evaluation_custom.log"

    # Copy custom eval outputs into optimization directory for convenience
    CUSTOM_EVAL_DIR="$MODEL_DIR/evaluate/evaluate_step${TIMESTEPS}_beam9x1_normal_stop0_nf1_custom"
    if [ -d "$CUSTOM_EVAL_DIR" ]; then
        cp "$CUSTOM_EVAL_DIR/bleu.txt" "$OPT_DIR/standard_bleu_custom.txt" 2>/dev/null
        if [ -d "$CUSTOM_EVAL_DIR/sacrebleu" ]; then
            cp -r "$CUSTOM_EVAL_DIR/sacrebleu" "$OPT_DIR/sacrebleu_custom" 2>/dev/null
        fi
    fi

    echo "Evaluation with custom timesteps completed. Logs saved to: $OPT_DIR/evaluation_custom.log"
fi

# Run comprehensive evaluation with standard NeoDiff evaluation script
echo ""
echo "Running comprehensive evaluation with optimized schedule..."

# Find the latest evaluation directory from optimization
EVAL_DIRS=$(find $MODEL_DIR/evaluate -name "*optimize_step${TIMESTEPS}_*" -type d 2>/dev/null | sort -r | head -1)

if [ -n "$EVAL_DIRS" ]; then
    echo "Found optimization evaluation directory: $EVAL_DIRS"
    
    # Extract BLEU scores
    if [ -f "$EVAL_DIRS/bleu.txt" ]; then
        echo ""
        echo "Optimization Results (BLEU):"
        echo "============================="
        cat "$EVAL_DIRS/bleu.txt"
        
        # Copy results to optimization directory
        cp "$EVAL_DIRS/bleu.txt" "$OPT_DIR/"
    fi
else
    echo "Warning: No optimization evaluation directory found. Running standard evaluation..."
fi

# Parse best timesteps from optimization log (as comma-separated values)
BEST_TIMESTEPS_LINE=$(grep -E "Best timesteps:" "$OPT_DIR/optimization.log" | tail -n 1)
if [ -n "$BEST_TIMESTEPS_LINE" ]; then
    echo "Debug: Found best timesteps line: $BEST_TIMESTEPS_LINE"
    # Extract content inside brackets [] or after colon, handle multiple formats
    if [[ "$BEST_TIMESTEPS_LINE" =~ \[([^\]]+)\] ]]; then
        # Format: Best timesteps: [0.1, 0.2, 0.3]
        BEST_TIMESTEPS=$(echo "${BASH_REMATCH[1]}" | tr -d ' ' | tr -d '"' | tr -d "'")
    else
        # Format: Best timesteps: 0.1, 0.2, 0.3 (without brackets)
        BEST_TIMESTEPS=$(echo "$BEST_TIMESTEPS_LINE" | sed -E 's/.*Best timesteps: *([0-9.,\- ]+).*/\1/' | tr -d ' ')
    fi
    echo "Debug: Extracted timesteps: $BEST_TIMESTEPS"
    if [ -n "$BEST_TIMESTEPS" ]; then
        echo "Detected best timesteps: $BEST_TIMESTEPS"
        # Persist best timesteps for future reuse
        echo "$BEST_TIMESTEPS" > "$MODEL_DIR/best_timesteps.txt"
        echo "$BEST_TIMESTEPS" > "$OPT_DIR/best_timesteps.txt"
        echo "Running test evaluation with optimized custom timesteps..."
        # Run test evaluation with custom timesteps (steps equals timesteps count)
        bash scripts/iwslt14_de_en/evaluate_neodiff.sh "$MODEL_NAME" "$TIMESTEPS" normal "$DEVICE" 0 "$BEST_TIMESTEPS"
        
        # Copy custom evaluation results back to optimization directory
        CUSTOM_EVAL_DIR="$MODEL_DIR/evaluate/evaluate_step${TIMESTEPS}_beam9x1_normal_stop0_nf1_custom"
        if [ -d "$CUSTOM_EVAL_DIR" ]; then
            echo "Copying custom-timesteps evaluation results from: $CUSTOM_EVAL_DIR"
            cp "$CUSTOM_EVAL_DIR/bleu.txt" "$OPT_DIR/standard_bleu_custom.txt" 2>/dev/null
            if [ -d "$CUSTOM_EVAL_DIR/sacrebleu" ]; then
                cp -r "$CUSTOM_EVAL_DIR/sacrebleu" "$OPT_DIR/sacrebleu_custom" 2>/dev/null
            fi
        else
            echo "Warning: Custom evaluation directory not found: $CUSTOM_EVAL_DIR"
        fi
    else
        echo "Warning: Failed to extract best timesteps from optimization log."
    fi
else
    echo "Warning: No 'Best timesteps' line found in optimization log."
fi

# Run standard evaluation for comparison
echo ""
echo "Running standard evaluation for comparison..."
bash scripts/iwslt14_de_en/evaluate_neodiff.sh $MODEL_NAME $TIMESTEPS normal $DEVICE

# Calculate SacreBLEU if evaluation files exist
echo ""
echo "Calculating SacreBLEU scores..."

# Find the most recent evaluation directory
STANDARD_EVAL_DIR=$(find $MODEL_DIR/evaluate -name "evaluate_step${TIMESTEPS}_*" -type d 2>/dev/null | sort -r | head -1)

if [ -n "$STANDARD_EVAL_DIR" ] && [ -d "$STANDARD_EVAL_DIR" ]; then
    echo "Processing evaluation results from: $STANDARD_EVAL_DIR"
    
    # Process SacreBLEU results
    if [ -d "$STANDARD_EVAL_DIR/sacrebleu" ]; then
        echo ""
        echo "SacreBLEU Results:"
        echo "=================="
        
        for result_file in "$STANDARD_EVAL_DIR/sacrebleu"/*.txt; do
            if [ -f "$result_file" ]; then
                echo "File: $(basename $result_file)"
                python scripts/iwslt14_de_en/sacre_bleu.py "$result_file" en 2>/dev/null || echo "Error calculating SacreBLEU for $result_file"
                echo ""
            fi
        done
        
        # Copy SacreBLEU results to optimization directory
        cp -r "$STANDARD_EVAL_DIR/sacrebleu" "$OPT_DIR/" 2>/dev/null
    fi
    
    # Copy standard evaluation results to optimization directory
    cp "$STANDARD_EVAL_DIR/bleu.txt" "$OPT_DIR/standard_bleu.txt" 2>/dev/null
fi

# Generate summary report
echo ""
echo "Generating optimization summary..."

SUMMARY_FILE="$OPT_DIR/summary.txt"
cat > $SUMMARY_FILE << EOF
NeoDiff Bayesian Optimization Summary
====================================

Model: $MODEL_NAME
Timesteps: $TIMESTEPS
Device: GPU $DEVICE
Mode: $MODE
Timestamp: $TIMESTAMP

Optimization Parameters:
- Initial points: $INIT_POINTS
- Iterations: $N_ITER

Results Directory: $OPT_DIR

Files Generated:
EOF

# List generated files
ls -la $OPT_DIR >> $SUMMARY_FILE

echo ""
echo "Optimization completed successfully!"
echo "=========================================="
echo "Results saved to: $OPT_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "To view results:"
echo "  - Optimization log: cat $OPT_DIR/optimization.log"
echo "  - BLEU scores: cat $OPT_DIR/bleu.txt"
echo "  - Summary: cat $SUMMARY_FILE" 