#!/bin/bash

# 10 text-to-motion generations with different prompts
# Output goes to separate folders under ./generations/

MODEL_PATH="save/diffusion_baseline/model000500000.pt"  # Update this to your actual model path

#Update prompts as desired
PROMPTS=(
    "a person walks forward slowly"
    "a person runs and then stops"
    "a person jumps and lands"
    "a person waves their right hand"
    "a person sits down on a chair"
    "a person turns around and walks back"
    "a person kicks with their right leg"
    "a person stretches their arms above their head"
    "a person crouches down and stands back up"
    "a person punches forward with both hands"
)

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    # convert prompt to a folder-safe slug: lowercase, spaces to underscores, remove special chars
    SLUG=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd '[:alnum:]_')
    OUTPUT_DIR="./generations/diffusion/$(printf '%02d' $i)_${SLUG}" # Edit to change generation output path
    echo "=========================================="
    echo "Generation $((i+1))/10"
    echo "Prompt: $PROMPT"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="

    python3 -m sample.generate \
        --model_path "$MODEL_PATH" \
        --device -1 \
        --text_prompt "$PROMPT" \
        --num_samples 1 \
        --num_repetitions 1 \
        --output_dir "$OUTPUT_DIR"

    echo "Running visualization..."
    python3 visualize/visualize_new.py \
        --npy_path "$OUTPUT_DIR/results.npy" \
        --output_path "$OUTPUT_DIR/visualization.mp4"

    echo "Done with generation $((i+1))"
    echo ""
done

echo "All 10 generations complete. Results in ./generations/"