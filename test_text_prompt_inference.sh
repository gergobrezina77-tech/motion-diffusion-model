#!/bin/bash
# Quick test to verify text-only inference works with the fix

cd /home/gergo/projects/motion-diffusion-model

# Activate environment
source mdm_env/bin/activate 2>/dev/null || true

echo "=== Testing Text-Only Inference Fix ==="
echo ""
echo "Running: python sample/generate.py with --text_prompt only"
echo "(No dataset files required, only normalization constants)"
echo ""

# Run with a simple text prompt
python sample/generate.py \
  --model_path save/humanml_trans_enc_512/model000200000.pt \
  --text_prompt "a person walking forward" \
  --motion_length 6.0 \
  --num_samples 1 \
  --seed 42 \
  --device cpu \
  --output_dir /tmp/mdm_test_output

EXIT_CODE=$?

echo ""
echo "=== Result ==="
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ SUCCESS: Text-only inference completed without dataset loading errors"
  echo "Output saved to: /tmp/mdm_test_output"
else
  echo "✗ FAILED: Inference returned exit code $EXIT_CODE"
fi

exit $EXIT_CODE
