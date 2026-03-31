#!/bin/bash

echo "Downloading pretrained models from HuggingFace..."

# Create save directories if they don't exist
mkdir -p save

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Download Flow Matching 540K
echo ""
echo "=== Downloading Flow Matching 540K ==="
python << EOF
from huggingface_hub import hf_hub_download

repo_id = "gergobrezina/mdm-flow-matching"

files = [
    ("flow_matching_540k/model000540000.pt", "save/flow_matching_540k/model000540000.pt"),
    ("flow_matching_540k/opt000540000.pt", "save/flow_matching_540k/opt000540000.pt"),
    ("flow_matching_540k/args.json", "save/flow_matching_540k/args.json"),
]

for remote_file, local_file in files:
    try:
        print(f"Downloading {remote_file.split('/')[-1]}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_file,
            local_dir="save",
            local_dir_use_symlinks=False
        )
        print(f"✓ {remote_file.split('/')[-1]} downloaded")
    except Exception as e:
        print(f"⚠ Could not download {remote_file.split('/')[-1]}: {e}")

print("✓ Flow Matching 540K complete: save/flow_matching_540k/")
EOF

# Download Diffusion 540K
echo ""
echo "=== Downloading Diffusion 540K ==="
python << EOF
from huggingface_hub import hf_hub_download

repo_id = "gergobrezina/mdm-flow-matching"

files = [
    ("diffusion_540k/model000540000.pt", "save/diffusion_540k/model000540000.pt"),
    ("diffusion_540k/opt000540000.pt", "save/diffusion_540k/opt000540000.pt"),
    ("diffusion_540k/args.json", "save/diffusion_540k/args.json"),
]

for remote_file, local_file in files:
    try:
        print(f"Downloading {remote_file.split('/')[-1]}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_file,
            local_dir="save",
            local_dir_use_symlinks=False
        )
        print(f"✓ {remote_file.split('/')[-1]} downloaded")
    except Exception as e:
        print(f"⚠ Could not download {remote_file.split('/')[-1]}: {e}")

print("✓ Diffusion 540K complete: save/diffusion_540k/")
EOF

echo ""
echo "========================================="
echo "Download complete!"
echo ""
echo "Models saved in:"
echo "  • save/flow_matching/"
echo "    - model000540000.pt (checkpoint)"
echo "    - opt000540000.pt (config)"
echo "    - args.json (arguments)"
echo ""
echo "  • save/diffusion/"
echo "    - model000540000.pt (checkpoint)"
echo "    - opt000540000.pt (config)"
echo "    - args.json (arguments)"
echo ""
echo "Quick test:"
echo "  python -m sample.generate \\"
echo "    --model_path save/flow_matching/model000540000.pt \\"
echo "    --text_prompt \"a person walks forward\" \\"
echo "    --num_samples 3"