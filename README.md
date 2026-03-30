# Flow Matching for Text-to-Motion Generation
**User Guide: Setup, Training, Generation, and Evaluation**

A controlled comparison of Gaussian Diffusion and Conditional Flow Matching for text-to-motion synthesis. Both generative engines share the same Transformer backbone from [MDM](https://github.com/GuyTevet/motion-diffusion-model), trained on HumanML3D with identical hyperparameters.

**Key Result**: Flow Matching achieves ~7.7× inference speedup (10 ODE steps vs 1000 DDPM steps) with competitive metrics, but produces jitter artifacts in local joints.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dataset Setup](#dataset-setup)
3. [Training Models](#training-models)
4. [Generating Motion](#generating-motion)
5. [Evaluating Models](#evaluating-models)


---

## Getting Started

### Prerequisites

**Hardware Requirements:**
- GPU with 8GB+ VRAM (tested on RTX 5070)
- 50GB+ free disk space for dataset


**Software Requirements:**
- Python 3.8 or later
- CUDA 11.x or later
- Git

### Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/gergobrezina77-tech/motion-diffusion-model.git
cd motion-diffusion-model
```

**Step 2: Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` (PyTorch with CUDA)
- `torchdiffeq` (for ODE integration in Flow Matching)
- `clip` (for text encoding)
- `einops` (for tensor operations)

**Step 4: Download supporting files**
```bash
# Download CLIP text encoder and evaluation models
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_smpl_files.sh
bash prepare/download_diffusion_flow_matching.sh
```

**What these download:**
- `glove/` - GloVe word embeddings
- `t2m/` - Pre-trained evaluators for R-Precision and FID metrics
- `body_models/` - SMPL body model for visualization
- `diffusion flow matching` - Hugging Face models both at 540K

## Dataset Setup

### Downloading HumanML3D

HumanML3D is a large-scale text-motion dataset with 14,616 motion sequences and 44,970 text descriptions.

**Option 1: Download from Official Source**

1. Clone the HumanML3D repository:
```bash
cd ~/
git clone https://github.com/EricGuo5513/HumanML3D.git
cd HumanML3D
```

2. Follow the official [HumanML3D instructions](https://github.com/EricGuo5513/HumanML3D) to:
   - Download AMASS dataset
   - Download HumanML3D annotations
   - Run preprocessing scripts

3. This will create the processed dataset in `~/HumanML3D/HumanML3D/`

**Option 2: Request Processed Dataset**

### Installing the Dataset

Once you have the processed HumanML3D data, copy it to your MDM repository:

```bash
# Navigate to your MDM repository
cd ~/motion-diffusion-model

# Create dataset directory
mkdir -p dataset

# Copy HumanML3D data
cp -r ~/HumanML3D/HumanML3D dataset/
```

### Verify Dataset Structure

Check that your dataset has the correct structure:

```bash
ls dataset/HumanML3D/
```

**Expected files and directories:**
```
Mean.npy              # Normalization statistics
Std.npy               # Normalization statistics
train.txt             # Training split (sample IDs)
test.txt              # Test split (sample IDs)
val.txt               # Validation split (sample IDs)
new_joint_vecs/       # Processed motion data (263-dim vectors)
new_joints/           # Joint positions
texts/                # Text descriptions
```

**Verify file counts:**
```bash
echo "Train samples: $(wc -l < dataset/HumanML3D/train.txt)"
echo "Test samples: $(wc -l < dataset/HumanML3D/test.txt)"
echo "Val samples: $(wc -l < dataset/HumanML3D/val.txt)"
```

**Expected counts:**
- Train: ~20,000 samples
- Test: ~4,000 samples  
- Val: ~2,000 samples

If you see these files and reasonable counts, your dataset is ready!
### Verify Installation

Run a quick 100-step smoke test to verify everything works:

```bash
python -m train.train_mdm \
    --save_dir save/smoke_test \
    --dataset humanml \
    --batch_size 64 \
    --num_steps 100 \
    --diffusion_type flow
```

**Expected output:**
- Training should start without errors
- Checkpoint saved at `save/smoke_test/model000000100.pt`
- Takes ~1-2 minutes

If this runs successfully, your installation is complete!

---
## Training Models

### Flow Matching (Fast Inference, Good Alignment)

Train a Flow Matching model for 540K steps:

```bash
python -m train.train_mdm \
    --save_dir save/flow_matching_540k \
    --dataset humanml \
    --batch_size 64 \
    --num_steps 540000 \
    --lr 1e-4 \
    --diffusion_type flow \
    --save_interval 50000
```

### Gaussian Diffusion (Reliable, Smooth Motion)

Train a Gaussian Diffusion baseline for 540K steps:

```bash
python -m train.train_mdm \
    --save_dir save/diffusion_540k \
    --dataset humanml \
    --batch_size 64 \
    --num_steps 540000 \
    --lr 1e-4 \
    --diffusion_type diffusion \
    --save_interval 50000
```


### Training Configuration Options

**Adjust batch size** (if GPU memory is limited):
```bash
--batch_size 32  # Reduces memory usage, increases training time
```

**Checkpoint frequency**:
```bash
--save_interval 10000  # Save every 10K steps instead of 50K
```

**Learning rate** (default is 1e-4, rarely needs changing):
```bash
--lr 5e-5  # Lower learning rate for more stable training
```

### Monitoring Training

**Watch training progress:**
```bash
# View loss in real-time
tail -f save/flow_matching_540k/log.txt
```

**Check checkpoints:**
```bash
ls -lh save/flow_matching_540k/
```


### Resuming Training

If training is interrupted, resume from the latest checkpoint (Example):

```bash
python -m train.train_mdm \
    --save_dir save/flow_matching_540k \
    --dataset humanml \
    --batch_size 64 \
    --num_steps 540000 \
    --diffusion_type flow \
    --resume_checkpoint save/flow_matching_540k/model000200000.pt
```

---
## Generation and Visualization

### Generating Motion Flow

```bash
./generate_flow_matching.sh
```
You can set the model location, output location, prompt and generation parameters in the `generate_flow_matching.sh` file.

For solving the ODE, you can use `--ode_method euler`/`rk4` solvers by setting the `--ode_steps` flag or use the adaptive `--ode_method dopri5`.

The output folder will contain the results and the `.mp4` file.


### Generating Diffusion

```bash
./generate_diffusions.sh
```
You can set the model location, output location, and prompt in `generate_diffusions.sh`.

The output folder will contain the results and the `.mp4` file.


## Evaluating Models

### Standard Evaluation

Evaluate a trained model on the HumanML3D test set:

```bash
python -m eval.eval_humanml     --model_path save/model_folder/model000100000.pt     --eval_mode wo_mm 2>&1 | tee log_folder.log
```


### File Locations

- **Dataset**: `dataset/HumanML3D/`
- **Checkpoints**: `save/[model_name]/model*.pt`
- **Generated videos**: `generations/[generation_name]/[prompt_name]/`



### Default Hyperparameters


- **Optimizer**: AdamW
- **Flow Matching ODE steps**: 10
- **Diffusion sampling steps**: 1000 (DDPM)
- **Guidance scale**: 2.5
- **Motion length**: 6.0 seconds

---

## License

This project inherits the license from the original MDM repository. See LICENSE file for details.

## Acknowledgments

This code builds directly on:
- [MDM (Tevet et al.)](https://github.com/GuyTevet/motion-diffusion-model) - Transformer backbone and training infrastructure
- [Guided Diffusion (OpenAI)](https://github.com/openai/guided-diffusion) - Original diffusion utilities
- [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747) - Flow matching theory
- [Motion Flow Matching (Hu et al.)](https://arxiv.org/abs/2312.05708) - Application to motion domain

Thanks to the HumanML3D and CLIP teams for datasets and pretrained encoders.