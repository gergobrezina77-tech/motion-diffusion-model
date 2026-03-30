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
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

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
git clone https://github.com/your-username/motion-diffusion-model.git
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
```

**What these download:**
- `glove/` - GloVe word embeddings
- `t2m/` - Pre-trained evaluators for R-Precision and FID metrics
- `body_models/` - SMPL body model for visualization

### Verify Installation

Run a quick 100-step smoke test to verify everything works:

```bash
python -m train.train_mdm \
    --save_dir save/smoke_test \
    --dataset humanml \
    --batch_size 64 \
    --num_steps 100 \
    --use_flow_matching
```

**Expected output:**
- Training should start without errors
- Checkpoint saved at `save/smoke_test/model000000100.pt`
- Takes ~1-2 minutes

If this runs successfully, your installation is complete!

---

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

Contact the HumanML3D authors or your research group for the already-processed dataset (saves ~1-2 days of preprocessing).

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
    --use_flow_matching \
    --save_interval 50000
```

**What happens:**
- Training runs for ~3-4 days on RTX 5070
- Checkpoints saved every 50K steps in `save/flow_matching_540k/`
- Loss logged to console (and wandb if configured)

**Checkpoints:**
- `model000050000.pt` through `model000540000.pt`
- Evaluate at 100K, 200K, 340K, 540K for comparison

**When to stop:**
- **100K steps**: Early results, useful for debugging
- **200K steps**: Competitive quality, good for quick experiments
- **340K steps**: Near-optimal performance
- **540K steps**: Maximum quality (diminishing returns beyond this)

### Gaussian Diffusion (Best Quality, Smooth Motion)

Train a Gaussian Diffusion baseline for 340K steps:

```bash
python -m train.train_mdm \
    --save_dir save/diffusion_340k \
    --dataset humanml \
    --batch_size 64 \
    --num_steps 340000 \
    --lr 1e-4 \
    --save_interval 50000
```

**What happens:**
- Training runs for ~2-3 days
- Best FID typically around 340K-400K steps
- Smoother convergence than Flow Matching

**Note**: No `--use_flow_matching` flag = Gaussian Diffusion by default

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

**Optional: Use Weights & Biases**

If you have wandb configured, training will automatically log:
- Loss curves
- Learning rate schedule
- Generated samples (periodic)

```bash
# Login to wandb (one-time setup)
wandb login

# Training will now log to wandb dashboard
```

### Resuming Training

If training is interrupted, resume from the latest checkpoint:

```bash
python -m train.train_mdm \
    --save_dir save/flow_matching_540k \
    --dataset humanml \
    --batch_size 64 \
    --num_steps 540000 \
    --use_flow_matching \
    --resume_checkpoint save/flow_matching_540k/model000200000.pt
```

---

## Generating Motion

### Basic Text-to-Motion Generation

Generate motion from a text prompt:

```bash
python -m sample.generate \
    --model_path save/flow_matching_540k/model000540000.pt \
    --text_prompt "a person walks forward slowly" \
    --num_samples 5 \
    --num_repetitions 1
```

**Output:**
- Individual videos: `generations/[timestamp]/sample00_rep00.mp4` through `sample04_rep00.mp4`
- Motion data: `generations/[timestamp]/results.npy`
- Text prompts: `generations/[timestamp]/results.txt`

**Parameters:**
- `--num_samples 5`: Generate 5 different motions
- `--num_repetitions 1`: One version of each motion
- `--motion_length 6.0`: Duration in seconds (default: 6.0)

### Batch Generation (Multiple Prompts)

Create a file with multiple prompts (one per line):

```bash
cat > prompts.txt << EOF
a person walks forward slowly
a person jumps high in the air
a person waves hello with right hand
a person sits down on a chair
a person runs and then stops
EOF
```

Generate motions for all prompts:

```bash
python -m sample.generate \
    --model_path save/flow_matching_540k/model000540000.pt \
    --input_text prompts.txt \
    --num_repetitions 3
```

**Output:**
- 5 prompts × 3 repetitions = 15 videos total
- Each prompt gets multiple variations

### Flow Matching Specific Options

**Adjust ODE integration steps** (speed vs quality tradeoff):

```bash
# Faster inference (5 steps, more jitter)
python -m sample.generate \
    --model_path save/flow_matching_540k/model000540000.pt \
    --text_prompt "a person dances" \
    --ode_steps 5

# Default (10 steps, good balance)
--ode_steps 10

# Higher quality (20 steps, smoother but slower)
--ode_steps 20
```

**Use different ODE solvers:**

```bash
# Euler method (default, fast)
--ode_method euler_raw

# Dopri5 (adaptive, more accurate but slower)
--ode_method dopri5 \
--ode_rtol 1e-5 \
--ode_atol 1e-5
```

### Classifier-Free Guidance

Control the strength of text conditioning:

```bash
# Stronger guidance (more adherence to text, less diversity)
--guidance_param 3.0

# Default guidance
--guidance_param 2.5

# Weaker guidance (more diversity, less text adherence)
--guidance_param 1.5

# No guidance (unconditional generation)
--guidance_param 1.0
```

### Generation Tips

**For best results:**
- Use checkpoints from 340K+ steps
- Start with `--guidance_param 2.5` (default)
- Flow Matching: Use 10 ODE steps as baseline
- Generate multiple repetitions to see diversity

**Common prompts that work well:**
- "a person walks forward"
- "a person jumps high"
- "a person sits down"
- "a person waves with right hand"
- "a person runs and stops"

**Prompts to avoid:**
- Very long descriptions (>15 words)
- Ambiguous actions
- Multiple simultaneous actions

---

## Evaluating Models

### Standard Evaluation

Evaluate a trained model on the HumanML3D test set:

```bash
python -m eval.eval_humanml \
    --model_path save/flow_matching_540k/model000540000.pt \
    --eval_mode wo_mm
```

**What happens:**
- Generates 32 batches of samples from test set
- Computes metrics against ground truth
- Takes ~20-30 minutes

**Metrics reported:**
- **R-Precision (top-1, top-2, top-3)**: Text-motion alignment (higher is better)
- **FID**: Motion realism (lower is better)
- **Diversity**: Motion variety (should match ground truth ~9.5)

**Example output:**
```
R-Precision (top-1): 0.548 ± 0.014
R-Precision (top-2): 0.632 ± 0.023
R-Precision (top-3): 0.646 ± 0.032
FID: 0.353 ± 0.331
Diversity: 9.609 ± 0.119
```

### Evaluate Multiple Checkpoints

Compare different training stages:

```bash
# Evaluate checkpoints at 100K, 200K, 340K, 540K
for ckpt in 100000 200000 340000 540000; do
    echo "Evaluating checkpoint ${ckpt}..."
    python -m eval.eval_humanml \
        --model_path save/flow_matching_540k/model${ckpt}.pt \
        --eval_mode wo_mm \
        > eval_results_${ckpt}.txt
done
```

### Understanding Metrics

**R-Precision (Text-Motion Alignment):**
- Measures how well generated motion matches the text prompt
- Top-3 R-Precision > 0.63 is competitive
- Flow Matching typically achieves higher R-Precision than Diffusion

**FID (Fréchet Inception Distance):**
- Measures how realistic the generated motion is
- FID < 0.5 is good, < 0.4 is excellent
- Gaussian Diffusion typically achieves lower FID than Flow Matching

**Diversity:**
- Measures variety in generated motions
- Should be close to ground truth (~9.5)
- Too low = mode collapse, too high = incoherent motion

**Multimodality** (optional, `--eval_mode mm`):
- Requires 30 repetitions per prompt (very slow)
- Measures diversity for the same prompt
- We typically skip this (`--eval_mode wo_mm`)

### Typical Results (540K Flow Matching vs 340K Diffusion)

| Metric | Flow Matching | Gaussian Diffusion | Ground Truth |
|--------|--------------|-------------------|--------------|
| R-Precision (top-3) | 0.632 ± 0.032 | 0.548 ± 0.014 | 0.797 |
| FID | 0.353 ± 0.331 | 0.681 ± 0.074 | 0.002 |
| Diversity | 9.609 ± 0.119 | 9.503 ± 0.107 | 9.503 |
| Inference | ~0.5s (10 steps) | ~4s (1000 steps) | — |

### Save Evaluation Results

Redirect output to a file for later analysis:

```bash
python -m eval.eval_humanml \
    --model_path save/flow_matching_540k/model000540000.pt \
    --eval_mode wo_mm \
    | tee eval_fm_540k.txt
```

---

## Troubleshooting

### Installation Issues

**"torch not compiled with CUDA enabled"**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**"ModuleNotFoundError: torchdiffeq"**
```bash
pip install torchdiffeq
```

### Dataset Issues

**"FileNotFoundError: dataset/HumanML3D"**
- Dataset not copied to correct location
- Run: `cp -r ~/HumanML3D/HumanML3D dataset/`

**"FileNotFoundError: dataset/HumanML3D/Mean.npy"**
- Dataset incomplete or not preprocessed
- Verify all files exist: `ls dataset/HumanML3D/`

**"No data loaded" or empty dataset**
- Check split files: `wc -l dataset/HumanML3D/*.txt`
- Verify `new_joint_vecs/` has `.npy` files

### Training Issues

**"CUDA out of memory"**
- Reduce batch size: `--batch_size 32` or `--batch_size 16`
- Check no other processes using GPU: `nvidia-smi`
- Close other applications

**"Loss is NaN"**
- Check learning rate (should be `1e-4`)
- Verify dataset loaded correctly
- Try restarting from a working checkpoint

**Training very slow**
- Check GPU utilization: `nvidia-smi`
- Verify CUDA version matches PyTorch
- Reduce `--batch_size` if GPU is under-utilized

### Generation Issues

**"TypeError: must be real number, not NoneType" (moviepy error)**
- Known issue with video concatenation
- Individual sample videos still generate correctly
- Ignore the error, check `generations/` for output files

**Jittery/shaking motion (Flow Matching)**
- This is expected behavior in Flow Matching outputs
- Try increasing `--ode_steps` to 20 (slower but smoother)
- Or use Gaussian Diffusion for smooth motion

**No videos generated**
- Check if `results.npy` exists (motion data saved successfully)
- Install visualization dependencies: `pip install matplotlib`
- Check disk space

### Evaluation Issues

**"ModuleNotFoundError: t2m"**
- Evaluation models not downloaded
- Run: `bash prepare/download_t2m_evaluators.sh`
- Verify `t2m/` folder exists in project root

**FID computation fails**
- Ensure you generated enough samples
- Check CLIP model downloaded correctly
- Verify test set exists: `ls dataset/HumanML3D/test.txt`

**Very high FID (> 5.0)**
- Model undertrained (< 100K steps)
- Wrong checkpoint loaded
- Dataset mismatch

---

## FAQ

**Q: How much GPU memory do I need?**  
A: 12GB minimum. Tested on RTX 5070 (12GB). RTX 3080/4070 or better recommended. Can reduce `--batch_size` for GPUs with less memory.

**Q: Can I train on CPU?**  
A: Technically yes, but prohibitively slow (~months instead of days). GPU required for practical training.

**Q: How long does training take?**  
A: Flow Matching 540K steps: ~3-4 days on RTX 5070. Gaussian Diffusion 340K steps: ~2-3 days. Scales with GPU performance.

**Q: Can I train on multiple GPUs?**  
A: Code supports it but not extensively tested. Use `--device 0,1` for two GPUs.

**Q: Why is Flow Matching FID higher than Diffusion?**  
A: FID doesn't capture all quality aspects. Flow Matching has better text alignment (R-Precision) but visible jitter in hands/feet. It's a tradeoff, not strictly worse.

**Q: How do I fix the jitter in Flow Matching outputs?**  
A: Known limitation. Try `--ode_steps 20` (slower but smoother). For smooth motion, use Gaussian Diffusion. The FlowMotion paper suggests switching from velocity-field to direct x₀ prediction (not implemented here).

**Q: Can I use my own text encoder instead of CLIP?**  
A: Requires code modification in `model/mdm.py`. CLIP is frozen and required for evaluation metrics.

**Q: Which model should I use?**  
A: **Flow Matching** if you need speed (real-time, interactive). **Gaussian Diffusion** if you need best quality (smooth, artifact-free). See comparison table in Evaluation section.

**Q: Can I fine-tune on my own dataset?**  
A: Yes, but requires modifying data loaders. HumanML3D's 263-dim representation is specific to their preprocessing.

**Q: Where are the checkpoints saved?**  
A: In `save/[model_name]/`. Each checkpoint is ~200MB. Clean up old checkpoints to save disk space.

**Q: How do I cite this work?**  
A: See the main README for citation information (MDM, Flow Matching, HumanML3D papers).

**Q: The code crashes with "RuntimeError: CUDA error"**  
A: Check CUDA compatibility: `torch.cuda.is_available()`. Update GPU drivers. Verify PyTorch installed with correct CUDA version.

---

## Quick Reference

### Essential Commands

**Train Flow Matching:**
```bash
python -m train.train_mdm --save_dir save/fm --dataset humanml --use_flow_matching --num_steps 540000
```

**Train Gaussian Diffusion:**
```bash
python -m train.train_mdm --save_dir save/diff --dataset humanml --num_steps 340000
```

**Generate from text:**
```bash
python -m sample.generate --model_path save/fm/model000540000.pt --text_prompt "a person walks"
```

**Evaluate model:**
```bash
python -m eval.eval_humanml --model_path save/fm/model000540000.pt --eval_mode wo_mm
```

### File Locations

- **Dataset**: `dataset/HumanML3D/`
- **Checkpoints**: `save/[model_name]/model*.pt`
- **Generated videos**: `generations/[timestamp]/`
- **Evaluation models**: `t2m/`, `body_models/`, `glove/`
- **Training logs**: `save/[model_name]/log.txt`

### Default Hyperparameters

- **Batch size**: 64
- **Learning rate**: 1e-4
- **Optimizer**: AdamW
- **Flow Matching ODE steps**: 10
- **Diffusion sampling steps**: 1000 (DDPM)
- **Guidance scale**: 2.5
- **Motion length**: 6.0 seconds

---

## Support

**Issues & Bugs:**  
Open an issue on GitHub: [link to repo]

**Questions:**  
Contact: [your email]

**Research conducted at:**  
Technical University of Munich, CAMP Chair  
March 2026

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