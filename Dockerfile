# MDM - Human Motion Diffusion Model
# GPU-enabled inference image with CUDA 11.8
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    unzip \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Copy the entire repo (excluding venv and git history via .dockerignore)
COPY . .

# Install PyTorch GPU first (largest dep, separate layer for caching)
RUN pip install --upgrade pip wheel && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
RUN pip install "numpy<2.0" scipy scikit-learn && \
    pip install matplotlib tqdm trimesh pillow h5py einops && \
    pip install spacy && python3 -m spacy download en_core_web_sm && \
    pip install ftfy regex && \
    pip install git+https://github.com/openai/CLIP.git && \
    pip install smplx==0.1.28 && \
    pip install "moviepy<2.0" && \
    pip install gdown wandb pydantic requests beautifulsoup4 lxml

# Install chumpy from source with Python 3.12 fixes
RUN wget -O /tmp/chumpy-0.70.tar.gz https://pypi.org/packages/source/c/chumpy/chumpy-0.70.tar.gz && \
    cd /tmp && tar -xf chumpy-0.70.tar.gz && \
    cd chumpy-0.70 && \
    printf 'from setuptools import setup, find_packages\nsetup(\n    name="chumpy",\n    version="0.70",\n    packages=find_packages(),\n    install_requires=["numpy", "scipy", "six"],\n)\n' > setup.py && \
    sed -i "s/inspect.getargspec/inspect.getfullargspec/g" chumpy/ch.py && \
    pip install .

# Suppress matplotlib display
ENV MPLBACKEND=Agg

# Output directory for generated motions
RUN mkdir -p /app/outputs

# Download pre-trained model (50-step fast model)
RUN pip install gdown && \
    gdown 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr -O /tmp/model.zip && \
    cd /app && unzip -q /tmp/model.zip && \
    rm /tmp/model.zip

ENTRYPOINT ["python3", "-m", "sample.generate", "--model_path", "./save/humanml_trans_enc_512/model000200000.pt", "--cuda", "False", "--num_samples", "1", "--num_repetitions", "1", "--output_dir", "/app/outputs"]
