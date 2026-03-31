#!/bin/bash

# Installation script for Flow Matching MDM
# Handles problematic packages like chumpy

set -e  # Exit on error

echo "=========================================="
echo "Installing Flow Matching MDM Dependencies"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or later required"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install main requirements (excluding chumpy)
echo ""
echo "Installing main dependencies..."
pip install -r requirements.txt

# Patch chumpy for numpy 1.24+ compatibility
# numpy removed type aliases (bool, int, float, complex, object, str) in 1.24
echo ""
echo "Patching chumpy for numpy compatibility..."

CHUMPY_LOCATION=$(pip show chumpy 2>/dev/null | grep "^Location:" | awk '{print $2}')

if [ -z "$CHUMPY_LOCATION" ]; then
    echo "✗ chumpy not installed (pip show found nothing)"
    exit 1
fi

CHUMPY_INIT="$CHUMPY_LOCATION/chumpy/__init__.py"

if [ ! -f "$CHUMPY_INIT" ]; then
    echo "✗ Could not find chumpy __init__.py at $CHUMPY_INIT"
    exit 1
fi

echo "Found chumpy at: $CHUMPY_INIT"

# Replace the broken numpy import line with compatible aliases
sed -i 's/^from numpy import bool, int, float, complex, object, str, nan, inf$/from numpy import nan, inf\nimport numpy as _np\nbool = _np.bool_\nint = _np.int_\nfloat = _np.float64\ncomplex = _np.complex128\nobject = _np.object_\nstr = _np.str_/' "$CHUMPY_INIT"

echo "✓ Chumpy __init__.py patched"

# Fix inspect.getargspec removed in Python 3.11+ (replaced by getfullargspec)
CHUMPY_CH="$CHUMPY_LOCATION/chumpy/ch.py"
if [ -f "$CHUMPY_CH" ]; then
    sed -i 's/inspect\.getargspec(/inspect.getfullargspec(/g' "$CHUMPY_CH"
    echo "✓ Chumpy ch.py patched (getargspec -> getfullargspec)"
fi

echo "✓ Chumpy patched successfully"

# Verify critical imports
echo ""
echo "Verifying installation..."
python << EOF
import sys
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    import torchdiffeq
    print("✓ torchdiffeq")

    import einops
    print("✓ einops")

    import clip
    print("✓ CLIP")

    import spacy
    print("✓ spacy")

    import chumpy
    print("✓ chumpy")

    print("\n✓ All critical packages installed successfully!")

except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
