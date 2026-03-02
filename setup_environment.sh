#!/bin/bash
# Setup script for Atrophy Advisor development environment

set -euo pipefail

echo "=========================================="
echo "Atrophy Advisor - Environment Setup"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 is not installed or not on PATH."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install other requirements
echo "Installing requirements..."
python -m pip install -r requirements.txt

# Install frontend dependencies for GUI validation scripts
if [ -f "src/frontend/package.json" ]; then
    echo "Installing frontend dependencies..."
    npm --prefix src/frontend install
fi

# Install Playwright browser runtime (best-effort)
if python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("playwright") else 1)
PY
then
    echo "Installing Playwright Chromium browser..."
    python -m playwright install chromium || echo "⚠️ Playwright browser install failed (continuing)."
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the API server:"
echo "  ./start_api.sh"
echo ""
echo "To start the frontend server:"
echo "  ./start_frontend.sh"
echo ""
