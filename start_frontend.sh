#!/bin/bash
# Start the Atrophy Advisor Frontend

set -euo pipefail

echo "=========================================="
echo "Starting Atrophy Advisor Frontend"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v npm >/dev/null 2>&1; then
    echo "❌ npm is not installed or not on PATH."
    exit 1
fi

# Check if node_modules exists
if [ ! -d "src/frontend/node_modules" ]; then
    echo "⚠️  Dependencies not installed."
    echo "Installing npm packages..."
    cd "src/frontend"
    if [ -f "package-lock.json" ]; then
        npm ci
    else
        npm install
    fi
    cd ../..
fi

# Start the frontend development server
cd "src/frontend"
echo "Starting Vite development server on http://localhost:3000"
echo "Make sure the API server is running on http://localhost:8000"
echo ""

exec npm run dev
