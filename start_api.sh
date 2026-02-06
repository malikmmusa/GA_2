#!/bin/bash
# Start the Atrophy Advisor API server

echo "=========================================="
echo "Starting Atrophy Advisor API Server"
echo "=========================================="
echo ""

# Change to project root
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found."
    echo "Please run ./setup_environment.sh first."
    exit 1
fi

# Start the API server
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo ""

PYTHONPATH="$(pwd)" ./venv/bin/python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
