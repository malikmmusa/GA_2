#!/bin/bash
# Start the Atrophy Advisor Frontend

echo "=========================================="
echo "Starting Atrophy Advisor Frontend"
echo "=========================================="
echo ""

# Check if node_modules exists
if [ ! -d "src/frontend/node_modules" ]; then
    echo "⚠️  Dependencies not installed."
    echo "Installing npm packages..."
    cd src/frontend
    npm install
    cd ../..
fi

# Start the frontend development server
cd src/frontend
echo "Starting Vite development server on http://localhost:3000"
echo "Make sure the API server is running on http://localhost:8000"
echo ""

npm run dev
