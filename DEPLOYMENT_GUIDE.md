# Atrophy Advisor - Deployment Guide

## Quick Start

### Prerequisites

- Python 3.8+ (with pip)
- Node.js 18+ (with npm)
- PyTorch 2.0+ (for MPS/CUDA support)

### One-Time Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd OCT_Project

# 2. Set up Python environment
./setup_environment.sh

# 3. Install frontend dependencies
cd src/frontend
npm install
cd ../..
```

### Running the Application

**Terminal 1 - Backend API:**
```bash
./start_api.sh
# Backend will run on http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
# Frontend will run on http://localhost:3000
```

**Open your browser to:** `http://localhost:3000`

---

## Detailed Setup

### 1. Python Environment Setup

The application requires specific Python packages. We recommend using a virtual environment.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install PyTorch (with MPS support for Apple Silicon)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 2. Model Weights

The trained RETFound U-Net model weights must be present at:
```
weights/best_disc_model.pth
```

If missing, you'll need to:
1. Train a new model, OR
2. Obtain pre-trained weights from the research team

### 3. Frontend Dependencies

```bash
cd src/frontend
npm install
```

This installs:
- React 18
- TypeScript
- Vite
- Tailwind CSS
- Axios

---

## Architecture

```
┌─────────────┐
│   Browser   │  http://localhost:3000
│  (Frontend) │
└──────┬──────┘
       │ API Requests
       ├── POST /api/detect-disc
       ├── POST /api/detect-fovea
       ├── POST /api/segment-ga
       ├── POST /api/calculate-distance
       └── POST /api/calculate-progression
       ▼
┌─────────────┐
│  FastAPI    │  http://localhost:8000
│  (Backend)  │
└──────┬──────┘
       │
       ├── DiscDetectorService → RETFound U-Net
       ├── FoveaDetectorService → Anatomy-aware logic
       ├── GASegmenterService → K-means clustering
       └── CalculatorServices → Distance & progression
```

---

## Development Workflow

### Backend Development

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python tests/test_disc_detector.py

# Start API server (with auto-reload)
cd src/api
python -m uvicorn main:app --reload --port 8000

# View API documentation
open http://localhost:8000/docs
```

### Frontend Development

```bash
cd src/frontend

# Start dev server (with hot reload)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

---

## API Endpoints

### POST /api/detect-disc
Upload OCT image → Returns disc coordinates + 1800µm reference

### POST /api/detect-fovea
Upload image + disc coords → Returns fovea location + eye side

### POST /api/segment-ga
Upload image → Returns GA region contours

### POST /api/calculate-distance
Fovea coords + GA region → Returns distance in pixels & microns

### POST /api/calculate-progression
Before/after distances + dates → Returns rate & prediction

**Full API documentation:** http://localhost:8000/docs

---

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Solution:** Activate virtual environment and install dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: weights/best_disc_model.pth`
**Solution:** Ensure model weights are in the correct location
```bash
ls -lh weights/best_disc_model.pth
```

### Issue: Frontend can't connect to backend
**Solution:** Ensure backend is running on port 8000
```bash
curl http://localhost:8000/health
```

### Issue: `npm ERR! Cannot find module`
**Solution:** Install frontend dependencies
```bash
cd src/frontend
rm -rf node_modules package-lock.json
npm install
```

---

## Production Deployment

### Backend (FastAPI)

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
cd src/api
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend (React)

```bash
cd src/frontend

# Build production bundle
npm run build

# Serve with nginx, Apache, or any static file server
# dist/ folder contains the built files
```

### Docker (Future)

A Dockerfile will be provided in Phase 4 for containerized deployment.

---

## Environment Variables

### Backend

```bash
# Optional: Set device preference
export DEVICE="cuda"  # or "mps" or "cpu"

# Optional: Change model path
export MODEL_PATH="weights/custom_model.pth"
```

### Frontend

Create `src/frontend/.env`:
```bash
VITE_API_BASE_URL=http://localhost:8000/api
```

---

## Performance Considerations

### Backend
- **GPU Acceleration**: Use MPS (Apple Silicon) or CUDA (NVIDIA) for faster inference
- **Model Loading**: Model is loaded once at startup and cached
- **Image Processing**: Runs on CPU, optimize with OpenCV SIMD if needed

### Frontend
- **Canvas Rendering**: Scales images to max 800px width
- **API Calls**: Sequential for each image, could be parallelized
- **State Management**: React state, consider Redux for complex flows

---

## Monitoring

### Backend Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### API Status
```bash
curl http://localhost:8000/api/disc-detector/status
# Returns model info and device
```

---

## Security Notes

⚠️ **Current Status: Development Only**

- No authentication implemented
- CORS allows all origins
- No rate limiting
- No input sanitization beyond FastAPI validation

**Before production deployment:**
1. Add authentication (JWT, OAuth, etc.)
2. Configure CORS properly
3. Add rate limiting
4. Implement logging and monitoring
5. Use HTTPS
6. Add input validation and sanitization

---

## Support

For issues, questions, or contributions:
1. Check the [API Documentation](API_DOCUMENTATION.md)
2. Review [README.md](README.md)
3. Contact the development team
