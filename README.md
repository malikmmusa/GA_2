# Atrophy Advisor

A web-based OCT image analysis tool for tracking Geographic Atrophy (GA) progression and predicting foveal involvement.

## Overview

Atrophy Advisor allows clinicians to upload two temporal OCT images (before and after) of the same eye, automatically detect key anatomical landmarks (optic disc, fovea), segment GA lesions, and calculate the rate of disease progression toward the fovea. The tool provides a predicted date for when GA may reach the fovea based on the observed progression rate.

## Features

- **Dual Image Upload**: Side-by-side upload areas for before/after temporal comparison
- **Date Selection**: Date pickers for each image to calculate progression rate
- **Automatic Eye Detection**: Auto-detects OD (right eye) vs OS (left eye) with manual override
- **Optic Disc Detection**: RETFound U-Net model draws a red vertical line representing 1800 microns
- **Fovea Detection**: Anatomy-aware auto-detection with interactive click-to-adjust
- **Image Registration**: Vessel-based alignment auto-transfers confirmed fovea from Image 1 to Image 2, with confidence scoring and manual override
- **GA Segmentation**: K-means clustering with multi-channel features highlights GA regions
- **Interactive Selection**: Hover to highlight GA regions, click to select
- **Distance Calculation**: Measures fovea-to-GA distance in microns using optic disc as reference
- **Progression Analysis**: Calculates rate of change and predicts foveal involvement date
- **PDF Report**: Downloadable report with annotated images and calculations

---

## Quick Start

### Prerequisites

- Python 3.8+ (with pip)
- Node.js 18+ (with npm)
- PyTorch 2.0+ (for MPS/CUDA support)
- Trained model weights at `weights/best_disc_model.pth`

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd OCT_Project

# 2. Set up Python environment
./setup_environment.sh

# 3. Install frontend dependencies
cd src/frontend && npm install && cd ../..
```

### Run

```bash
# Terminal 1: Start backend API
./start_api.sh
# Backend runs on http://localhost:8000
# Interactive API docs at http://localhost:8000/docs

# Terminal 2: Start frontend
./start_frontend.sh
# Frontend runs on http://localhost:3000
```

Open your browser to `http://localhost:3000`.

---

## Technical Architecture

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Backend** | FastAPI | Production-ready, async support, automatic OpenAPI docs |
| **Frontend** | React + TypeScript | Component-based, production-ready, excellent ecosystem |
| **ML Inference** | PyTorch | Existing RETFound U-Net model |
| **Image Processing** | OpenCV, NumPy, SciPy | Existing pipeline compatibility |
| **PDF Generation** | ReportLab | Professional PDF output |
| **Styling** | Tailwind CSS | Utility-first, rapid UI development |

### Architecture Diagram

```
┌─────────────┐
│   Browser    │  http://localhost:3000
│  (React)     │
└──────┬───────┘
       │ API Requests (Axios)
       ▼
┌─────────────┐
│  FastAPI     │  http://localhost:8000
│  (Backend)   │
└──────┬───────┘
       │
       ├── DiscDetectorService     → RETFound U-Net model
       ├── FoveaDetectorService    → Anatomy-aware logic
       ├── GASegmenterService      → K-means clustering
       ├── ImageRegistrarService   → ORB feature matching + RANSAC
       └── CalculatorService       → Distance & progression math
```

---

## API Endpoints

All endpoints are under the `/api` prefix. Interactive documentation is available at `/docs` (Swagger) and `/redoc` (ReDoc) when the server is running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect-disc` | Upload OCT image, returns disc coordinates + 1800 µm reference line |
| `POST` | `/api/detect-fovea` | Upload image + disc coords, returns fovea location + eye side |
| `POST` | `/api/segment-ga` | Upload image, returns GA region contours |
| `POST` | `/api/register-images` | Upload two images + landmarks, returns transformed fovea coords with confidence |
| `POST` | `/api/calculate-distance` | Fovea coords + GA region, returns distance in pixels & microns |
| `POST` | `/api/calculate-progression` | Before/after distances + dates, returns rate & predicted foveal involvement date |
| `GET`  | `/health` | Health check |

### Micron Calculation

The optic disc vertical diameter is used as the anatomical reference:

```
Optic Disc Vertical Diameter ≈ 1800 µm (standard anatomical value)

pixel_to_micron_ratio = 1800 / disc_height_pixels
distance_microns = distance_pixels × pixel_to_micron_ratio
```

---

## Application Workflow

### Per-Image Processing

Each image follows this pipeline independently:

```
Image Upload → Eye Detection (OD/OS) → Optic Disc Detection (RETFound U-Net)
  → Fovea Detection (green line / geometric fallback) → User confirms fovea
  → GA Segmentation (K-means) → User selects GA region → Distance measured
```

### Image Registration (Automatic)

When Image 1 fovea is confirmed and Image 2 is uploaded:

```
Extract en-face regions → CLAHE vessel enhancement → ORB keypoint detection
  → BFMatcher + Lowe's ratio test → RANSAC affine transform
  → Transform fovea coordinates → Confidence-based behavior:
     High (≥80%):  Auto-apply registered fovea
     Moderate (40-79%): Show suggestion marker alongside auto-detected
     Failed (<40%):  Silent fallback to independent detection
```

### Progression Analysis (Requires Both Images)

```
Validate same eye (OD/OD or OS/OS)
  → distance_change = distance_before − distance_after
  → rate = distance_change / time_elapsed
  → predicted_date = date_after + (distance_after / rate)
```

---

## Project Structure

```
OCT_Project/
├── src/
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # Application entry point
│   │   ├── models/
│   │   │   └── schemas.py            # Pydantic request/response models
│   │   ├── routes/
│   │   │   ├── calculations.py       # Distance & progression endpoints
│   │   │   ├── disc_detection.py     # Optic disc detection endpoint
│   │   │   ├── fovea_detection.py    # Fovea detection endpoint
│   │   │   ├── ga_segmentation.py    # GA segmentation endpoint
│   │   │   └── registration.py       # Image registration endpoint
│   │   └── services/
│   │       ├── calculator.py         # Distance & progression logic
│   │       ├── disc_detector.py      # RETFound U-Net disc detection
│   │       ├── fovea_detector.py     # Anatomy-aware fovea detection
│   │       ├── ga_segmenter.py       # K-means GA segmentation
│   │       └── image_registrar.py    # ORB/RANSAC image registration
│   │
│   ├── frontend/                     # React + TypeScript frontend
│   │   ├── src/
│   │   │   ├── App.tsx               # Main application component
│   │   │   ├── components/
│   │   │   │   ├── ErrorBoundary.tsx  # Error handling wrapper
│   │   │   │   ├── ImageCanvas.tsx    # Canvas rendering & interaction
│   │   │   │   ├── ImageUpload.tsx    # Drag-and-drop upload
│   │   │   │   └── ResultsPanel.tsx   # Results display
│   │   │   ├── services/api.ts       # API client (Axios)
│   │   │   ├── types/api.ts          # TypeScript interfaces
│   │   │   └── utils/errorHandling.ts
│   │   ├── package.json
│   │   └── vite.config.ts
│   │
│   ├── models/
│   │   └── retfound_unet.py          # RETFound U-Net architecture
│   │
│   └── utils/
│       └── image_utils.py            # Image splitting, detection utilities
│
├── weights/
│   └── best_disc_model.pth           # Trained optic disc model
│
├── tests/
│   ├── test_disc_detector.py
│   ├── test_ga_local_segmentation.py
│   └── test_image_registration.py
│
├── input_images/                      # Sample OCT images
├── requirements.txt
├── setup_environment.sh
├── start_api.sh
├── start_frontend.sh
└── README.md
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Image upload fails | Display error message, allow retry |
| Eye mismatch (OD vs OS) | Block calculation, show warning |
| No GA regions detected | Display message, allow manual retry |
| No progression (same distance) | Display "No progression detected" |
| Negative progression | Display error: check measurements |
| Registration fails | Silent fallback to independent fovea detection |
| Invalid/corrupt image | Display error, request valid image |

---

## Development

### Backend

```bash
source venv/bin/activate

# Run tests
pytest tests/ -v

# Start with auto-reload
python -m uvicorn src.api.main:app --reload --port 8000
```

### Frontend

```bash
cd src/frontend

npm run dev       # Dev server with hot reload
npm run build     # Production build
npm run lint      # Lint check
```

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | `source venv/bin/activate && pip install -r requirements.txt` |
| `FileNotFoundError: weights/best_disc_model.pth` | Obtain trained weights from the research team |
| Frontend can't connect to backend | Ensure backend is running: `curl http://localhost:8000/health` |
| `npm ERR! Cannot find module` | `cd src/frontend && rm -rf node_modules && npm install` |

---

## Medical Constraints

- **Optic disc vertical diameter** = 1800 µm (anatomical standard, used as reference for all measurements)
- **Fovea-disc distance**: Expected 2-3 disc diameters temporal to disc (1.5-4.0 range)
- **Progression validation**: Positive distance change = valid progression; negative = likely measurement error
- **Eye matching**: Both images must be from the same eye (OD/OD or OS/OS)

---

## Security Notes

**Current status: Development only.**

- No authentication
- CORS allows all origins
- No rate limiting

Before production: add auth (JWT/OAuth), configure CORS, add rate limiting, use HTTPS.
