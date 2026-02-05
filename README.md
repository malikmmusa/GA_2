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
- **GA Segmentation**: K-means clustering highlights multiple GA regions
- **Interactive Selection**: Hover to highlight GA regions, click to select
- **Distance Calculation**: Measures fovea-to-GA distance in microns using optic disc as reference
- **Progression Analysis**: Calculates rate of change and predicts foveal involvement date
- **PDF Report**: Downloadable report with annotated images and calculations

---

## Technical Architecture

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Backend** | FastAPI | Production-ready, async support, automatic OpenAPI docs |
| **Frontend** | React + TypeScript | Component-based, production-ready, excellent ecosystem |
| **ML Inference** | PyTorch | Existing RETFound U-Net model |
| **Image Processing** | OpenCV, NumPy, SciPy | Existing pipeline compatibility |
| **PDF Generation** | ReportLab or WeasyPrint | Professional PDF output |
| **Deployment** | Docker + Cloud (TBD) | Portable, scalable |

### Key Files & Resources

| File | Purpose |
|------|---------|
| `src/run_interface.py` | Main web application entry point (TO BE CREATED) |
| `src/run_analysis.py` | GA segmentation logic (K-means clustering) |
| `src/run_inference.py` | Optic disc detection pipeline |
| `src/models/retfound_unet.py` | RETFound U-Net model architecture |
| `src/utils/image_utils.py` | Image splitting, fovea detection utilities |
| `weights/best_disc_model.pth` | Trained optic disc detection weights |

---

## Application Workflow

### Per-Image Processing (Independent)

Each image follows this workflow independently. Users can complete one image fully or switch between images at any point.

```
┌─────────────────────────────────────────────────────────────────┐
│                         IMAGE UPLOAD                            │
│  User uploads OCT image (drag & drop or file picker)            │
│  User selects date via date picker                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EYE DETECTION                             │
│  System auto-detects OD/OS based on optic disc position         │
│  Manual dropdown override available if detection is wrong       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIC DISC DETECTION                         │
│  RETFound U-Net generates heatmap                               │
│  Red vertical line drawn (top to bottom of disc)                │
│  Line length = 1800 microns (anatomical reference)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FOVEA DETECTION                             │
│  Anatomy-aware auto-detection (green line or geometric)         │
│  Green marker displayed on image                                │
│  User can CLICK to move fovea location                          │
│  User confirms with "Confirm Location" button OR Enter key      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GA SEGMENTATION                            │
│  K-means clustering identifies potential GA regions             │
│  Multiple regions displayed with distinct outlines              │
│  HOVER over region → highlight effect                           │
│  CLICK on region → select it (visual confirmation)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DISTANCE MEASUREMENT                          │
│  Line drawn from fovea to NEAREST EDGE of selected GA           │
│  Distance calculated in pixels                                  │
│  Converted to microns: (pixels / disc_line_pixels) × 1800       │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison & Prediction (Requires Both Images)

Once both images are fully processed:

```
┌─────────────────────────────────────────────────────────────────┐
│                      VALIDATION                                 │
│  Check both images are from SAME EYE (OD/OD or OS/OS)           │
│  If mismatch → Display warning, block calculation               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROGRESSION CALCULATION                       │
│                                                                 │
│  distance_change = distance_before - distance_after             │
│  time_elapsed = date_after - date_before                        │
│                                                                 │
│  IF distance_change > 0:                                        │
│     rate = distance_change / time_elapsed (microns/day)         │
│     predicted_date = date_after + (distance_after / rate)       │
│                                                                 │
│  IF distance_change == 0:                                       │
│     Display "No progression detected"                           │
│                                                                 │
│  IF distance_change < 0:                                        │
│     Display "Error: Negative progression detected"              │
│     (GA appears further from fovea - likely measurement error)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RESULTS DISPLAY                            │
│  Show both annotated images side by side                        │
│  Display: dates, distances, rate of change, predicted date      │
│  "Download PDF Report" button                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## UI Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ATROPHY ADVISOR                                  │
│                        [Placeholder Logo]                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐      │
│  │      IMAGE 1 (BEFORE)       │    │      IMAGE 2 (AFTER)        │      │
│  │                             │    │                             │      │
│  │  [Drag & Drop / Upload]     │    │  [Drag & Drop / Upload]     │      │
│  │                             │    │                             │      │
│  │  Date: [____Date Picker____]│    │  Date: [____Date Picker____]│      │
│  │  Eye:  [Auto: OD ▼]         │    │  Eye:  [Auto: OS ▼]         │      │
│  │                             │    │                             │      │
│  │  ┌─────────────────────┐    │    │  ┌─────────────────────┐    │      │
│  │  │                     │    │    │  │                     │    │      │
│  │  │   [OCT Image with   │    │    │  │   [OCT Image with   │    │      │
│  │  │    annotations]     │    │    │  │    annotations]     │    │      │
│  │  │                     │    │    │  │                     │    │      │
│  │  │  • Red line (disc)  │    │    │  │  • Red line (disc)  │    │      │
│  │  │  • Green dot (fovea)│    │    │  │  • Green dot (fovea)│    │      │
│  │  │  • Yellow GA regions│    │    │  │  • Yellow GA regions│    │      │
│  │  │  • Cyan measurement │    │    │  │  • Cyan measurement │    │      │
│  │  │                     │    │    │  │                     │    │      │
│  │  └─────────────────────┘    │    │  └─────────────────────┘    │      │
│  │                             │    │                             │      │
│  │  Status: Fovea confirmed    │    │  Status: Select GA region   │      │
│  │  Distance: 342 μm           │    │  Distance: --               │      │
│  │                             │    │                             │      │
│  │  [Confirm Fovea Location]   │    │  [Confirm Fovea Location]   │      │
│  └─────────────────────────────┘    └─────────────────────────────┘      │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                           RESULTS                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  Before Date:     01/15/2025         After Date:      07/20/2025   │  │
│  │  Before Distance: 542 μm             After Distance:  342 μm       │  │
│  │                                                                    │  │
│  │  Time Elapsed:         186 days                                    │  │
│  │  Distance Change:      200 μm                                      │  │
│  │  Rate of Progression:  1.08 μm/day (32.3 μm/month)                 │  │
│  │                                                                    │  │
│  │  ⚠️  PREDICTED FOVEAL INVOLVEMENT: March 28, 2026                   │  │
│  │                                                                    │  │
│  │                      [Download PDF Report]                         │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## PDF Report Contents

The downloadable PDF report includes:

1. **Header**
   - "Atrophy Advisor" title/logo (placeholder)
   - Report generation date

2. **Image Section**
   - Before image (annotated) with date
   - After image (annotated) with date
   - Annotations include: red optic disc line, green fovea marker, yellow GA outline, cyan measurement line

3. **Measurements**
   - Eye: OD or OS
   - Before distance: X μm
   - After distance: Y μm

4. **Analysis**
   - Time between images
   - Rate of progression (μm/day and μm/month)
   - Predicted date of foveal involvement (or "No progression" / "Error" message)

---

## Micron Calculation Reference

The optic disc vertical diameter is used as the anatomical reference:

```
Optic Disc Vertical Diameter ≈ 1800 μm (standard anatomical value)

pixel_to_micron_ratio = 1800 / disc_line_length_in_pixels

fovea_to_ga_distance_microns = fovea_to_ga_distance_pixels × pixel_to_micron_ratio
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Image upload fails | Display error message, allow retry |
| Eye mismatch (OD vs OS) | Block calculation, show warning |
| No GA regions detected | Display message, allow manual retry |
| No progression (same distance) | Display "No progression detected" |
| Negative progression | Display "Error: Negative progression detected (check measurements)" |
| Invalid/corrupt image | Display error, request valid image |

---

## File Structure (Planned)

```
OCT_Project/
├── src/
│   ├── api/                      # FastAPI backend
│   │   ├── main.py               # API entry point
│   │   ├── routes/
│   │   │   ├── upload.py         # Image upload endpoints
│   │   │   ├── analysis.py       # Analysis endpoints
│   │   │   └── report.py         # PDF generation endpoint
│   │   └── services/
│   │       ├── disc_detector.py  # Optic disc detection service
│   │       ├── fovea_detector.py # Fovea detection service
│   │       ├── ga_segmenter.py   # GA segmentation service
│   │       └── calculator.py     # Distance & progression calculator
│   │
│   ├── frontend/                 # React frontend
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── ImageUpload.tsx
│   │   │   │   ├── ImageCanvas.tsx
│   │   │   │   ├── ResultsPanel.tsx
│   │   │   │   └── PDFDownload.tsx
│   │   │   ├── hooks/
│   │   │   └── App.tsx
│   │   └── package.json
│   │
│   ├── models/                   # ML models
│   │   ├── __init__.py
│   │   └── retfound_unet.py
│   │
│   ├── utils/                    # Shared utilities
│   │   ├── __init__.py
│   │   └── image_utils.py
│   │
│   ├── run_analysis.py           # Legacy/reference (GA segmentation)
│   └── run_inference.py          # Legacy/reference (disc detection)
│
├── weights/
│   └── best_disc_model.pth
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Development Phases

### Phase 1: Backend API
- [ ] Set up FastAPI project structure
- [ ] Create image upload endpoint
- [ ] Integrate optic disc detection (from run_inference.py)
- [ ] Integrate fovea detection (from image_utils.py)
- [ ] Integrate GA segmentation (from run_analysis.py)
- [ ] Create distance calculation service
- [ ] Create progression calculation service
- [ ] Create PDF generation endpoint

### Phase 2: Frontend UI
- [ ] Set up React project with TypeScript
- [ ] Create dual image upload component
- [ ] Create interactive image canvas (click handling, hover effects)
- [ ] Create date picker integration
- [ ] Create eye selection dropdown
- [ ] Create results display panel
- [ ] Create PDF download button

### Phase 3: Integration & Polish
- [ ] Connect frontend to backend API
- [ ] Add keyboard shortcuts (Enter to confirm)
- [ ] Add loading states and progress indicators
- [ ] Error handling and user feedback
- [ ] Responsive design adjustments

### Phase 4: Deployment
- [ ] Create Dockerfile
- [ ] Set up CI/CD pipeline
- [ ] Deploy to cloud platform (TBD)
- [ ] Performance optimization

---

## Running the Application (Future)

```bash
# Backend
cd src/api
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd src/frontend
npm install
npm run dev
```

---

## Legacy Scripts Reference

The following existing scripts contain logic that will be integrated:

- **`src/run_inference.py`**: Optic disc detection pipeline using RETFound U-Net
- **`src/run_analysis.py`**: GA segmentation using K-means clustering (`segment_macular_ga_kmeans`)
- **`src/utils/image_utils.py`**: Image splitting, fovea detection, manual adjustment UI
