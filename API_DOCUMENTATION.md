# Atrophy Advisor API Documentation

## Overview

The Atrophy Advisor API provides endpoints for OCT image analysis to track Geographic Atrophy (GA) progression and predict foveal involvement.

**Base URL**: `http://localhost:8000/api`

**Interactive Documentation**: `http://localhost:8000/docs`

---

## Authentication

Currently, no authentication is required. This will be added in a future phase.

---

## Endpoints

### 1. Optic Disc Detection

**Endpoint**: `POST /api/detect-disc`

**Description**: Detects the optic disc in a composite OCT image and returns a vertical line representing 1800 microns (the standard disc diameter). This line serves as the anatomical reference for all distance measurements.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: Image file (PNG, JPG, etc.)

**Response**:
```json
{
  "disc_center_x": 1250.5,
  "disc_center_y": 512.0,
  "disc_top_y": 412.5,
  "disc_bottom_y": 611.5,
  "disc_height_pixels": 199.0,
  "pixel_to_micron_ratio": 9.045,
  "en_face_split_x": 850
}
```

**Notes**:
- All coordinates are in original image space
- The vertical line from `disc_top_y` to `disc_bottom_y` represents exactly 1800 microns
- Uses the "New Algorithm" (Contour/Energy method) from the legacy codebase

---

### 2. Fovea Detection

**Endpoint**: `POST /api/detect-fovea`

**Description**: Detects the fovea location using anatomy-aware methods (green line anchor or geometric fallback based on disc position).

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: Image file (PNG, JPG, etc.)
  - `request_data`: JSON object with:
    ```json
    {
      "disc_center_x": 1250.5,
      "disc_center_y": 512.0,
      "disc_height_pixels": 199.0,
      "en_face_split_x": 850,
      "use_manual_adjustment": false
    }
    ```

**Response**:
```json
{
  "fovea_x": 2150.3,
  "fovea_y": 542.8,
  "detection_method": "green_line",
  "eye_side": "OD"
}
```

**Detection Methods**:
- `green_line`: Precise detection using scan line marker
- `geometric_fallback`: Estimated based on disc position (2.5 disc diameters temporal, 0.15 inferior)
- `manual`: User-adjusted via interactive UI

**Eye Side**:
- `OD`: Right eye (Oculus Dexter)
- `OS`: Left eye (Oculus Sinister)

---

### 3. GA Segmentation

**Endpoint**: `POST /api/segment-ga`

**Description**: Segments Geographic Atrophy (GA) regions using K-means clustering with anatomical constraints.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Query Parameters** (all optional):
  - `disc_center_x`: float
  - `disc_center_y`: float
  - `disc_height_pixels`: float
  - `en_face_split_x`: int
- **Body**:
  - `file`: Image file (PNG, JPG, etc.)

**Response**:
```json
{
  "regions": [
    [
      [1050, 450],
      [1052, 448],
      ...
    ],
    [
      [1200, 500],
      [1202, 498],
      ...
    ]
  ],
  "region_count": 2
}
```

**Algorithm**:
1. CLAHE contrast enhancement
2. K-means clustering (3 clusters, brightest = GA)
3. Morphological cleanup (opening + closing)
4. Filtering:
   - Minimum area: 500 pixels
   - Maximum circularity: 0.8 (reject circular structures)
   - Border rejection: regions touching edges are excluded
5. Size-based prioritization: Keep regions ≥ 20% of largest
6. Limit to top 3 regions

---

### 4. Distance Calculation

**Endpoint**: `POST /api/calculate-distance`

**Description**: Calculates the shortest distance from the fovea to a selected GA region edge.

**Request**:
```json
{
  "fovea_x": 2150.3,
  "fovea_y": 542.8,
  "selected_ga_region_index": 0,
  "ga_regions": [
    [[1050, 450], [1052, 448], ...],
    [[1200, 500], [1202, 498], ...]
  ],
  "pixel_to_micron_ratio": 9.045
}
```

**Response**:
```json
{
  "distance_pixels": 245.6,
  "distance_microns": 2221.0,
  "nearest_ga_point_x": 1100,
  "nearest_ga_point_y": 460
}
```

**Formula**:
```
distance_microns = distance_pixels × pixel_to_micron_ratio
where pixel_to_micron_ratio = 1800 / disc_height_pixels
```

---

### 5. Progression Calculation

**Endpoint**: `POST /api/calculate-progression`

**Description**: Calculates the rate of GA progression and predicts when GA will reach the fovea.

**Request**:
```json
{
  "date_before": "2025-01-15",
  "date_after": "2025-07-20",
  "distance_before_microns": 2542.0,
  "distance_after_microns": 2342.0,
  "eye_side_before": "OD",
  "eye_side_after": "OD"
}
```

**Response (Progression Detected)**:
```json
{
  "status": "progression",
  "error_message": null,
  "days_elapsed": 186,
  "distance_change_microns": 200.0,
  "rate_microns_per_day": 1.075,
  "rate_microns_per_month": 32.3,
  "predicted_foveal_involvement_date": "2026-03-28"
}
```

**Response (No Progression)**:
```json
{
  "status": "no_progression",
  "error_message": null,
  "days_elapsed": 186,
  "distance_change_microns": 0.0,
  "rate_microns_per_day": 0.0,
  "rate_microns_per_month": 0.0,
  "predicted_foveal_involvement_date": null
}
```

**Response (Error - Negative Progression)**:
```json
{
  "status": "error",
  "error_message": "Negative progression detected (-50.0 µm). GA appears further from fovea. Check measurements.",
  "days_elapsed": 186,
  "distance_change_microns": -50.0,
  "rate_microns_per_day": null,
  "rate_microns_per_month": null,
  "predicted_foveal_involvement_date": null
}
```

**Response (Error - Eye Mismatch)**:
```json
{
  "status": "error",
  "error_message": "Eye mismatch: Before is OD, After is OS",
  "days_elapsed": 0,
  "distance_change_microns": 0.0,
  "rate_microns_per_day": null,
  "rate_microns_per_month": null,
  "predicted_foveal_involvement_date": null
}
```

**Logic**:
- **Positive `distance_change`**: GA is progressing toward fovea → Calculate rate and predict date
- **Zero `distance_change`**: No progression detected
- **Negative `distance_change`**: Error condition (GA appears further away)

**Validation**:
- Both images must be from the same eye (`eye_side_before` == `eye_side_after`)
- `date_after` must be later than `date_before`

---

## Workflow Example

### Complete Analysis Pipeline

```bash
# 1. Upload image and detect disc
curl -X POST "http://localhost:8000/api/detect-disc" \
  -F "file=@image_before.png" \
  > disc_result.json

# 2. Detect fovea
curl -X POST "http://localhost:8000/api/detect-fovea" \
  -F "file=@image_before.png" \
  -F "request_data=$(cat disc_result.json)" \
  > fovea_result.json

# 3. Segment GA regions
curl -X POST "http://localhost:8000/api/segment-ga?disc_center_x=1250&disc_center_y=512&disc_height_pixels=199&en_face_split_x=850" \
  -F "file=@image_before.png" \
  > ga_result.json

# 4. Calculate distance
curl -X POST "http://localhost:8000/api/calculate-distance" \
  -H "Content-Type: application/json" \
  -d @distance_request.json \
  > distance_result.json

# 5. Repeat steps 1-4 for "after" image

# 6. Calculate progression
curl -X POST "http://localhost:8000/api/calculate-progression" \
  -H "Content-Type: application/json" \
  -d @progression_request.json \
  > progression_result.json
```

---

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input (e.g., corrupted image, invalid parameters)
- `500 Internal Server Error`: Server-side error (e.g., model inference failed)

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Medical Constraints

### Anatomical Reference
- Optic disc vertical diameter = **1800 microns** (standard)
- All distance measurements use this as the reference

### Fovea-Disc Distance
- Expected: 2-3 disc diameters temporal to disc
- Minimum: 1.5 disc diameters
- Maximum: 4.0 disc diameters
- Outside this range triggers a warning

### Progression Validation
- `distance_change > 0`: Valid progression
- `distance_change == 0`: No progression
- `distance_change < 0`: **ERROR** - Likely measurement error

### Eye Side Matching
- Both images in a comparison **must** be from the same eye
- Mixing OD and OS is blocked with an error

---

## Development

### Starting the Server

```bash
# First-time setup
./setup_environment.sh

# Start server
./start_api.sh

# Or manually:
source venv/bin/activate
cd src/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API testing and full schema documentation.

---

## Future Enhancements (Phase 2-4)

- [ ] PDF report generation endpoint
- [ ] Batch processing for multiple images
- [ ] WebSocket support for real-time progress updates
- [ ] Authentication and user management
- [ ] Image storage and retrieval
- [ ] Historical progression tracking
