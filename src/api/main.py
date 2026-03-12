"""Main FastAPI application entry point for Atrophy Advisor."""
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from .constants import API_VERSION
from .models.schemas import HealthStatusResponse, RootStatusResponse
from .routes import calculations, disc_detection, fovea_detection, ga_segmentation, registration, report
from .utils.status import build_status_payload
from .utils.logger import get_logger

logger = get_logger("main")


def _preload_disc_detector() -> None:
    """Pre-warm disc detector (incl. weight download) in a background thread."""
    try:
        from .dependencies import get_disc_detector
        svc = get_disc_detector()
        if svc.model is not None:
            logger.info("Disc detector pre-loaded successfully on %s", svc.device)
        else:
            logger.warning("Disc detector running in fallback mode (no model weights)")
    except Exception as exc:  # pragma: no cover
        logger.warning("Disc detector pre-load failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off weight download + model load before the first request arrives.
    # Runs in a daemon thread so startup is not blocked.
    t = threading.Thread(target=_preload_disc_detector, daemon=True, name="disc-preload")
    t.start()
    logger.info("Disc detector pre-load started in background thread")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Atrophy Advisor API",
    description="OCT image analysis for Geographic Atrophy progression tracking",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROUTERS = (
    (disc_detection.router, "Disc Detection"),
    (fovea_detection.router, "Fovea Detection"),
    (ga_segmentation.router, "GA Segmentation"),
    (calculations.router, "Calculations"),
    (registration.router, "Registration"),
    (report.router, "Report"),
)

for router, tag in ROUTERS:
    app.include_router(router, prefix="/api", tags=[tag])

@app.get("/api/status", response_model=RootStatusResponse)
async def root() -> RootStatusResponse:
    """Root endpoint with API information."""
    return RootStatusResponse(
        **build_status_payload(
            "operational",
            message="Atrophy Advisor API",
            version=API_VERSION,
            docs="/docs",
        )
    )

@app.get("/health", response_model=HealthStatusResponse)
async def health_check() -> HealthStatusResponse:
    """Health check endpoint."""
    return HealthStatusResponse(**build_status_payload("healthy"))

# Serve built frontend static files in production (when /static dir exists)
_static_dir = Path(__file__).parent.parent.parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
