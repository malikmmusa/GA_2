"""Main FastAPI application entry point for Atrophy Advisor."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routes import disc_detection, fovea_detection, ga_segmentation, calculations, registration

# Initialize FastAPI app
app = FastAPI(
    title="Atrophy Advisor API",
    description="OCT image analysis for Geographic Atrophy progression tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    disc_detection.router,
    prefix="/api",
    tags=["Disc Detection"]
)

app.include_router(
    fovea_detection.router,
    prefix="/api",
    tags=["Fovea Detection"]
)

app.include_router(
    ga_segmentation.router,
    prefix="/api",
    tags=["GA Segmentation"]
)

app.include_router(
    calculations.router,
    prefix="/api",
    tags=["Calculations"]
)

app.include_router(
    registration.router,
    prefix="/api",
    tags=["Registration"]
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Atrophy Advisor API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
