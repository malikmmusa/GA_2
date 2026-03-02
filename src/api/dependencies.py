"""Shared dependency providers for API routes."""

from functools import lru_cache

from .services.calculator import DistanceCalculatorService, ProgressionCalculatorService
from .services.disc_detector import DiscDetectorService
from .services.fovea_detector import FoveaDetectorService
from .services.ga_segmenter import GASegmenterService
from .services.image_registrar import ImageRegistrarService


@lru_cache(maxsize=1)
def get_disc_detector() -> DiscDetectorService:
    return DiscDetectorService()


@lru_cache(maxsize=1)
def get_fovea_detector() -> FoveaDetectorService:
    return FoveaDetectorService()


@lru_cache(maxsize=1)
def get_ga_segmenter() -> GASegmenterService:
    return GASegmenterService(use_sam=True)


@lru_cache(maxsize=1)
def get_distance_calculator() -> DistanceCalculatorService:
    return DistanceCalculatorService()


@lru_cache(maxsize=1)
def get_progression_calculator() -> ProgressionCalculatorService:
    return ProgressionCalculatorService()


@lru_cache(maxsize=1)
def get_registrar() -> ImageRegistrarService:
    return ImageRegistrarService()
