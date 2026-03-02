"""Shared dependency providers for API routes."""

from functools import lru_cache
from typing import TYPE_CHECKING

from .services.calculator import DistanceCalculatorService, ProgressionCalculatorService
from .services.ga_segmenter import GASegmenterService
from .services.image_registrar import ImageRegistrarService

if TYPE_CHECKING:
    from .services.disc_detector import DiscDetectorService
    from .services.fovea_detector import FoveaDetectorService


@lru_cache(maxsize=1)
def get_disc_detector() -> "DiscDetectorService":
    from .services.disc_detector import DiscDetectorService

    return DiscDetectorService()


@lru_cache(maxsize=1)
def get_fovea_detector() -> "FoveaDetectorService":
    from .services.fovea_detector import FoveaDetectorService

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
