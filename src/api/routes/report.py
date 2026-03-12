"""PDF report generation endpoint."""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
import io
from typing import Optional

from ..services.report_generator import ReportGeneratorService
from ..utils.errors import route_error_handler

router = APIRouter()

_report_service = ReportGeneratorService()


@router.post("/generate-report")
@route_error_handler("Report generation")
async def generate_report(
    image_before: UploadFile = File(...),
    image_after: UploadFile = File(...),
    date_before: str = Form(...),
    date_after: str = Form(...),
    eye_side: str = Form(...),
    distance_before_microns: float = Form(...),
    distance_after_microns: float = Form(...),
    days_elapsed: int = Form(...),
    distance_change_microns: float = Form(...),
    status: str = Form(...),
    rate_microns_per_year: Optional[float] = Form(None),
    rate_microns_per_month: Optional[float] = Form(None),
    years_until_involvement: Optional[float] = Form(None),
    predicted_foveal_involvement_date: Optional[str] = Form(None),
) -> StreamingResponse:
    """
    Generate a PDF progression report combining both images and analysis results.

    Returns a downloadable PDF file.
    """
    before_bytes = await image_before.read()
    after_bytes = await image_after.read()

    pdf_bytes = _report_service.generate(
        image_before_bytes=before_bytes,
        image_after_bytes=after_bytes,
        date_before=date_before,
        date_after=date_after,
        eye_side=eye_side,
        distance_before_microns=distance_before_microns,
        distance_after_microns=distance_after_microns,
        days_elapsed=days_elapsed,
        distance_change_microns=distance_change_microns,
        rate_microns_per_year=rate_microns_per_year,
        rate_microns_per_month=rate_microns_per_month,
        years_until_involvement=years_until_involvement,
        predicted_foveal_involvement_date=predicted_foveal_involvement_date,
        status=status,
    )

    filename = f"ga_progression_report_{date_before}_to_{date_after}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
