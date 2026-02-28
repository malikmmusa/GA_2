"""Helpers for handling uploaded image files."""

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile


async def decode_uploaded_image(
    file: UploadFile,
    *,
    file_role: str = "image",
) -> np.ndarray:
    """
    Decode an uploaded file into an OpenCV BGR image.

    Raises a 400 HTTPException when decoding fails.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {file_role} file. Uploaded file is empty.",
        )

    nparr = np.frombuffer(contents, np.uint8)
    if nparr.size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {file_role} file. Uploaded file contains no data.",
        )

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {file_role} file. Could not decode image.",
        )

    return image
