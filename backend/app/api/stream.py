"""
MJPEG Stream API Router

Provides a live MJPEG video stream feed endpoint for browser displays (<img src="/api/stream" />).
Supports single_frame query parameter for instant response verification in unit tests.
"""

import asyncio
import numpy as np
import cv2
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app import config
from app.api.recognition import global_engine


router = APIRouter(prefix="/stream", tags=["stream"])


async def generate_mjpeg_stream(request: Request, single_frame: bool = False):
    """Async generator yielding JPEG frames formatted as a multipart stream."""
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "OFFLINE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    _, dummy_buffer = cv2.imencode(".jpg", dummy_img)
    dummy_bytes = dummy_buffer.tobytes()

    while not await request.is_disconnected():
        frame = global_engine.get_frame()
        frame_bytes = dummy_bytes
        if frame is not None:
            success, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if success:
                frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        if single_frame:
            break
        await asyncio.sleep(0.03)


@router.get("")
async def video_stream_endpoint(request: Request, single_frame: bool = False):
    """Returns a live MJPEG video stream feed."""
    return StreamingResponse(
        generate_mjpeg_stream(request, single_frame=single_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
