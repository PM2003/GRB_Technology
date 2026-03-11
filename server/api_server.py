"""
FastAPI inference server for GRB_Technology Virtual Try-On.
Uses OpenCV-based cloth warping — no pretrained weights required.
"""

import io
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import cv2
import uvicorn

app = FastAPI(
    title='GRB_Technology Virtual Try-On API',
    description='POST a person photo and clothing item, receive a try-on result PNG.',
    version='2.0.0'
)

ALLOWED = {'image/jpeg', 'image/png', 'image/webp'}


def remove_cloth_background(cloth_arr: np.ndarray) -> np.ndarray:
    """
    Extract clothing foreground mask from a light/white background.
    Returns float32 alpha mask in range [0, 1].
    """
    gray = cv2.cvtColor(cloth_arr, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return mask.astype(np.float32) / 255.0


def estimate_torso_box(h: int, w: int) -> tuple:
    """
    Estimate torso bounding box from image dimensions.
    Returns (x1, y1, x2, y2).
    """
    return (
        int(w * 0.10),
        int(h * 0.17),
        int(w * 0.90),
        int(h * 0.65)
    )


def run_tryon(person_img: Image.Image, cloth_img: Image.Image) -> Image.Image:
    """
    Composite the clothing item onto the person's torso region.
    Works entirely with OpenCV — no deep learning weights needed.
    """
    # Standardise resolution
    person_img = person_img.resize((384, 512), Image.LANCZOS)
    person = np.array(person_img.convert('RGB'))
    cloth  = np.array(cloth_img.convert('RGB'))

    h, w = person.shape[:2]
    x1, y1, x2, y2 = estimate_torso_box(h, w)
    rw, rh = x2 - x1, y2 - y1

    # Resize cloth to torso region
    cloth_r = cv2.resize(cloth, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

    # Build soft alpha mask
    alpha = remove_cloth_background(cloth_r)[:, :, np.newaxis]  # [H, W, 1]

    # Alpha-blend cloth onto person
    result = person.copy().astype(np.float32)
    region = result[y1:y2, x1:x2]
    region[:] = cloth_r.astype(np.float32) * alpha + region * (1.0 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'GRB_Technology v2'}


@app.post('/api/transform')
async def transform(
    model: UploadFile = File(..., description='Full-body person photo'),
    cloth: UploadFile = File(..., description='Clothing item on white background')
):
    if model.content_type not in ALLOWED:
        raise HTTPException(400, 'Person image must be JPEG/PNG/WEBP')
    if cloth.content_type not in ALLOWED:
        raise HTTPException(400, 'Cloth image must be JPEG/PNG/WEBP')

    try:
        person_img = Image.open(io.BytesIO(await model.read())).convert('RGB')
        cloth_img  = Image.open(io.BytesIO(await cloth.read())).convert('RGB')

        result = run_tryon(person_img, cloth_img)

        buf = io.BytesIO()
        result.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/png')

    except Exception as exc:
        raise HTTPException(500, f'Inference failed: {exc}')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run('server.api_server:app', host='0.0.0.0', port=port, reload=False)
