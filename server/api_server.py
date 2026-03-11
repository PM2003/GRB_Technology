"""
FastAPI inference server for GRB_Technology Virtual Try-On.
Accepts person + cloth images via multipart POST and streams back the result PNG.
"""

import io
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn

from server.viton_inference import run_tryon

app = FastAPI(
    title='GRB_Technology Virtual Try-On API',
    description='POST a person photo and clothing item, receive a try-on result PNG.',
    version='1.0.0'
)

ALLOWED = {'image/jpeg', 'image/png', 'image/webp'}


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'GRB_Technology'}


@app.post('/api/transform')
async def transform(
    model: UploadFile = File(..., description='Full-body person photo'),
    cloth: UploadFile = File(..., description='Flat-lay clothing image')
):
    if model.content_type not in ALLOWED:
        raise HTTPException(400, f'Person image must be JPEG/PNG/WEBP')
    if cloth.content_type not in ALLOWED:
        raise HTTPException(400, f'Cloth image must be JPEG/PNG/WEBP')

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
