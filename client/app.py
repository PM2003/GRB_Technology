"""
Flask web client for GRB_Technology Virtual Try-On.
Sends uploaded images to the inference server and renders the result.
"""

import os
import base64
import io
from flask import Flask, request, render_template
from PIL import Image
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

SERVER_URL = os.environ.get('INFERENCE_SERVER_URL', 'http://localhost:8000/api/transform')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tryon', methods=['POST'])
def tryon():
    person_file = request.files.get('person')
    cloth_file  = request.files.get('cloth')

    if not person_file or not cloth_file:
        return render_template('index.html',
            error='Please upload both a person photo and a clothing item.')

    if person_file.filename == '' or cloth_file.filename == '':
        return render_template('index.html', error='Both files must be selected.')

    try:
        resp = requests.post(
            SERVER_URL,
            files={
                'model': (person_file.filename, person_file.stream, person_file.content_type),
                'cloth': (cloth_file.filename,  cloth_file.stream,  cloth_file.content_type)
            },
            timeout=120
        )
        resp.raise_for_status()

        result_img = Image.open(io.BytesIO(resp.content))
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        result_b64 = base64.b64encode(buf.getvalue()).decode()
        return render_template('index.html', result=result_b64)

    except requests.exceptions.ConnectionError:
        return render_template('index.html',
            error='Cannot connect to inference server. Is it running?')
    except requests.exceptions.Timeout:
        return render_template('index.html',
            error='Inference timed out. Try again with a smaller image.')
    except Exception as e:
        return render_template('index.html', error=f'Error: {e}')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
