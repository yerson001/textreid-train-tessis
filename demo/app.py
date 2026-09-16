"""
Demo web - Flask: sube una imagen con varias personas + prompt textual y la app
devuelve la imagen anotada con el ranking de quien matchea la descripcion.

Uso:
    .venv/bin/python demo/app.py            # -> http://localhost:5001
"""

import os
import sys
import uuid
import base64
import threading

from flask import Flask, jsonify, render_template, request, url_for

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'demo'))

from reid_pipeline import ReIDPipeline

CHECKPOINT = os.path.join(_REPO_ROOT, 'data', 'checkpoints', 'TextReIDNet_latest.pth.tar')
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ReIDPipeline(checkpoint=CHECKPOINT)
    return _pipeline


@app.route('/')
def index():
    return render_template('index.html', ckpt_found=os.path.isfile(CHECKPOINT))


@app.route('/api/search', methods=['POST'])
def api_search():
    """JSON puro (estilo core.py original): {image_b64, results}."""
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    file_bytes = bytes(data.get('image_bytes', []))
    if not text or not file_bytes:
        return jsonify({'error': 'se requieren "text" y "image_bytes"'}), 400

    import io
    from PIL import Image
    image = Image.open(io.BytesIO(file_bytes))
    pipeline = get_pipeline()
    annotated, results = pipeline.do_reid(image, text)
    buf = io.BytesIO()
    annotated.save(buf, format='JPEG', quality=92)
    return jsonify({
        'image_b64': base64.b64encode(buf.getvalue()).decode(),
        'results': results,
        'checkpoint_epoch': pipeline.checkpoint_epoch,
    })


@app.route('/search', methods=['POST'])
def search():
    """Render web: imagen anotada + tabla de resultados."""
    text = request.form.get('text', '').strip()
    upload = request.files.get('image')
    if not text or upload is None:
        return render_template('index.html', error='Sube una imagen y escribe el prompt.')

    uid = uuid.uuid4().hex
    ext = os.path.splitext(upload.filename or '')[-1] or '.jpg'
    in_path = os.path.join(UPLOAD_DIR, f'input_{uid}{ext}')
    upload.save(in_path)

    pipeline = get_pipeline()
    annotated, results = pipeline.do_reid(in_path, text)
    out_name = f'result_{uid}.jpg'
    annotated.save(os.path.join(STATIC_DIR, out_name), format='JPEG', quality=92)

    return render_template(
        'results.html',
        image_url=url_for('static', filename=out_name),
        text=text,
        results=results,
        detected=len(results),
        ckpt_epoch=pipeline.checkpoint_epoch,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'Demo TextReIDNet -> http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)