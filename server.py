#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from typing import List, Dict, Any

# Import tree generator
from generate_trees import generate_trees
import uuid
from datetime import datetime

# Lazy PDF support. We'll try to import at runtime if not available at startup.
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # handled via ensure_pdf_support()

def ensure_pdf_support() -> bool:
    global PdfReader
    if PdfReader is not None:
        return True
    try:
        from PyPDF2 import PdfReader as _PdfReader  # type: ignore
        PdfReader = _PdfReader
        return True
    except Exception:
        return False

app = Flask(__name__, static_folder='.', static_url_path='')

UPLOAD_FOLDER = 'uploads'
TREES_FOLDER = 'trees'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TREES_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return send_from_directory('.', 'annotation_tool.html')


@app.post('/api/upload')
def upload():
    """Accept a JSON file containing a list of flattened paper records.
    Writes trees to trees/tree_<PAPER_ID>.json and refreshes papers_summary.json.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return jsonify({'error': 'JSON must be a list of records'}), 400

        generated = generate_trees(data, output_dir=TREES_FOLDER)

        # Update papers_summary.json
        summary_file = os.path.join(TREES_FOLDER, 'papers_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as sf:
            json.dump(generated, sf, indent=2, ensure_ascii=False)

        return jsonify({'ok': True, 'generated': generated}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.post('/api/upload_pdf')
def upload_pdf():
    """Accept a single PDF file and create a basic tree from its contents.
    This is a lightweight fallback that extracts the first page text for title/description.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please upload a PDF file'}), 400

    if not ensure_pdf_support():
        return jsonify({'error': 'PDF support not available. Install PyPDF2 in the server environment.'}), 500

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    try:
        reader = PdfReader(path)
        doc_info_title = None
        try:
            meta = reader.metadata
            if meta and getattr(meta, 'title', None):
                doc_info_title = str(meta.title)
        except Exception:
            pass

        first_page_text = ''
        try:
            if len(reader.pages) > 0:
                first_page_text = reader.pages[0].extract_text() or ''
        except Exception:
            pass

        # Heuristic title
        title = doc_info_title or (first_page_text.split('\n', 1)[0].strip() if first_page_text else os.path.splitext(filename)[0])
        description = (first_page_text or '')[:600]

        paper_id = uuid.uuid4().hex[:24]
        created_at = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # Minimal flattened record compatible with generator
        record = {
            'paper title': title,
            'paper experiments domain': 'unspecified',
            'paper experiments description': description or 'PDF imported without extracted abstract.',
            'paper experiments code_location': '',
            'created_at': created_at,
            '_version': 1,
            '_is_latest': True,
            '_result_id': uuid.uuid4().hex[:24],
            '_paper_id': paper_id,
        }

        generated = generate_trees([record], output_dir=TREES_FOLDER)

        # Update papers_summary.json
        summary_file = os.path.join(TREES_FOLDER, 'papers_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as sf:
            json.dump(generated, sf, indent=2, ensure_ascii=False)

        return jsonify({'ok': True, 'generated': generated}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port, debug=True)


