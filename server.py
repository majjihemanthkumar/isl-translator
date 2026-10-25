"""
ISL Real-Time Translator - Local Server
Serves static assets with correct MIME types for WebAssembly & ONNX,
and provides a local Python ONNX inference fallback endpoint (/api/predict)
using the local isl_model.onnx model.
"""

import sys
import os
import json
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Register proper MIME types for WASM, JavaScript, and ML model binary assets
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('application/octet-stream', '.onnx')
mimetypes.add_type('application/octet-stream', '.tflite')
mimetypes.add_type('application/octet-stream', '.data')
mimetypes.add_type('application/x-protobuf', '.binarypb')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'isl_model.onnx')
LABEL_MAP_PATH = os.path.join(BASE_DIR, 'processed_data', 'label_map.json')

ort_session = None
id_to_word = {}
input_name = 'float_input'
last_model_mtime = 0

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    np = None
    ort = None

def get_ort_session():
    """Lazily load or reload model and label map if updated on disk."""
    global ort_session, id_to_word, input_name, last_model_mtime

    if ort is None:
        return None

    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
            return ort_session

        cur_mtime = os.path.getmtime(MODEL_PATH)
        if ort_session is None or cur_mtime != last_model_mtime:
            with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
                lmap = json.load(f)
            id_to_word = {int(v): k for k, v in lmap.items()}

            ort_session = ort.InferenceSession(MODEL_PATH)
            input_name = ort_session.get_inputs()[0].name
            last_model_mtime = cur_mtime
            print(f"[OK] Loaded updated isl_model.onnx ({len(id_to_word)} classes, input: '{input_name}')")
    except Exception as e:
        print(f"[WARN] Error updating ONNX session: {e}")

    return ort_session


class ISLRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and disable caching for reliable local reloading
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return

        if self.path == '/api/status':
            session = get_ort_session()
            status_data = {
                'status': 'online',
                'model': 'isl_model.onnx',
                'python_ort_ready': session is not None,
                'classes': len(id_to_word)
            }
            body = json.dumps(status_data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        super().do_GET()

    def do_POST(self):
        if self.path == '/api/predict':
            session = get_ort_session()
            if session is None:
                err_resp = json.dumps({'error': 'Local Python ONNX engine unavailable'}).encode('utf-8')
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(err_resp)))
                self.end_headers()
                self.wfile.write(err_resp)
                return

            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))
                features = data.get('features', [])

                if len(features) != 234:
                    features = (features + [0.0] * 234)[:234]

                # Hand Activity Gate: If left hand and right hand (indices 99 to 224) are all 0.0,
                # the user is in an idle resting state. Do NOT classify noise into random words!
                has_hand_activity = any(abs(f) > 1e-4 for f in features[99:225])
                if not has_hand_activity:
                    idle_resp = json.dumps({
                        'word': '--',
                        'confidence': 0.0,
                        'classId': -1,
                        'source': 'idle'
                    }).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(idle_resp)))
                    self.end_headers()
                    self.wfile.write(idle_resp)
                    return

                features_np = np.array([features], dtype=np.float32)
                outputs = session.run(None, {input_name: features_np})

                class_id = int(outputs[0][0])
                probs = outputs[1][0] if len(outputs) > 1 else []
                conf = float(probs[class_id]) if len(probs) > class_id else 0.90
                word = id_to_word.get(class_id, f"Class {class_id}")

                res_obj = {
                    'word': word,
                    'confidence': min(max(conf, 0.01), 0.999),
                    'classId': class_id,
                    'source': 'python_local_onnx'
                }
                res_bytes = json.dumps(res_obj).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)
                return

            except Exception as e:
                err_resp = json.dumps({'error': str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(err_resp)))
                self.end_headers()
                self.wfile.write(err_resp)
                return

        self.send_response(404)
        self.end_headers()


def run(port=8000):
    server_address = ('', port)
    attempts = 0
    httpd = None
    while attempts < 10:
        try:
            httpd = HTTPServer(server_address, ISLRequestHandler)
            break
        except OSError:
            print(f"[INFO] Port {port} is busy. Trying port {port + 1}...")
            port = port + 1
            server_address = ('', port)
            attempts += 1

    if not httpd:
        print("[ERROR] Could not bind to an available port.")
        sys.exit(1)

    print("=" * 60)
    print(f"  ISL Real-Time Translator Local Server")
    print(f"  Serving at: http://localhost:{port}")
    print(f"  Using local model: {MODEL_PATH}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)