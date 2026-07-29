import os
import joblib
import numpy as np

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

MODEL_PATH = os.path.join(PROCESSED_DIR, "isl_model.pkl")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.pkl")
ONNX_PATH = os.path.join(BASE_DIR, "isl_model.onnx")

print("=" * 65)
print("EXPORTING SKLEARN PIPELINE TO ONNX (zipmap=False for clean Float32 array)")
print("=" * 65)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Combine Scaler + Classifier into a single Scikit-Learn Pipeline
pipe = Pipeline([
    ('scaler', scaler),
    ('model', model)
])

# Define input shape: [None, 234] float32 array
initial_type = [('float_input', FloatTensorType([None, 234]))]

# Disable ZipMap so probability output is a clean Float32 Tensor [1, num_classes]
options = {id(model): {'zipmap': False}}

print("Converting Pipeline to ONNX format...")
onx = convert_sklearn(pipe, initial_types=initial_type, target_opset=12, options=options)

with open(ONNX_PATH, "wb") as f:
    f.write(onx.SerializeToString())

print(f"ONNX model successfully saved to: {ONNX_PATH}")
print(f"File size: {os.path.getsize(ONNX_PATH) / (1024 * 1024):.2f} MB")
print("=" * 65)