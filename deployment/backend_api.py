
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import tempfile
import traceback
import importlib.util
import sys

app = FastAPI()

# Get absolute project root path
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Paths where models exist (absolute paths)
ML_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ml")
CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn", "cnn_model.keras")

# ================================================================
# Load inference module directly from src/inference.py
# ================================================================
def load_inference_module():
    base_dir = PROJECT_ROOT
    # Ensure `src` is on sys.path so imports inside `src/inference.py`
    # like `from feature_engineering import ...` can be resolved.
    src_dir = os.path.join(base_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    src_path = os.path.join(base_dir, "src", "inference.py")

    spec = importlib.util.spec_from_file_location("inference", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Preload inference module at import time to avoid heavy per-request imports
# (TensorFlow + model loading) causing the API request to exceed frontend timeouts.
INFERENCE_MODULE = None
try:
    INFERENCE_MODULE = load_inference_module()
except Exception:
    # Defer raising until a request if import fails; keep server up so errors are visible.
    INFERENCE_MODULE = None
else:
    # Try to initialize/load heavy models at startup to avoid per-request delays
    try:
        if hasattr(INFERENCE_MODULE, 'init_models'):
            INFERENCE_MODULE.init_models(ML_MODEL_DIR, CNN_MODEL_PATH)
    except Exception:
        # keep server up; errors will be returned on requests
        pass

# ================================================================
# API ENDPOINT
# ================================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    try:
        # Save uploaded file to temp
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        # Use preloaded inference module when available (faster)
        inference = INFERENCE_MODULE or load_inference_module()

        # Run prediction
        prob, feature_dict = inference.predict(tmp_path, ML_MODEL_DIR, CNN_MODEL_PATH)

        label = 1 if prob >= 0.5 else 0

        result = {
            "prediction": int(label),
            "confidence": float(prob),
            "top_features": feature_dict,
        }

        return JSONResponse(content=result)

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": tb}
        )

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ================================================================