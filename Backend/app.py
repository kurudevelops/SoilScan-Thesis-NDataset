#!/usr/bin/env python3
"""
FastAPI Backend for SoilScan.
"""
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict_api import SoilScanPredictor

app = FastAPI(title="SoilScan NPK API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None


@app.on_event("startup")
def load_models():
    global predictor
    model_file = Path("models_out/production_stack.joblib")
    if not model_file.exists():
        raise RuntimeError("Missing models_out/production_stack.joblib! Run export_models.py first.")
    predictor = SoilScanPredictor(model_path=model_file)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/scan")
async def analyze_soil(file: UploadFile = File(...)):
    suffix = Path(file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        res = predictor.predict_image(tmp_path)
        if res.get("status") == "error":
            raise HTTPException(status_code=422, detail=res)
        return res
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)