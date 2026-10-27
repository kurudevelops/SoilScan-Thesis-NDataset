#!/usr/bin/env python3
"""
Single-image inference engine for SoilScan FastAPI backend.
Matched to the exact filenames inside Backend/.
"""
import base64
import io
from pathlib import Path

import joblib
import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image
from skimage.color import rgb2lab

from lasso import auto_soil_hsv, make_iso_preview
from soil_common import (
    TARGET_WHITE,
    clean_reference,
    detect_mode,
    find_white,
    lin_to_srgb,
    load_srgb,
    set_scale,
    srgb_to_lin,
)
from soil_stack_common import USE_META, extract_79_soil_features
from T_cnn_n88 import soil_crop

LEVEL_MAP = {0: "L", 1: "M", 2: "H"}
LEVEL_NAME = {0: "Low", 1: "Medium", 2: "High"}


class SoilScanPredictor:
    def __init__(self, model_path="models_out/production_stack.joblib", max_side=800):
        self.max_side = max_side
        self.payload = joblib.load(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = tvm.resnet50(weights="DEFAULT").to(self.device).eval()
        self.cnn.fc = torch.nn.Identity()
        self.mu = np.array([0.485, 0.456, 0.406])
        self.sd = np.array([0.229, 0.224, 0.225])

    def _extract_single_cnn(self, rgb_wb, mask):
        crop = soil_crop(rgb_wb, mask, out=224)
        batch = ((crop[None, ...] - self.mu) / self.sd).transpose(0, 3, 1, 2)
        x = torch.tensor(batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.cnn(x).cpu().numpy().astype(np.float32)

    def predict_image(self, image_path: str | Path, normalise_exposure: bool = False) -> dict:
        set_scale(self.max_side)
        rgb, cam = load_srgb(image_path, self.max_side)

        # 1. White Balance Standardization
        white = find_white(rgb)
        mode = detect_mode(rgb, white)
        is_container = mode == "container"
        ref_mask = clean_reference(rgb, white, is_container=is_container)

        lin = srgb_to_lin(rgb)
        ref = np.maximum(np.median(lin[ref_mask], axis=0), 1e-6)
        ref_lab = np.median(rgb2lab(rgb)[ref_mask], axis=0)

        flags = []
        if white.mean() < 0.10:
            flags.append("little white reference found")
        if np.hypot(ref_lab[1], ref_lab[2]) > 8:
            flags.append("white reference not neutral")

        gain = (TARGET_WHITE / ref) if normalise_exposure else (ref.mean() / ref)
        rgb_wb = lin_to_srgb(np.clip(lin * gain, 0, 1))

        # 2. HSV Lasso Mask
        set_scale(600)
        mask = auto_soil_hsv(rgb_wb)
        soil_ratio = float(mask.mean())
        if soil_ratio < 0.05 or soil_ratio > 0.85 or mask.sum() < 200:
            flags.append("implausible soil area")
            if mask.sum() < 200:
                return {
                    "status": "error",
                    "message": "Could not isolate soil. Ensure soil is placed on white paper/container.",
                    "qc_flags": flags,
                }

        # 3. Extract 88 Features + ResNet50 Embeddings
        lab = rgb2lab(rgb_wb)
        f79 = extract_79_soil_features(lab[mask], lab[..., 0], mask)

        cam_lower = cam.lower()
        meta = np.array([
            float(mode == "container"),
            soil_ratio,
            float("iphone" in cam_lower),
            float("infinix" in cam_lower),
            float(any(k in cam_lower for k in ("samsung", "xiaomi", "redmi"))),
            float("no-exif" in cam_lower),
            float(mode == "paper"),
            1.0,
            0.0,
        ], dtype=np.float32)

        if not USE_META:
            meta = np.zeros_like(meta)

        X88 = np.hstack([f79, meta])[None, :]
        X_color_meta = np.hstack([f79[:39], meta])[None, :]
        X_cnn = self._extract_single_cnn(rgb_wb, mask)

        # 4. Predict N, P, K using the 3-Model Stack
        results = {}
        pattern_digits = []
        for target in ["N", "P", "K"]:
            m = self.payload["models"][target]
            p_svm = m["svm"].predict_proba(X_color_meta)
            p_xgb = m["xgb"].predict_proba(X88)

            X88_s = m["hyb_s88"].transform(X88)
            Xcnn_s = m["hyb_scnn"].transform(X_cnn)
            Xcnn_p = m["hyb_pca"].transform(Xcnn_s)
            X_fused = np.hstack([X88_s, Xcnn_p])
            p_hyb = m["hyb_mlp"].predict_proba(X_fused)

            X_meta = np.hstack([p_svm, p_xgb, p_hyb])
            p_stack = m["meta_lr"].predict_proba(X_meta)[0]

            pred_idx = int(np.argmax(p_stack))
            pattern_digits.append(str(pred_idx))
            results[target] = {
                "code": pred_idx,
                "level": LEVEL_MAP[pred_idx],
                "label": LEVEL_NAME[pred_idx],
                "confidence": round(float(p_stack[pred_idx]), 4),
            }

        iso_img = make_iso_preview(rgb_wb, mask)
        buf = io.BytesIO()
        iso_img.save(buf, format="JPEG", quality=85)
        iso_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "pattern": "".join(pattern_digits),
            "mode_detected": mode,
            "qc_flags": flags,
            "results": results,
            "isolated_preview_base64": iso_b64,
        }