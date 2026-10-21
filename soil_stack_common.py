#!/usr/bin/env python3
"""
Shared feature extractor & deterministic CV splitter for multi-PC distributed training.
Keep this file on every machine alongside `soil_common.py`.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import skew
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from soil_common import load_plain


def extract_79_soil_features(lab_pixels, L_full, mask):
    """39 CIELAB/Chroma/Hue distribution features + 40 multi-scale GLCM & LBP texture features."""
    if len(lab_pixels) > 25000:
        lab_pixels = lab_pixels[np.random.default_rng(0).choice(len(lab_pixels), 25000, replace=False)]

    feats = []
    a, b = lab_pixels[:, 1], lab_pixels[:, 2]
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a)
    for ch in (lab_pixels[:, 0], a, b, chroma, hue):
        p10, p25, p50, p75, p90 = np.percentile(ch, [10, 25, 50, 75, 90])
        feats.extend([p10, p25, p50, p75, p90, float(np.mean(ch)), float(np.std(ch))])

    L_med, a_med, b_med, c_med = feats[2], feats[9], feats[16], feats[23]
    feats.extend([
        L_med / max(c_med, 1e-4),
        a_med / max(abs(b_med), 1e-4),
        (a_med ** 2) / max(L_med * abs(b_med), 1e-4),
        float(skew(lab_pixels[:, 0])),
    ])

    rng = np.random.default_rng(0)
    inner = ndi.binary_erosion(mask, iterations=4)
    q = np.clip(L_full / 100.0 * 31, 0, 31).astype(np.uint8)
    patch_feats = []
    for size in (64, 32):
        ok = ndi.uniform_filter(inner.astype(float), size=size) > 0.99
        ys, xs = np.nonzero(ok)
        if len(ys) >= 10:
            for k in rng.choice(len(ys), min(10, len(ys)), replace=False):
                y0, x0 = ys[k] - size // 2, xs[k] - size // 2
                p = q[y0 : y0 + size, x0 : x0 + size]
                if p.shape != (size, size):
                    continue
                g = graycomatrix(
                    p, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=32, symmetric=True, normed=True
                )
                gl = []
                for pr in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity"):
                    vals = graycoprops(g, pr)
                    gl.extend([float(vals.mean()), float(vals.std()), float(vals[0].mean()), float(vals[-1].mean())])
                lbp1 = local_binary_pattern(p, 8, 1, "uniform")
                h1 = np.bincount(lbp1.astype(int).ravel(), minlength=10)[:10] / lbp1.size
                lbp2 = local_binary_pattern(p, 8, 2, "uniform")
                h2 = np.bincount(lbp2.astype(int).ravel(), minlength=10)[:10] / lbp2.size
                patch_feats.append(gl + list(h1) + list(h2))
            break

    if patch_feats:
        feats.extend(list(np.mean(patch_feats, axis=0)))
    else:
        lbp_g = local_binary_pattern(q, 8, 1, "uniform")[mask]
        h_g = np.bincount(lbp_g.astype(int).ravel(), minlength=10)[:10] / max(lbp_g.size, 1)
        feats.extend([0.0] * 20 + list(h_g) + list(h_g))

    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)


def get_base_dataset(standardized="standardized", masks="masks", out_dir="models_out", block_size=5):
    """Loads or builds the deterministic 88-feature dataset (`features_88.pkl`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "features_88.pkl"

    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        rep = pd.read_csv(Path(standardized) / "wb_report.csv", dtype=str).fillna("")
        rows = []
        print(f"Extracting 88 soil features from {standardized}/ and {masks}/ ...")
        for i, r in enumerate(rep.itertuples(), 1):
            mp = Path(masks) / r.pattern / f"{r.stem}.png"
            sp = Path(standardized) / r.pattern / f"{r.stem}.png"
            if not mp.exists() or not sp.exists():
                continue
            rgb = load_plain(sp)
            m = np.asarray(Image.open(mp)) > 0
            if m.shape != rgb.shape[:2] or m.sum() < 200:
                continue
            lab = rgb2lab(rgb)
            f79 = extract_79_soil_features(lab[m], lab[..., 0], m)
            rows.append({
                "key": f"{r.pattern}/{r.stem}",
                "pattern": str(r.pattern),
                "file": r.file,
                "stem": r.stem,
                "camera": r.camera,
                "mode": getattr(r, "mode", "paper"),
                "area_ratio": float(m.mean()),
                "f79": f79,
            })
            if i % 100 == 0:
                print(f"  processed {i}/{len(rep)} entries...")
        df = pd.DataFrame(rows)
        df.to_pickle(cache)

    # Strictly sort by (pattern, camera, file) so every computer has the exact same row order
    df = df[df["pattern"].str.match(r"^[0-2]{3}$")].sort_values(["pattern", "camera", "file"]).reset_index(drop=True)
    df["N"] = df["pattern"].str[0].astype(int)
    df["P"] = df["pattern"].str[1].astype(int)
    df["K"] = df["pattern"].str[2].astype(int)
    df["group"] = (
        df["pattern"] + "_" + df["camera"] + "_" + (df.groupby(["pattern", "camera"]).cumcount() // block_size).astype(str)
    )

    cam_lower = df["camera"].str.lower()
    meta = np.column_stack([
        (df["mode"] == "container").astype(np.float32),
        df["area_ratio"].values.astype(np.float32),
        cam_lower.str.contains("iphone").astype(np.float32),
        cam_lower.str.contains("infinix").astype(np.float32),
        cam_lower.str.contains("samsung|xiaomi|redmi").astype(np.float32),
        cam_lower.str.contains("no-exif").astype(np.float32),
        (df["mode"] == "paper").astype(np.float32),
        np.ones(len(df), dtype=np.float32),
        np.zeros(len(df), dtype=np.float32),
    ])
    X88 = np.hstack([np.vstack(df["f79"].values), meta])
    X_color_meta = np.hstack([X88[:, :39], meta])
    return df, X_color_meta, X88


def get_oof_probabilities(fit_predict_proba_fn, X, y, groups, classes, n_splits=5):
    """Generates leakage-safe Out-Of-Fold (OOF) probability matrices across 5 burst-block folds."""
    classes = np.asarray(classes)
    oof = np.zeros((len(y), len(classes)), dtype=np.float32)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    for tr, te in gkf.split(X, y, groups):
        present = np.unique(y[tr])
        if len(present) < 2:
            c_idx = int(np.where(classes == present[0])[0][0])
            oof[te, c_idx] = 1.0
            continue
        probs = fit_predict_proba_fn(X[tr], y[tr], X[te], present)
        for j, c in enumerate(present):
            c_idx = int(np.where(classes == c)[0][0])
            oof[te, c_idx] = probs[:, j]
    return oof


def summarize_model_oof(model_name, df, oof_pat, oof_N, oof_P, oof_K, pat_classes):
    """Evaluates Single 19-Class vs. 3 Independent NPK models from their OOF probability matrices."""
    pat_classes = np.asarray(pat_classes)
    pred_pat_single = pat_classes[oof_pat.argmax(axis=1)]
    n_s = np.array([int(str(p)[0]) for p in pred_pat_single])
    p_s = np.array([int(str(p)[1]) for p in pred_pat_single])
    k_s = np.array([int(str(p)[2]) for p in pred_pat_single])

    n_i = np.array([0, 1, 2])[oof_N.argmax(axis=1)]
    p_i = np.array([0, 1, 2])[oof_P.argmax(axis=1)]
    k_i = np.array([0, 1, 2])[oof_K.argmax(axis=1)]
    pred_pat_ind = np.array([f"{n}{p}{k}" for n, p, k in zip(n_i, p_i, k_i)])

    y_pat = df["pattern"].values
    rows = [
        {
            "Model": model_name, "Target": "N",
            "Single_19Class_Acc": accuracy_score(df["N"], n_s),
            "Independent_NPK_Acc": accuracy_score(df["N"], n_i),
            "Independent_BalAcc": balanced_accuracy_score(df["N"], n_i),
            "Independent_MacroF1": f1_score(df["N"], n_i, average="macro", zero_division=0),
        },
        {
            "Model": model_name, "Target": "P",
            "Single_19Class_Acc": accuracy_score(df["P"], p_s),
            "Independent_NPK_Acc": accuracy_score(df["P"], p_i),
            "Independent_BalAcc": balanced_accuracy_score(df["P"], p_i),
            "Independent_MacroF1": f1_score(df["P"], p_i, average="macro", zero_division=0),
        },
        {
            "Model": model_name, "Target": "K",
            "Single_19Class_Acc": accuracy_score(df["K"], k_s),
            "Independent_NPK_Acc": accuracy_score(df["K"], k_i),
            "Independent_BalAcc": balanced_accuracy_score(df["K"], k_i),
            "Independent_MacroF1": f1_score(df["K"], k_i, average="macro", zero_division=0),
        },
        {
            "Model": model_name, "Target": "Pattern_All3",
            "Single_19Class_Acc": accuracy_score(y_pat, pred_pat_single),
            "Independent_NPK_Acc": accuracy_score(y_pat, pred_pat_ind),
            "Independent_BalAcc": balanced_accuracy_score(y_pat, pred_pat_ind),
            "Independent_MacroF1": f1_score(y_pat, pred_pat_ind, average="macro", zero_division=0),
        },
    ]
    return pd.DataFrame(rows)