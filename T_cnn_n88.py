#!/usr/bin/env python3
"""
COMPUTER 3 (or GPU Laptop): Train Hybrid CNN x Net-88 Model
Also evaluates Pure 2D-CNN and Pure Net-88 for your thesis ablation study, and saves
the Hybrid Out-Of-Fold probabilities to `models_out/oof_hybrid.npz` for the 3-model stack.

Run: py step4c_train_hybrid_cnn_net88.py
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")      # MLP "max_iter reached" notices
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from soil_common import load_plain
from soil_stack_common import get_base_dataset, get_oof_probabilities, summarize_model_oof


def soil_crop(rgb, mask, out=224):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    mean = rgb[mask].mean(axis=0)
    crop = np.where(mask[y0:y1, x0:x1, None], rgb[y0:y1, x0:x1], mean)
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.ones((s, s, 3)) * mean
    sq[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = crop
    return np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((out, out), Image.BILINEAR)) / 255.0


def _extract_embeddings(df, standardized, masks, backbone):
    import torch
    import torchvision.models as tvm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CNN] Extracting {backbone} embeddings on {device} for {len(df)} photos...")
    net = getattr(tvm, backbone)(weights="DEFAULT").to(device).eval()
    if hasattr(net, "fc"):
        net.fc = torch.nn.Identity()
    else:
        net.classifier = torch.nn.Identity()

    crops = []
    for r in df.itertuples():
        rgb = load_plain(Path(standardized) / r.pattern / f"{r.stem}.png")
        m = np.asarray(Image.open(Path(masks) / r.pattern / f"{r.stem}.png")) > 0
        crops.append(soil_crop(rgb, m))

    mu, sd = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 32):
            batch = ((np.stack(crops[i : i + 32]) - mu) / sd).transpose(0, 3, 1, 2)
            x = torch.tensor(batch, dtype=torch.float32, device=device)
            out.append(net(x).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def get_cnn_embeddings(df, standardized="standardized", masks="masks", out_dir="models_out", backbone="resnet50"):
    """Embeddings aligned to df's rows. Cached per photo AND mask version, so a changed photo set or a re-lassoed mask
    is re-extracted, and unchanged photos are never recomputed."""
    cache = Path(out_dir) / f"cnn_embeddings_{backbone}.npz"
    ids = [f"{k}@{mt:.3f}" for k, mt in zip(df["key"], df["mask_mtime"])]
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = dict(zip(z["ids"].tolist(), z["emb"]))
    need = [i for i, k in enumerate(ids) if k not in have]
    if need and not (Path(standardized) / "wb_report.csv").exists():
        # No photos on this computer: fall back to the old position-based cache if it matches this dataset exactly.
        legacy = Path(out_dir) / f"cnn_embeddings_{backbone}.npy"
        if legacy.exists() and len(np.load(legacy)) == len(df):
            print(f"[CNN] no photos here -> using the old cache {legacy.name} (matched by row count only)")
            return np.load(legacy).astype(np.float32)
        raise SystemExit("No photos here to extract CNN embeddings from, and no matching embeddings cache.\n"
                         f"Copy cnn_embeddings_{backbone}.npz (or the old .npy) into models_out/, or run on the computer with standardized/ and masks/.")
    if need:
        new = _extract_embeddings(df.iloc[need], standardized, masks, backbone)
        for i, e in zip(need, new):
            have[ids[i]] = e
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ids=np.array(list(have.keys())), emb=np.vstack(list(have.values())))
    return np.vstack([have[k] for k in ids]).astype(np.float32)


def fit_predict_net88(X_tr, y_tr, X_te, present):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.5,
        learning_rate_init=0.002,
        max_iter=300,
        early_stopping=False,
        random_state=42,
    )
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)


def fit_predict_cnn_only(X_tr, y_tr, X_te, present):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    n_comp = min(64, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=300, early_stopping=False, random_state=42)
    clf.fit(X_tr_p, y_tr)
    return clf.predict_proba(X_te_p)


def make_hybrid_fitter(n_88_cols):
    """Fuses all 88 Net-88 features + top 48 fold-PCA components of the 2D CNN inside each CV fold (scalers and PCA are fit on the training photos of each fold only)."""
    def fit_predict_hybrid(X_tr, y_tr, X_te, present):
        X88_tr, Xcnn_tr = X_tr[:, :n_88_cols], X_tr[:, n_88_cols:]
        X88_te, Xcnn_te = X_te[:, :n_88_cols], X_te[:, n_88_cols:]

        s88 = StandardScaler()
        X88_tr_s = s88.fit_transform(X88_tr)
        X88_te_s = s88.transform(X88_te)

        scnn = StandardScaler()
        Xcnn_tr_s = scnn.fit_transform(Xcnn_tr)
        Xcnn_te_s = scnn.transform(Xcnn_te)

        n_comp = min(48, Xcnn_tr_s.shape[0] - 1, Xcnn_tr_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        Xcnn_tr_p = pca.fit_transform(Xcnn_tr_s)
        Xcnn_te_p = pca.transform(Xcnn_te_s)

        X_tr_fused = np.hstack([X88_tr_s, Xcnn_tr_p])
        X_te_fused = np.hstack([X88_te_s, Xcnn_te_p])

        clf = MLPClassifier(
            hidden_layer_sizes=(160, 80),
            activation="relu",
            alpha=0.6,
            learning_rate_init=0.002,
            max_iter=300,
            early_stopping=False,
            random_state=42,
        )
        clf.fit(X_tr_fused, y_tr)
        return clf.predict_proba(X_te_fused)

    return fit_predict_hybrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    a = ap.parse_args()

    df, _, X88 = get_base_dataset()
    X_cnn = get_cnn_embeddings(df, backbone=a.backbone)
    X_combined = np.hstack([X88, X_cnn])

    groups = df["group"].values
    pat_classes = np.unique(df["pattern"].values)
    npk_classes = np.array([0, 1, 2])
    out = Path("models_out")
    out.mkdir(exist_ok=True)

    # 1. Pure Net-88 (Ablation baseline)
    print("\n[1/3] Evaluating Pure Net-88 (88 Soil Features)...")
    n88_pat = get_oof_probabilities(fit_predict_net88, X88, df["pattern"].values, groups, pat_classes)
    n88_N = get_oof_probabilities(fit_predict_net88, X88, df["N"].values, groups, npk_classes)
    n88_P = get_oof_probabilities(fit_predict_net88, X88, df["P"].values, groups, npk_classes)
    n88_K = get_oof_probabilities(fit_predict_net88, X88, df["K"].values, groups, npk_classes)
    res_n88 = summarize_model_oof("Net-88 Only", df, n88_pat, n88_N, n88_P, n88_K, pat_classes)

    # 2. Pure 2D-CNN (Ablation baseline)
    print("[2/3] Evaluating Pure 2D-CNN (ResNet50)...")
    cnn_pat = get_oof_probabilities(fit_predict_cnn_only, X_cnn, df["pattern"].values, groups, pat_classes)
    cnn_N = get_oof_probabilities(fit_predict_cnn_only, X_cnn, df["N"].values, groups, npk_classes)
    cnn_P = get_oof_probabilities(fit_predict_cnn_only, X_cnn, df["P"].values, groups, npk_classes)
    cnn_K = get_oof_probabilities(fit_predict_cnn_only, X_cnn, df["K"].values, groups, npk_classes)
    res_cnn = summarize_model_oof("2D-CNN Only", df, cnn_pat, cnn_N, cnn_P, cnn_K, pat_classes)

    # 3. Hybrid CNN x Net-88 (Main 3rd Pillar of the Stack)
    print("[3/3] Evaluating Hybrid CNN x Net-88 (Fused Architecture)...")
    hybrid_fn = make_hybrid_fitter(X88.shape[1])
    hyb_pat = get_oof_probabilities(hybrid_fn, X_combined, df["pattern"].values, groups, pat_classes)
    hyb_N = get_oof_probabilities(hybrid_fn, X_combined, df["N"].values, groups, npk_classes)
    hyb_P = get_oof_probabilities(hybrid_fn, X_combined, df["P"].values, groups, npk_classes)
    hyb_K = get_oof_probabilities(hybrid_fn, X_combined, df["K"].values, groups, npk_classes)
    res_hyb = summarize_model_oof("Hybrid (CNN x Net-88)", df, hyb_pat, hyb_N, hyb_P, hyb_K, pat_classes)

    np.savez_compressed(
        out / "oof_hybrid.npz",
        keys=df["key"].values,
        pat_classes=pat_classes,
        oof_pat=hyb_pat,
        oof_N=hyb_N,
        oof_P=hyb_P,
        oof_K=hyb_K,
    )

    ablation = pd.concat([res_cnn, res_n88, res_hyb], ignore_index=True)
    ablation.to_csv(out / "results_hybrid_ablation.csv", index=False)
    print("\n=== ABLATION STUDY: 2D-CNN vs NET-88 vs HYBRID (CNN x NET-88) ===")
    print(ablation.to_string(index=False))
    print(f"\nSaved Hybrid OOF probabilities to: {out / 'oof_hybrid.npz'}")


if __name__ == "__main__":
    main()