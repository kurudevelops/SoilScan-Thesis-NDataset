#!/usr/bin/env python3
"""
COMPUTER 1: Train SVM Model (Single 19-Class + Independent N, P, K)
Saves Out-Of-Fold probabilities to `models_out/oof_svm.npz` for the final 3-model stack.

Run: py step4a_train_svm.py
"""
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Backend.soil_stack_common import get_base_dataset, get_oof_probabilities, summarize_model_oof


def fit_predict_svm(X_tr, y_tr, X_te, present):
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, probability=True, class_weight="balanced", random_state=42),
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)


def main():
    df, X_color_meta, _ = get_base_dataset()
    groups = df["group"].values
    pat_classes = np.unique(df["pattern"].values)
    npk_classes = np.array([0, 1, 2])

    print(f"[SVM] Training on {len(df)} photos across {len(pat_classes)} patterns...")
    oof_pat = get_oof_probabilities(fit_predict_svm, X_color_meta, df["pattern"].values, groups, pat_classes)
    oof_N = get_oof_probabilities(fit_predict_svm, X_color_meta, df["N"].values, groups, npk_classes)
    oof_P = get_oof_probabilities(fit_predict_svm, X_color_meta, df["P"].values, groups, npk_classes)
    oof_K = get_oof_probabilities(fit_predict_svm, X_color_meta, df["K"].values, groups, npk_classes)

    out = Path("models_out")
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "oof_svm.npz",
        keys=df["key"].values,
        pat_classes=pat_classes,
        oof_pat=oof_pat,
        oof_N=oof_N,
        oof_P=oof_P,
        oof_K=oof_K,
    )

    res = summarize_model_oof("SVM (Color)", df, oof_pat, oof_N, oof_P, oof_K, pat_classes)
    res.to_csv(out / "results_svm.csv", index=False)
    print("\n=== SVM STANDALONE RESULTS ===")
    print(res.to_string(index=False))
    print(f"\nSaved OOF probabilities to: {out / 'oof_svm.npz'}")


if __name__ == "__main__":
    main()