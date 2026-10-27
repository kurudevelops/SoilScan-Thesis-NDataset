#!/usr/bin/env python3
"""
COMPUTER 2: Train XGBoost Model (Single 19-Class + Independent N, P, K)
Saves Out-Of-Fold probabilities to `models_out/oof_xgboost.npz` for the final 3-model stack.

Run: py step4b_train_xgboost.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from Backend.soil_stack_common import get_base_dataset, get_oof_probabilities, summarize_model_oof


def fit_predict_xgb(X_tr, y_tr, X_te, present):
    from xgboost import XGBClassifier

    idx_map = {c: i for i, c in enumerate(present)}
    yy = np.array([idx_map[v] for v in y_tr])
    w = pd.Series(yy).map(len(yy) / (len(present) * pd.Series(yy).value_counts())).values

    clf = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_alpha=0.0,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_tr, yy, sample_weight=w)
    return clf.predict_proba(X_te)


def main():
    df, _, X88 = get_base_dataset()
    groups = df["group"].values
    pat_classes = np.unique(df["pattern"].values)
    npk_classes = np.array([0, 1, 2])

    print(f"[XGBoost] Training on {len(df)} photos using all 88 Color+Texture+Meta features...")
    oof_pat = get_oof_probabilities(fit_predict_xgb, X88, df["pattern"].values, groups, pat_classes)
    oof_N = get_oof_probabilities(fit_predict_xgb, X88, df["N"].values, groups, npk_classes)
    oof_P = get_oof_probabilities(fit_predict_xgb, X88, df["P"].values, groups, npk_classes)
    oof_K = get_oof_probabilities(fit_predict_xgb, X88, df["K"].values, groups, npk_classes)

    out = Path("models_out")
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "oof_xgboost.npz",
        keys=df["key"].values,
        pat_classes=pat_classes,
        oof_pat=oof_pat,
        oof_N=oof_N,
        oof_P=oof_P,
        oof_K=oof_K,
    )

    res = summarize_model_oof("XGBoost (Color+Tex)", df, oof_pat, oof_N, oof_P, oof_K, pat_classes)
    res.to_csv(out / "results_xgboost.csv", index=False)
    print("\n=== XGBOOST STANDALONE RESULTS ===")
    print(res.to_string(index=False))
    print(f"\nSaved OOF probabilities to: {out / 'oof_xgboost.npz'}")


if __name__ == "__main__":
    main()