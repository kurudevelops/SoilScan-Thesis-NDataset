#!/usr/bin/env python3
"""
STEP 5: Production Export
Trains the final SVM, XGBoost, and Hybrid models on 100% of the dataset.
Trains the Meta-Learners using the Out-Of-Fold (OOF) probabilities.
Saves everything into a single 'production_stack.joblib' file for the API.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from soil_stack_common import get_base_dataset
from T_cnn_n88 import get_cnn_embeddings


def get_xgb_weights(y, classes):
    idx_map = {c: i for i, c in enumerate(classes)}
    yy = np.array([idx_map[v] for v in y])
    w = pd.Series(yy).map(len(yy) / (len(classes) * pd.Series(yy).value_counts())).values
    return yy, w


def main():
    out = Path("models_out")
    out.mkdir(exist_ok=True)
    
    print("Loading 19-pattern dataset and OOF predictions...")
    df, X_color_meta, X88 = get_base_dataset()
    X_cnn = get_cnn_embeddings(df, backbone="resnet50")
    X_hybrid = np.hstack([X88, X_cnn])
    
    req_files = ["oof_svm.npz", "oof_xgboost.npz", "oof_hybrid.npz"]
    for f in req_files:
        if not (out / f).exists():
            raise SystemExit(f"Missing {out / f}. Run the 3 training scripts first.")
            
    svm_oof = np.load(out / "oof_svm.npz", allow_pickle=True)
    xgb_oof = np.load(out / "oof_xgboost.npz", allow_pickle=True)
    hyb_oof = np.load(out / "oof_hybrid.npz", allow_pickle=True)
    
    npk_classes = np.array([0, 1, 2])
    
    # Structure the payload for the Chatbot API
    production_dict = {
        "classes": npk_classes,
        "models": {"N": {}, "P": {}, "K": {}}
    }
    
    for target in ["N", "P", "K"]:
        print(f"\n--- Training 100% Production Models for {target} ---")
        y = df[target].values
        
        # 1. SVM (Color + Meta)
        print("Training SVM...")
        svm = make_pipeline(
            StandardScaler(), 
            SVC(kernel="rbf", C=10.0, probability=True, class_weight="balanced", random_state=42)
        )
        svm.fit(X_color_meta, y)
        
        # 2. XGBoost (88 Features)
        print("Training XGBoost...")
        yy, w = get_xgb_weights(y, npk_classes)
        xgb = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2, reg_alpha=0.0, reg_lambda=2.0,
            n_jobs=-1, random_state=42, verbosity=0
        )
        xgb.fit(X88, yy, sample_weight=w)
        
        # 3. Hybrid (CNN x Net-88)
        print("Training Hybrid...")
        n_88_cols = X88.shape[1]
        X88_all, Xcnn_all = X_hybrid[:, :n_88_cols], X_hybrid[:, n_88_cols:]
        
        s88 = StandardScaler()
        X88_s = s88.fit_transform(X88_all)
        
        scnn = StandardScaler()
        Xcnn_s = scnn.fit_transform(Xcnn_all)
        
        n_comp = min(48, Xcnn_s.shape[0] - 1, Xcnn_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        Xcnn_p = pca.fit_transform(Xcnn_s)
        
        X_fused = np.hstack([X88_s, Xcnn_p])
        
        hyb = MLPClassifier(
            hidden_layer_sizes=(160, 80), activation="relu", alpha=0.6, 
            learning_rate_init=0.002, max_iter=600, early_stopping=True, random_state=42
        )
        hyb.fit(X_fused, y)
        
        # 4. Meta-Learner Stack
        print("Training Meta-Learner Stack...")
        X_meta = np.hstack([svm_oof[f"oof_{target}"], xgb_oof[f"oof_{target}"], hyb_oof[f"oof_{target}"]])
        meta_lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        meta_lr.fit(X_meta, y)
        
        # Store in payload
        production_dict["models"][target] = {
            "svm": svm,
            "xgb": xgb,
            "hyb_s88": s88,
            "hyb_scnn": scnn,
            "hyb_pca": pca,
            "hyb_mlp": hyb,
            "meta_lr": meta_lr
        }
        
    print(f"\nExporting final weights to {out / 'production_stack.joblib'}...")
    joblib.dump(production_dict, out / "production_stack.joblib")
    print("Success! The chatbot API is ready for inference.")


if __name__ == "__main__":
    main()