#!/usr/bin/env python3
"""
FINAL STEP: Combine SVM + XGBoost + Hybrid (CNN x Net-88) into the 3-Model Stack.
Reads:
  - models_out/oof_svm.npz
  - models_out/oof_xgboost.npz
  - models_out/oof_hybrid.npz

Outputs:
  - models_out/master_comparison.csv         (All individual models + Soft Vote + Meta-Stack)
  - models_out/stack_summary_report.txt      (Detailed classification report & confusion matrices)
  - models_out/final_stack_predictions.csv   (Per-photo predictions across every model)

Run: py step4d_stack_and_compare.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

from soil_stack_common import get_base_dataset, summarize_model_oof


def stack_oof_probs(prob_list, y, groups, classes, n_splits=5):
    """Fits a regularized LogisticRegression meta-learner on the concatenated OOF probabilities."""
    X_meta = np.hstack(prob_list)
    classes = np.asarray(classes)
    oof_stack = np.zeros((len(y), len(classes)), dtype=np.float32)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    for tr, te in gkf.split(X_meta, y, groups):
        present = np.unique(y[tr])
        if len(present) < 2:
            c_idx = int(np.where(classes == present[0])[0][0])
            oof_stack[te, c_idx] = 1.0
            continue
        meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        meta.fit(X_meta[tr], y[tr])
        probs = meta.predict_proba(X_meta[te])
        for j, c in enumerate(meta.classes_):
            c_idx = int(np.where(classes == c)[0][0])
            oof_stack[te, c_idx] = probs[:, j]
    return oof_stack


def main():
    out = Path("models_out")
    req_files = ["oof_svm.npz", "oof_xgboost.npz", "oof_hybrid.npz"]
    for f in req_files:
        if not (out / f).exists():
            raise SystemExit(f"Missing {out / f}. Make sure you ran step4a, step4b, and step4c and copied the .npz files into models_out/.")

    df, _, _ = get_base_dataset()
    groups = df["group"].values
    pat_classes = np.unique(df["pattern"].values)
    npk_classes = np.array([0, 1, 2])

    svm = np.load(out / "oof_svm.npz", allow_pickle=True)
    xgb = np.load(out / "oof_xgboost.npz", allow_pickle=True)
    hyb = np.load(out / "oof_hybrid.npz", allow_pickle=True)

    # Verify that all 3 computers evaluated the exact same photo order
    if not (np.array_equal(df["key"].values, svm["keys"]) and np.array_equal(df["key"].values, xgb["keys"]) and np.array_equal(df["key"].values, hyb["keys"])):
        raise SystemExit("Key mismatch between .npz files! Ensure all computers used the same wb_report.csv and masks.")

    # 1. Summarize the 3 individual pillar models
    t_svm = summarize_model_oof("1. SVM (Color)", df, svm["oof_pat"], svm["oof_N"], svm["oof_P"], svm["oof_K"], pat_classes)
    t_xgb = summarize_model_oof("2. XGBoost (Color+Tex)", df, xgb["oof_pat"], xgb["oof_N"], xgb["oof_P"], xgb["oof_K"], pat_classes)
    t_hyb = summarize_model_oof("3. Hybrid (CNN x Net-88)", df, hyb["oof_pat"], hyb["oof_N"], hyb["oof_P"], hyb["oof_K"], pat_classes)

    # 2. Soft-Voting Ensemble (Average probabilities of SVM + XGBoost + Hybrid)
    sv_pat = (svm["oof_pat"] + xgb["oof_pat"] + hyb["oof_pat"]) / 3.0
    sv_N = (svm["oof_N"] + xgb["oof_N"] + hyb["oof_N"]) / 3.0
    sv_P = (svm["oof_P"] + xgb["oof_P"] + hyb["oof_P"]) / 3.0
    sv_K = (svm["oof_K"] + xgb["oof_K"] + hyb["oof_K"]) / 3.0
    t_sv = summarize_model_oof("4. Soft-Vote (3-Model Avg)", df, sv_pat, sv_N, sv_P, sv_K, pat_classes)

    # 3. Meta-Learner Stack (Logistic Regression on SVM + XGBoost + Hybrid OOF Probabilities)
    st_pat = stack_oof_probs([svm["oof_pat"], xgb["oof_pat"], hyb["oof_pat"]], df["pattern"].values, groups, pat_classes)
    st_N = stack_oof_probs([svm["oof_N"], xgb["oof_N"], hyb["oof_N"]], df["N"].values, groups, npk_classes)
    st_P = stack_oof_probs([svm["oof_P"], xgb["oof_P"], hyb["oof_P"]], df["P"].values, groups, npk_classes)
    st_K = stack_oof_probs([svm["oof_K"], xgb["oof_K"], hyb["oof_K"]], df["K"].values, groups, npk_classes)
    t_st = summarize_model_oof("5. 3-Model Stack (Meta-LR)", df, st_pat, st_N, st_P, st_K, pat_classes)

    # Combine all results into Master CSV
    tables = [t_svm, t_xgb, t_hyb, t_sv, t_st]
    if (out / "results_hybrid_ablation.csv").exists():
        ab = pd.read_csv(out / "results_hybrid_ablation.csv")
        ab_sub = ab[ab["Model"].isin(["2D-CNN Only", "Net-88 Only"])]
        tables = [t_svm, t_xgb, ab_sub, t_hyb, t_sv, t_st]

    master = pd.concat(tables, ignore_index=True)
    master["Gain_Ind_vs_Single_Pct"] = ((master["Independent_NPK_Acc"] - master["Single_19Class_Acc"]) * 100).map(lambda v: f"{v:+.2f}%")
    master["Single_19Class_Acc_Pct"] = (master["Single_19Class_Acc"] * 100).map(lambda v: f"{v:.2f}%")
    master["Independent_NPK_Acc_Pct"] = (master["Independent_NPK_Acc"] * 100).map(lambda v: f"{v:.2f}%")
    master["Independent_BalAcc_Pct"] = (master["Independent_BalAcc"] * 100).map(lambda v: f"{v:.2f}%")
    master["Independent_MacroF1_Pct"] = (master["Independent_MacroF1"] * 100).map(lambda v: f"{v:.2f}%")

    master.to_csv(out / "master_comparison.csv", index=False)

    # Save per-photo predictions
    pred_df = pd.DataFrame({
        "pattern": df["pattern"],
        "file": df["file"],
        "camera": df["camera"],
        "mode": df["mode"],
        "true_N": df["N"], "true_P": df["P"], "true_K": df["K"],
        "SVM_N": sv_N.argmax(1), "XGB_N": xgb["oof_N"].argmax(1), "HYB_N": hyb["oof_N"].argmax(1), "STACK_N": st_N.argmax(1),
        "SVM_P": sv_P.argmax(1), "XGB_P": xgb["oof_P"].argmax(1), "HYB_P": hyb["oof_P"].argmax(1), "STACK_P": st_P.argmax(1),
        "SVM_K": sv_K.argmax(1), "XGB_K": xgb["oof_K"].argmax(1), "HYB_K": hyb["oof_K"].argmax(1), "STACK_K": st_K.argmax(1),
    })
    pred_df.to_csv(out / "final_stack_predictions.csv", index=False)

    # Build text report
    display_cols = ["Model", "Target", "Single_19Class_Acc_Pct", "Independent_NPK_Acc_Pct", "Gain_Ind_vs_Single_Pct", "Independent_BalAcc_Pct", "Independent_MacroF1_Pct"]
    report_lines = [
        "==========================================================================================",
        " 3-MODEL STACK THESIS REPORT: SVM + XGBoost + Hybrid (CNN x Net-88)",
        "==========================================================================================",
        f"Total Photos Evaluated: {len(df)} | Total NPK Patterns: {len(pat_classes)} | Validation: 20-Fold Burst-Block GroupKFold",
        "",
        master[display_cols].to_string(index=False),
        "",
        "==========================================================================================",
        " DETAILED CLASSIFICATION REPORTS FOR 3-MODEL STACK (INDEPENDENT NPK)",
        "==========================================================================================",
    ]
    for nut, target_col, oof_m in [("Nitrogen (N)", "N", st_N), ("Phosphorus (P)", "P", st_P), ("Potassium (K)", "K", st_K)]:
        y_true = df[target_col].values
        y_pred = oof_m.argmax(axis=1)
        report_lines.append(f"\n--- {nut} (0=Low, 1=Medium, 2=High) ---")
        report_lines.append(classification_report(y_true, y_pred, digits=3, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        report_lines.append("Confusion Matrix (rows=Actual 0/1/2, cols=Predicted 0/1/2):\n" + str(cm))

    (out / "stack_summary_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines[:35]))
    print(f"\nSaved Master CSV -> {out / 'master_comparison.csv'}")
    print(f"Saved Full Text Report -> {out / 'stack_summary_report.txt'}")


if __name__ == "__main__":
    main()