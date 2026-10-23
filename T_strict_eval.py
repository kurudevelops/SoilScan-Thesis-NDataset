#!/usr/bin/env python3
"""
Strict re-evaluation of YOUR SVM / XGBoost code (imports fit_predict_svm and fit_predict_xgb unchanged).
Put this file next to T_SVM.py, T_XGBoost.py, soil_stack_common.py and soil_common.py, then run:

    py T_strict_eval.py                 # full grid (a few minutes)
    py T_strict_eval.py --models svm    # SVM only, much faster

It answers four questions with one table:
  1. features : color only (39) / color+texture (79) / color+meta (48) / all 88   -> do the meta columns (camera, mode, soil-area) inflate scores?
  2. camera   : all phones pooled vs iPhone only                                    -> are results mostly about which phone took the photo?
  3. cv       : blocks   = hold out blocks of photos (your current test)
                pattern  = hold out WHOLE patterns (N/P/K only) -> does colour generalise to NEW nutrient combinations?
  4. baseline : the score of always guessing the most common level

  (use --block-size 20 for a stricter "blocks" test that holds out whole sessions)
  5. phone test (always run at the end, on folders that contain 2+ phones): train on some phones, test on a phone the model
     never saw, for the SAME patterns. This is the test that matters for folders mixed with several phones.

Phone names come from the photo metadata. Photos without metadata (shown as no-EXIF) are ignored by the phone test; if you know which
phone shot them, pass --phone-map phones.csv (columns: pattern,phone) and they are labelled accordingly.

Output: models_out/strict_eval.csv, models_out/phone_test.csv and printed summary tables.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from soil_stack_common import get_base_dataset, get_oof_probabilities
from T_SVM import fit_predict_svm
from T_XGBoost import fit_predict_xgb

FITTERS = {"svm": fit_predict_svm, "xgb": fit_predict_xgb}


def phone_test(df_all, X88_all, a, feature_sets):
    """Leave-one-phone-out on the patterns that were photographed with 2+ phones."""
    known = ~df_all["camera"].str.lower().str.startswith("no-exif").values
    df, X88 = df_all[known].reset_index(drop=True), X88_all[known]
    cnt = df.groupby(["pattern", "camera"]).size().rename("n").reset_index()
    cnt = cnt[cnt["n"] >= a.min_per_phone]
    mixed = cnt.groupby("pattern")["camera"].nunique()
    mixed = mixed[mixed >= 2].index
    keep = df.set_index(["pattern", "camera"]).index.isin(list(zip(cnt["pattern"], cnt["camera"]))) & df["pattern"].isin(mixed).values
    df, X88 = df[keep].reset_index(drop=True), X88[keep]
    phones = sorted(df["camera"].unique())
    if df["pattern"].nunique() < 2 or len(phones) < 2:
        print("\n[phone test skipped] need patterns photographed with 2+ phones (>= %d photos each)" % a.min_per_phone)
        return
    pat_classes = np.unique(df["pattern"].values)
    print(f"\n=== PHONE TEST: {df['pattern'].nunique()} patterns {list(pat_classes)} photographed with {len(phones)} phones: {phones} ===")
    print("    'blocks' = same phone seen in training (as before);  'phone' = the test phone was NEVER in training")
    rows = []
    for model, (fs_name, cols) in itertools.product(a.models, feature_sets.items()):
        X = X88[:, cols]
        row = {"model": model, "features": fs_name, "n_photos": len(df), "patterns": " ".join(pat_classes), "phones": " | ".join(phones)}
        for cv, groups, ns in (("blocks", df["group"].values, a.folds), ("phone", df["camera"].values, len(phones))):
            for nut in "NPK":
                y = df[nut].values
                oof = get_oof_probabilities(FITTERS[model], X, y, groups, np.array([0, 1, 2]), n_splits=ns)
                row[f"{nut}_balacc_{cv}"] = balanced_accuracy_score(y, np.array([0, 1, 2])[oof.argmax(1)])
            y = df["pattern"].values
            oof = get_oof_probabilities(FITTERS[model], X, y, groups, pat_classes, n_splits=ns)
            row[f"pattern_acc_{cv}"] = (pat_classes[oof.argmax(1)] == y).mean()
        rows.append(row)
        print(f"  {model:4s} {fs_name:18s} pattern acc: blocks {row['pattern_acc_blocks']:.2f} -> phone {row['pattern_acc_phone']:.2f}   "
              f"N {row['N_balacc_blocks']:.2f}->{row['N_balacc_phone']:.2f}  P {row['P_balacc_blocks']:.2f}->{row['P_balacc_phone']:.2f}  "
              f"K {row['K_balacc_blocks']:.2f}->{row['K_balacc_phone']:.2f}")
    print(f"    (chance for pattern = {1 / len(pat_classes):.2f}; chance for N/P/K balanced accuracy = 0.33)")
    Path("models_out").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("models_out/phone_test.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["svm", "xgb"], choices=list(FITTERS))
    ap.add_argument("--cameras", nargs="+", default=["all", "iPhone"])
    ap.add_argument("--min-photos", type=int, default=15, help="drop patterns with fewer photos in the chosen camera subset")
    ap.add_argument("--skip-pattern-target", action="store_true", help="skip the slow 19-class pattern target")
    ap.add_argument("--phone-map", default=None, help="csv with columns pattern,phone for photos that have no metadata")
    ap.add_argument("--min-per-phone", type=int, default=5, help="phone test: minimum photos of a pattern from a phone")
    ap.add_argument("--skip-phone-test", action="store_true")
    ap.add_argument("--folds", type=int, default=5, help="number of cross-validation folds for the 'blocks' test (model_compare.py uses 20)")
    ap.add_argument("--block-size", type=int, default=None,
                    help="photos per held-out block (default: BLOCK_SIZE in soil_stack_common.py = 20). Bigger blocks hold out whole sessions")
    a = ap.parse_args()

    df_all, _, X88_all = get_base_dataset(block_size=a.block_size)
    if a.phone_map:
        pm = pd.read_csv(a.phone_map, dtype=str).set_index("pattern")["phone"].to_dict()
        miss = df_all["camera"].str.lower().str.startswith("no-exif")
        df_all.loc[miss, "camera"] = df_all.loc[miss, "pattern"].map(pm).fillna(df_all.loc[miss, "camera"])
    feature_sets = {
        "color(39)": slice(0, 39),
        "color+texture(79)": slice(0, 79),
        "color+meta(48)": np.r_[0:39, 79:88],
        "all(88)": slice(0, 88),
    }
    rows = []
    for cam in a.cameras:
        keep = np.ones(len(df_all), bool) if cam.lower() == "all" else df_all["camera"].str.contains(cam, case=False).values
        df, X88 = df_all[keep].reset_index(drop=True), X88_all[keep]
        cnt = df.groupby("pattern").size()
        ok = df["pattern"].isin(cnt[cnt >= a.min_photos].index).values
        df, X88 = df[ok].reset_index(drop=True), X88[ok]
        if df["pattern"].nunique() < 3:
            print(f"[skip camera={cam}] fewer than 3 patterns with >= {a.min_photos} photos")
            continue
        block_groups, pat_groups = df["group"].values, df["pattern"].values
        pat_classes = np.unique(df["pattern"].values)
        print(f"\n=== camera={cam}: {len(df)} photos, {len(pat_classes)} patterns ===")

        for model, (fs_name, cols), cv in itertools.product(a.models, feature_sets.items(), ["blocks", "pattern"]):
            X = X88[:, cols]
            groups = block_groups if cv == "blocks" else pat_groups
            n_splits = a.folds if cv == "blocks" else 10_000          # 10_000 -> one fold per pattern (leave-one-pattern-out)
            row = {"camera": cam, "model": model, "features": fs_name, "cv": cv, "n_photos": len(df)}
            for nut in "NPK":
                y = df[nut].values
                oof = get_oof_probabilities(FITTERS[model], X, y, groups, np.array([0, 1, 2]), n_splits=n_splits)
                pred = np.array([0, 1, 2])[oof.argmax(1)]
                row[f"{nut}_acc"], row[f"{nut}_balacc"] = (pred == y).mean(), balanced_accuracy_score(y, pred)
                row[f"{nut}_majority"] = pd.Series(y).value_counts(normalize=True).max()
            if cv == "blocks" and not a.skip_pattern_target:
                y = df["pattern"].values
                oof = get_oof_probabilities(FITTERS[model], X, y, groups, pat_classes)
                row["pattern_acc"] = (pat_classes[oof.argmax(1)] == y).mean()
            rows.append(row)
            print(f"  {model:4s} {fs_name:18s} cv={cv:7s} N {row['N_balacc']:.2f}  P {row['P_balacc']:.2f}  K {row['K_balacc']:.2f}"
                  + (f"  pattern_acc {row['pattern_acc']:.2f}" if "pattern_acc" in row else ""))

    if not a.skip_phone_test:
        phone_test(df_all, X88_all, a, feature_sets)

    out = pd.DataFrame(rows)
    Path("models_out").mkdir(exist_ok=True)
    out.to_csv("models_out/strict_eval.csv", index=False)
    print("\nBalanced accuracy for N / P / K (guessing = 0.33). Saved to models_out/strict_eval.csv")
    print(out.pivot_table(index=["camera", "model", "features"], columns="cv", values=["N_balacc", "P_balacc", "K_balacc"]).round(2).to_string())


if __name__ == "__main__":
    main()