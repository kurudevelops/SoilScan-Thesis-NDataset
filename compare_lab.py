#!/usr/bin/env python3

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from skimage.color import deltaE_ciede2000, rgb2lab

from Backend.soil_common import CH, load_plain, npk


# ---------------------------------------------------------------- features
def collect(a):
    rep_path = Path(a.standardized) / "wb_report.csv"
    if not rep_path.exists():
        raise SystemExit(f"Missing {rep_path}. Run 01_white_balance.py first.")
    rep = pd.read_csv(rep_path, dtype=str).fillna("")
    rows, missing = [], []
    for r in rep.itertuples():
        mp = Path(a.masks) / r.pattern / f"{r.stem}.png"
        sp = Path(a.standardized) / r.pattern / f"{r.stem}.png"
        if not mp.exists() or not sp.exists():
            missing.append(f"{r.pattern}/{r.file}")
            continue
        rgb = load_plain(sp)
        m = np.asarray(Image.open(mp)) > 0
        if m.shape != rgb.shape[:2] or m.sum() < 200:
            missing.append(f"{r.pattern}/{r.file} (empty or mismatched mask)")
            continue
        lab = rgb2lab(rgb)[m]
        row = {
            "pattern": r.pattern,
            "file": r.file,
            "camera": r.camera,
            "mode": r.mode,
            "wb_flag": r.flag,
            "n_pixels": int(m.sum()),
        }
        for i, c in enumerate(CH):
            row[c], row[c + "_std"] = float(np.median(lab[:, i])), float(lab[:, i].std())
        rows.append(row)
    return pd.DataFrame(rows), missing


# ---------------------------------------------------------------- statistics
def range_overlap(df, pats):
    rg = df.groupby("pattern")[CH].agg(["min", "max"])
    out = {}
    for c in CH:
        m = pd.DataFrame(1.0, index=pats, columns=pats)
        for p, q in combinations(pats, 2):
            lo = max(rg.loc[p, (c, "min")], rg.loc[q, (c, "min")])
            hi = min(rg.loc[p, (c, "max")], rg.loc[q, (c, "max")])
            un = max(rg.loc[p, (c, "max")], rg.loc[q, (c, "max")]) - min(rg.loc[p, (c, "min")], rg.loc[q, (c, "min")])
            m.loc[p, q] = m.loc[q, p] = max(0.0, hi - lo) / un if un > 0 else 0.0
        out[c] = m
    return out


def delta_e_matrix(cent, pats):
    m = pd.DataFrame(0.0, index=pats, columns=pats)
    for p, q in combinations(pats, 2):
        m.loc[p, q] = m.loc[q, p] = float(
            deltaE_ciede2000(cent.loc[p].values[None, None, :], cent.loc[q].values[None, None, :]).squeeze()
        )
    return m


def mahalanobis_matrix(df, cent, pats):
    res = df[CH].values - cent.loc[df["pattern"]].values
    inv = np.linalg.inv(res.T @ res / max(len(df) - len(pats), 1) + 1e-6 * np.eye(3))   # pooled covariance
    m = pd.DataFrame(0.0, index=pats, columns=pats)
    for p, q in combinations(pats, 2):
        d = cent.loc[p].values - cent.loc[q].values
        m.loc[p, q] = m.loc[q, p] = float(np.sqrt(d @ inv @ d))
    return m


def svm_cv(X, y, groups=None):
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, class_weight="balanced"))
    if groups is None:
        return cross_val_predict(clf, X, y, cv=LeaveOneOut())
    pred = np.empty(len(y), dtype=y.dtype)          # leave one whole pattern out per fold
    for g in np.unique(groups):
        te = groups == g
        tr_y = y[~te]
        if len(set(tr_y)) < 2:                      # training side has a single class: it can only guess that class
            pred[te] = tr_y[0]
        else:
            pred[te] = clf.fit(X[~te], tr_y).predict(X[te])
    return pred


# ---------------------------------------------------------------- plots
def plots(df, de, mh, out):
    pats = sorted(df["pattern"].unique())
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    for i, p in enumerate(pats):
        s, col = df[df["pattern"] == p], cmap(i % 20)
        # Distinguish paper (circles) vs container (triangles) in the scatter plot
        s_pap = s[s["mode"] == "paper"]
        s_con = s[s["mode"] == "container"]
        if len(s_pap):
            ax.scatter(s_pap["a"], s_pap["b"], color=col, marker="o", s=28)
        if len(s_con):
            ax.scatter(s_con["a"], s_con["b"], color=col, marker="^", s=34, edgecolors="k", linewidths=0.4)
        if len(s) >= 3:
            vals, vecs = np.linalg.eigh(np.cov(s[["a", "b"]].values.T))
            wd, ht = 4 * np.sqrt(np.maximum(vals[::-1], 1e-9))
            ax.add_patch(
                Ellipse(
                    s[["a", "b"]].mean().values,
                    wd,
                    ht,
                    angle=np.degrees(np.arctan2(*vecs[:, 1][::-1])),
                    fill=False,
                    color=col,
                    lw=1.2,
                )
            )
        ax.annotate(p, s[["a", "b"]].mean().values, fontsize=8, weight="bold")
    ax.set(xlabel="a*", ylabel="b*", title="a*-b* plane (circle = paper, triangle = container, ellipse ~ 2 SD)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "ab_scatter.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.32 * len(pats) + 2)), sharey=True)
    for ax, c in zip(axes, CH):
        g = df.groupby("pattern")[c].agg(["min", "max", "mean"]).loc[pats]
        for i, p in enumerate(pats):
            ax.hlines(i, g.loc[p, "min"], g.loc[p, "max"], lw=6, color=cmap(i % 20))
            ax.plot(g.loc[p, "mean"], i, "k|", ms=10)
        ax.set_yticks(range(len(pats)))
        ax.set_yticklabels(pats)
        ax.set_title(f"{c}* range")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "channel_ranges.png", dpi=150)
    plt.close(fig)

    for m, title, name in [
        (de, "Centre deltaE2000 (low = similar)", "deltaE_heatmap.png"),
        (mh, "Pooled Mahalanobis (low = similar)", "mahalanobis_heatmap.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(m.values, cmap="viridis_r")
        ax.set_xticks(range(len(m)))
        ax.set_xticklabels(m.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m.index, fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)

    if all(npk(p) for p in pats):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
        for ax, nut in zip(axes, "NPK"):
            lv = df["pattern"].map(lambda p: npk(p)[nut])
            sc_ = ax.scatter(df["a"], df["b"], c=lv, cmap="viridis", s=26, vmin=0, vmax=2)
            ax.set(title=f"coloured by {nut} level (0 low, 1 medium, 2 high)", xlabel="a*")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("b*")
        fig.colorbar(sc_, ax=axes, ticks=[0, 1, 2])
        fig.savefig(out / "npk_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standardized", default="standardized")
    ap.add_argument("--masks", default="masks")
    ap.add_argument("--out", default="results")
    ap.add_argument("--drop-wb-flagged", action="store_true", help="exclude photos whose white balance was flagged in step 1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df, missing = collect(a)
    dropped = []
    if a.drop_wb_flagged and len(df):
        dropped = [f"{r.pattern}/{r.file}: {r.wb_flag}" for r in df.itertuples() if r.wb_flag]
        df = df[df["wb_flag"] == ""]
    pats = sorted(df["pattern"].unique()) if len(df) else []
    if len(pats) < 2:
        raise SystemExit("Fewer than 2 patterns have masks. Run 02_lasso.py auto / manual first.")
    df.to_csv(out / "image_features.csv", index=False)
    cent = df.groupby("pattern")[CH].mean()
    summ = df.groupby("pattern")[CH].agg(["count", "mean", "std", "min", "max"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.reset_index().to_csv(out / "pattern_summary.csv", index=False)
    ov, de, mh = range_overlap(df, pats), delta_e_matrix(cent, pats), mahalanobis_matrix(df, cent, pats)
    for c in CH:
        ov[c].to_csv(out / f"range_overlap_{c}.csv")
    de.to_csv(out / "deltaE2000_centres.csv")
    mh.to_csv(out / "mahalanobis_centres.csv")

    pairs = []
    for p, q in combinations(pats, 2):
        d = {"A": p, "B": q, "dE2000": de.loc[p, q], "mahalanobis": mh.loc[p, q]}
        d.update({f"overlap_{c}": ov[c].loc[p, q] for c in CH})
        d["all3_overlap"] = min(d[f"overlap_{c}"] for c in CH) > 0
        d["ab_overlap"] = min(d["overlap_a"], d["overlap_b"]) > 0
        if npk(p) and npk(q):
            d["npk_digits_differ"] = sum(x != y for x, y in zip(p, q))
        pairs.append(d)
    pairs = pd.DataFrame(pairs).sort_values("dE2000")
    pairs.to_csv(out / "pairwise_table.csv", index=False)
    plots(df, de, mh, out)

    L = [
        f"Patterns: {len(pats)} | photos used: {len(df)}",
        "photos per pattern: " + ", ".join(f"{p}={n}" for p, n in df["pattern"].value_counts().sort_index().items()),
    ]
    if missing:
        L += ["", f"== Skipped, no usable mask ({len(missing)}) =="] + missing
    if dropped:
        L += ["", "== Dropped, white balance flagged =="] + dropped
    L += [
        "",
        "== Source camera per pattern (a pattern shot on a single phone can be confounded with that phone) ==",
        df.groupby(["pattern", "camera"]).size().to_string(),
    ]
    L += [
        "",
        "== Background mode per pattern (paper vs container mix) ==",
        df.groupby(["pattern", "mode"]).size().unstack(fill_value=0).to_string(),
    ]

    # Check if mixed patterns show a systematic difference between paper and container
    mixed_pats = [
        p for p, g in df.groupby("pattern") if set(g["mode"]) == {"paper", "container"}
    ]
    if mixed_pats:
        L += ["", "== Within-pattern Paper vs Container shift (for mixed folders) =="]
        mode_rows = []
        for p in mixed_pats:
            sub = df[df["pattern"] == p]
            m_pap = sub[sub["mode"] == "paper"][CH].mean()
            m_con = sub[sub["mode"] == "container"][CH].mean()
            de_pc = float(
                deltaE_ciede2000(m_pap.values[None, None, :], m_con.values[None, None, :]).squeeze()
            )
            dL = float(m_con["L"] - m_pap["L"])
            da = float(m_con["a"] - m_pap["a"])
            db = float(m_con["b"] - m_pap["b"])
            L.append(
                f"{p}: dE2000(paper vs container)={de_pc:.2f} | "
                f"dL={dL:+.1f}, da={da:+.1f}, db={db:+.1f} "
                f"(n_paper={int((sub['mode'] == 'paper').sum())}, n_container={int((sub['mode'] == 'container').sum())})"
            )
            mode_rows.append({
                "pattern": p,
                "n_paper": int((sub["mode"] == "paper").sum()),
                "n_container": int((sub["mode"] == "container").sum()),
                "dE2000_paper_vs_container": de_pc,
                "dL_container_minus_paper": dL,
                "da_container_minus_paper": da,
                "db_container_minus_paper": db,
            })
        pd.DataFrame(mode_rows).to_csv(out / "mixed_mode_shifts.csv", index=False)

    L += ["", "== Pattern pairs whose L,a,b ranges ALL overlap =="]
    L += [
        f"{r.A} vs {r.B}: dE2000={r.dE2000:.2f} mahalanobis={r.mahalanobis:.2f}"
        for r in pairs[pairs.all3_overlap].itertuples()
    ] or ["none"]
    L += ["", "== Pairs overlapping on a*,b* only (ignoring L) =="]
    L += [f"{r.A} vs {r.B}: dE2000={r.dE2000:.2f}" for r in pairs[pairs.ab_overlap].itertuples()] or ["none"]
    L += ["", "== 20 closest pattern pairs (dE2000 < ~2.3 is barely noticeable) =="]
    L += [f"{r.A} vs {r.B}: {r.dE2000:.2f}" for r in pairs.head(20).itertuples()]

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    y = df["pattern"].to_numpy(dtype=str)
    if len(df) > len(pats):
        for name, feats in [("L,a,b", CH), ("a,b only (ignores brightness)", ["a", "b"])]:
            pred = svm_cv(df[feats].values, y)
            labs = sorted(set(y))
            cm = confusion_matrix(y, pred, labels=labs)
            conf = sorted(
                ((int(cm[i, j]), labs[i], labs[j]) for i in range(len(labs)) for j in range(len(labs)) if i != j and cm[i, j]),
                reverse=True,
            )[:12]
            L += [
                "",
                f"== Leave-one-out SVM, identify the pattern from {name} ==",
                f"accuracy={np.mean(pred == y):.3f}  macro-F1={f1_score(y, pred, average='macro', zero_division=0):.3f}  (chance ~ {1 / len(pats):.2f})",
                classification_report(y, pred, zero_division=0),
                "most confused (count, true, predicted): " + str(conf),
            ]

    if all(npk(p) for p in pats):
        df2 = df.copy()
        for nut in "NPK":
            df2[nut] = df2["pattern"].map(lambda p: npk(p)[nut])
        lv = pd.concat({n: df2.groupby(n)[CH].agg(["mean", "std", "count"]) for n in "NPK"}, names=["nutrient", "level"])
        lv.columns = [f"{c}_{s}" for c, s in lv.columns]
        lv.to_csv(out / "nutrient_level_means.csv")
        corr = []
        for nut in "NPK":
            for c in CH:
                rho, pv = spearmanr(df2[nut], df2[c])
                corr.append({"nutrient": nut, "channel": c, "spearman_rho": rho, "p": pv})
        corr = pd.DataFrame(corr)
        corr.to_csv(out / "nutrient_correlations.csv", index=False)
        L += ["", "== Does Lab colour track each nutrient level? (Spearman, photo level) =="] + [
            f"{r.nutrient} vs {r.channel}*: rho={r.spearman_rho:+.2f} (p={r.p:.3g})" for r in corr.itertuples()
        ]
        L += ["", "== Predict each nutrient level from L,a,b, leaving out WHOLE patterns (honest test) =="]
        for nut in "NPK":
            yy = df2[nut].values
            if len(set(yy)) < 2:
                continue
            pred = svm_cv(df2[CH].values, yy, groups=df2["pattern"].values)
            base = pd.Series(yy).value_counts(normalize=True).max()
            L.append(f"{nut}: accuracy={np.mean(pred == yy):.3f} vs always-guess-majority={base:.3f}")
        L += [
            "",
            "Colour mainly reflects organic matter, moisture and iron; a colour difference between patterns is not",
            "proof it is caused by N, P or K. Treat these as evidence of separability, not of causation.",
        ]
    L += ["", "Note: with ~5 photos per pattern the ranges understate real variation. Treat results as preliminary."]
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:2]) + f"\n\nDone. Results saved in '{out}'.")


if __name__ == "__main__":
    main()