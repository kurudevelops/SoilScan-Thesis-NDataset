#!/usr/bin/env python3
"""
STEP 2 - Isolate the soil (remove paper / container / desk) and save one mask + isolated soil photo per image.
Upgraded with HSV Saturation thresholding to ignore dark shadows on white containers.

    python 02_lasso.py auto                 # auto-mask every photo, list the ones that need manual review
    python 02_lasso.py manual               # lasso the photos listed in lasso_todo.csv
    python 02_lasso.py manual --white       # also lasso a clean white patch (for container photos)

Outputs:
  - masks/<pattern>/<photo>.png             (binary mask)
  - masks/_qc/<pattern>_<photo>.jpg         (white-balanced photo with green lasso outline)
  - isolated/<pattern>/<photo>.png          (white-balanced soil ONLY, background removed & cropped)
  - isolated/isolated_sheet.jpg             (grid comparing isolated soil across all NPK patterns)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import find_boundaries

from Backend.soil_common import (
    load_ann,
    load_plain,
    make_sheet,
    poly_mask,
    save_ann,
    save_isolated,
    save_mask,
    set_scale,
)


def make_iso_preview(rgb, m):
    """Create an RGB preview of the isolated soil on a clean dark-grey background for contact sheets."""
    bg = np.full_like(rgb, 0.15)
    bg[m > 0] = rgb[m > 0]
    ys, xs = np.where(m > 0)
    if len(ys) > 0 and len(xs) > 0:
        bg = bg[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    im = Image.fromarray((bg * 255).astype(np.uint8))
    im.thumbnail((300, 300))
    return im


def auto_soil_hsv(rgb):
    """New HSV-based soil detector: uses color saturation with edge erosion."""
    bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Otsu threshold strictly on saturation
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up internal noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # THE FIX: Erode the outer boundary to shave off the white paper fringe
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, erode_kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(rgb.shape[:2], dtype=bool)
        
    largest_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return final_mask > 0

def cmd_auto(a):
    rep_path = Path(a.standardized) / "wb_report.csv"
    if not rep_path.exists():
        raise SystemExit(f"Missing {rep_path}. Run 01_white_balance.py first.")
    rep = pd.read_csv(rep_path, dtype=str).fillna("")
    qcdir = Path(a.masks) / "_qc"
    qcdir.mkdir(parents=True, exist_ok=True)
    Path(a.isolated).mkdir(parents=True, exist_ok=True)

    tiles, iso_tiles, todo = [], [], []
    for i, r in enumerate(rep.itertuples(), 1):
        rgb = load_plain(Path(a.standardized) / r.pattern / f"{r.stem}.png")
        h, w = rgb.shape[:2]
        set_scale(600)
        ann = load_ann(a.annotations, r.pattern, r.stem)
        flag, soil, src = "", None, "auto"

        # 1. Use manual annotation if it exists
        if ann and ann.get("soil"):
            soil, src = poly_mask(ann["soil"], h, w), "manual"
        # 2. Otherwise use the robust HSV saturation detector
        else:
            soil = auto_soil_hsv(rgb)
            src = "auto"
            # Simple QC flag if the mask seems way too small or way too big
            soil_ratio = float(soil.mean())
            if soil_ratio < 0.05 or soil_ratio > 0.85:
                flag = "implausible soil area"

        keep = soil is not None and soil.any() and (not flag or a.keep_flagged)
        if keep:
            save_mask(a.masks, r.pattern, r.stem, soil)
            save_isolated(a.isolated, r.pattern, r.stem, rgb, soil)
            iso_tiles.append((r.pattern, make_iso_preview(rgb, soil)))
        if flag and src == "auto":
            todo.append({"pattern": r.pattern, "file": r.file, "stem": r.stem, "flag": flag})

        ov = rgb.copy()
        if soil is not None and soil.any():
            ov[find_boundaries(soil, mode="thick")] = [0, 1, 0] if keep else [1, 0.5, 0]
        im = Image.fromarray((ov * 255).astype(np.uint8))
        im.thumbnail((360, 640))
        im.save(qcdir / f"{r.pattern}_{r.stem}.jpg", quality=80)
        tiles.append((r.pattern, im))
        print(f"[{i}/{len(rep)}] {r.pattern}/{r.file}  {src}  {flag or 'ok'}")

    make_sheet(tiles, Path(a.masks) / "mask_sheet.jpg")
    make_sheet(iso_tiles, Path(a.isolated) / "isolated_sheet.jpg", tw=220, th=220)
    pd.DataFrame(todo, columns=["pattern", "file", "stem", "flag"]).to_csv("lasso_todo.csv", index=False)
    print(f"\n{len(todo)} photo(s) need a manual lasso -> lasso_todo.csv. Run: python lasso_2.py manual")


def draw_poly(rgb, title):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    st = {"verts": None, "skip": False, "quit": False}

    try:
        sel = PolygonSelector(ax, lambda v: st.update(verts=v), props=dict(color="lime", linewidth=1.5))
    except TypeError:
        sel = PolygonSelector(ax, lambda v: st.update(verts=v), lineprops=dict(color="lime", linewidth=1.5))

    def on_key(e):
        if e.key in ("enter", "x", "q"):
            if e.key == "enter" and not st["verts"] and hasattr(sel, "verts") and len(sel.verts) >= 3:
                st["verts"] = list(sel.verts)
            st["skip"], st["quit"] = e.key == "x", e.key == "q"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if st["quit"]:
        raise KeyboardInterrupt
    if st["skip"] or not st["verts"] or len(st["verts"]) < 3:
        return None
    h, w = rgb.shape[:2]
    return [[float(x) / w, float(y) / h] for x, y in st["verts"]]


def cmd_manual(a):
    rep_path = Path(a.standardized) / "wb_report.csv"
    if not rep_path.exists():
        raise SystemExit(f"Missing {rep_path}. Run 01_white_balance.py first.")
    rep = pd.read_csv(rep_path, dtype=str).fillna("")
    if a.all:
        todo = rep
    elif Path("lasso_todo.csv").exists():
        t = pd.read_csv("lasso_todo.csv", dtype=str)
        todo_pairs = set(zip(t["pattern"], t["stem"]))
        todo = rep[[(p, s) in todo_pairs for p, s in zip(rep["pattern"], rep["stem"])]]
    else:
        raise SystemExit("Run `python lasso_2.py auto` first (or use --all).")

    if not a.redo:
        todo = todo[[not (load_ann(a.annotations, r.pattern, r.stem) or {}).get("soil") for r in todo.itertuples()]]

    qcdir = Path(a.masks) / "_qc"
    qcdir.mkdir(parents=True, exist_ok=True)
    Path(a.isolated).mkdir(parents=True, exist_ok=True)

    print(f"{len(todo)} photo(s) to lasso. Click 3+ points around soil, then ENTER = save/next, X = skip, Q = quit.")
    try:
        for i, r in enumerate(todo.itertuples(), 1):
            rgb = load_plain(Path(a.standardized) / r.pattern / f"{r.stem}.png")
            soil = draw_poly(
                rgb,
                f"[{i}/{len(todo)}] {r.pattern}/{r.file}\nlasso the SOIL only, then ENTER (X = skip photo, Q = quit)",
            )
            if soil is None:
                continue
            white = None
            if a.white:
                white = draw_poly(
                    rgb,
                    f"{r.pattern}/{r.file}\nlasso a CLEAN patch of the white reference (paper, or container rim in direct light)\nthen ENTER (X = auto-detect)",
                )
            save_ann(a.annotations, r.pattern, r.stem, soil=[soil], white=[white] if white else None)
            h, w = rgb.shape[:2]
            m = poly_mask([soil], h, w)
            save_mask(a.masks, r.pattern, r.stem, m)
            save_isolated(a.isolated, r.pattern, r.stem, rgb, m)

            # Update QC overlay image with manual lasso boundary
            ov = rgb.copy()
            ov[find_boundaries(m, mode="thick")] = [0, 1, 0]
            im = Image.fromarray((ov * 255).astype(np.uint8))
            im.thumbnail((360, 640))
            im.save(qcdir / f"{r.pattern}_{r.stem}.jpg", quality=80)
    except KeyboardInterrupt:
        print("Stopped. Everything saved so far is kept.")
    if a.white:
        print("White patches saved: re-run 01_white_balance.py so they are applied.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("auto", "manual"):
        p = sub.add_parser(name)
        p.add_argument("--standardized", default="standardized")
        p.add_argument("--masks", default="masks")
        p.add_argument("--isolated", default="isolated")
        p.add_argument("--annotations", default="annotations")
        if name == "auto":
            p.add_argument("--keep-flagged", action="store_true", help="save auto masks even if the check failed")
            p.set_defaults(fn=cmd_auto)
        else:
            p.add_argument("--white", action="store_true", help="also lasso a white reference patch")
            p.add_argument("--all", action="store_true", help="lasso every photo, not only the flagged ones")
            p.add_argument("--redo", action="store_true", help="redo photos that already have a lasso")
            p.set_defaults(fn=cmd_manual)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()