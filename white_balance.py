#!/usr/bin/env python3
"""
STEP 1 - White balance / standardisation of every photo.

Automatically detects whether each photo is on flat bond 'paper' or in a white 'container'
(even when a pattern folder like 010 has a mix of both), or you can force specific folders
with --container / --paper.

    python 01_white_balance.py data --out standardized --n-per-pattern 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from skimage.segmentation import find_boundaries

from soil_common import (
    TARGET_WHITE,
    clean_reference,
    detect_mode,
    find_white,
    lin_to_srgb,
    list_images,
    load_ann,
    load_srgb,
    make_sheet,
    poly_mask,
    set_scale,
    srgb_to_lin,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder with one sub-folder per pattern (all of them are read)")
    ap.add_argument("--out", default="standardized")
    ap.add_argument("--annotations", default="annotations")
    ap.add_argument("--n-per-pattern", type=int, default=5, help="0 = use all photos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patterns", nargs="*", default=None, help="only these pattern folders")
    ap.add_argument("--container", nargs="*", default=[], help="force these pattern folders to container mode")
    ap.add_argument("--paper", nargs="*", default=[], help="force these pattern folders to paper mode")
    ap.add_argument("--container-wb", choices=["auto", "none"], default="auto")
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument(
        "--normalise-exposure",
        action="store_true",
        help="also scale brightness so the paper is a fixed white (default: colour cast only, tested better)",
    )
    a = ap.parse_args()

    out = Path(a.out)
    (out / "wb_check").mkdir(parents=True, exist_ok=True)
    set_scale(a.max_side)
    items = list_images(a.root, a.patterns, a.n_per_pattern, a.seed)
    if not items:
        raise SystemExit("No images found. Expected data/<pattern>/<photo>.jpg")
    print(f"{len(set(p for p, _ in items))} patterns, {len(items)} photos")
    rows, tiles = [], []
    for i, (pat, path) in enumerate(items, 1):
        rgb, cam = load_srgb(path, a.max_side)
        h, w = rgb.shape[:2]
        ann = load_ann(a.annotations, pat, path.stem)
        white = find_white(rgb)

        # Determine mode per-photo (supports mixed folders!)
        if ann and ann.get("mode") in ("container", "paper"):
            mode = ann["mode"]
        elif pat in a.container:
            mode = "container"
        elif pat in a.paper:
            mode = "paper"
        else:
            mode = detect_mode(rgb, white)
        container = mode == "container"

        if ann and ann.get("white"):
            ref_mask, src = poly_mask(ann["white"], h, w), "manual"
            if not ref_mask.any():
                ref_mask, src = clean_reference(rgb, white, is_container=container), "auto"
        else:
            ref_mask, src = clean_reference(rgb, white, is_container=container), "auto"

        lin = srgb_to_lin(rgb)
        ref = np.maximum(np.median(lin[ref_mask], axis=0), 1e-6)
        ref_lab = np.median(rgb2lab(rgb)[ref_mask], axis=0)
        flags = []
        if container and a.container_wb == "none":
            gain = np.ones(3)
            flags.append("WB skipped (container-wb none)")
        else:
            gain = (TARGET_WHITE / ref) if a.normalise_exposure else (ref.mean() / ref)
            rel = gain / gain.mean()
            if container and src == "auto":
                flags.append("container wall used as reference (translucent): low confidence")
            if np.hypot(ref_lab[1], ref_lab[2]) > 8:
                flags.append(f"reference not neutral (chroma {np.hypot(ref_lab[1], ref_lab[2]):.1f})")
            if np.abs(rel - 1).max() > 0.12:
                flags.append("large correction - verify reference")
            if src == "auto" and white.mean() < 0.10:
                flags.append("little white reference found")

        rgb_wb = lin_to_srgb(np.clip(lin * gain, 0, 1))
        dest = out / pat / f"{path.stem}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((rgb_wb * 255).astype(np.uint8)).save(dest)

        before = rgb.copy()
        before[find_boundaries(ref_mask, mode="thick")] = [0, 0.6, 1]
        pair = Image.fromarray((np.hstack([before, rgb_wb]) * 255).astype(np.uint8))
        pair.thumbnail((520, 520))
        pair.save(out / "wb_check" / f"{pat}_{path.stem}.jpg", quality=80)
        tiles.append((pat, pair))
        g = gain / gain.mean()
        rows.append({
            "pattern": pat,
            "file": path.name,
            "stem": path.stem,
            "camera": cam,
            "mode": mode,
            "ref_source": src,
            "ref_L": ref_lab[0],
            "ref_a": ref_lab[1],
            "ref_b": ref_lab[2],
            "gain_R": g[0],
            "gain_G": g[1],
            "gain_B": g[2],
            "flag": "; ".join(flags),
        })
        print(f"[{i}/{len(items)}] {pat}/{path.name} [{mode}] gains {np.round(g, 3)}  {'; '.join(flags) or 'ok'}")

    new, rp = pd.DataFrame(rows), out / "wb_report.csv"
    if rp.exists():     # partial re-run (e.g. --patterns 102): keep the rows of the other photos
        old = pd.read_csv(rp, dtype=str)
        done = set(zip(new["pattern"].astype(str), new["stem"].astype(str)))
        old = old[[(str(p), str(s)) not in done for p, s in zip(old["pattern"], old["stem"])]]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.sort_values(["pattern", "file"]).to_csv(rp, index=False)
    make_sheet(tiles, out / "wb_sheet.jpg", tw=300, th=330)
    n = int((new["flag"].fillna("") != "").sum())
    print(f"\nDone. {n} photo(s) flagged in {rp}. Next: python 02_lasso.py auto")


if __name__ == "__main__":
    main()