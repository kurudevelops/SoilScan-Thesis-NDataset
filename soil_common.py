"""Shared helpers for the soil Lab pipeline. Keep this file next to the three step scripts."""
import io
import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageOps, ImageFile
from scipy import ndimage as ndi
from skimage.color import rgb2hsv, rgb2lab
from skimage.draw import polygon as draw_polygon
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image, disk

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CH = ["L", "a", "b"]
TARGET_WHITE = 0.85
SCALE = 1.0          # morphology radii are tuned for a 600 px working size


def set_scale(long_side):
    global SCALE
    SCALE = long_side / 600.0


def sc(r):
    return max(1, int(round(r * SCALE)))


# ---------------------------------------------------------------- colour / io
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def load_srgb(path, max_side):
    """Load, fix orientation, convert any embedded profile (e.g. Display P3) to sRGB, downscale."""
    img = Image.open(path)
    icc, exif = img.info.get("icc_profile"), img.getexif()
    cam = f"{exif.get(271, '')} {exif.get(272, '')}".replace("\x00", "").strip() or f"no-EXIF/{Path(path).name[:7]}"
    img = ImageOps.exif_transpose(img).convert("RGB")
    if icc:
        try:
            img = ImageCms.profileToProfile(
                img,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except Exception:
            pass
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(img).astype(np.float64) / 255.0, cam


def load_plain(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0


# ---------------------------------------------------------------- morphology
def largest_cc(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    return lab == 1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1))))


def op_open(m, r):
    return ndi.binary_opening(m, structure=disk(sc(r)))


def op_erode(m, r):
    return ndi.binary_erosion(m, structure=disk(sc(r)))


# ---------------------------------------------------------------- white reference + auto soil
def find_white(rgb):
    """Loose bright/neutral region (paper or container wall); defines the ROI around the soil."""
    hsv = rgb2hsv(rgb)
    V, S = hsv[..., 2], hsv[..., 1]
    cand = (V > 0.65 * np.percentile(V, 97)) & (S < 0.22)
    return largest_cc(op_open(cand, 5))


def detect_mode(rgb, white):
    """
    Auto-detect whether a single photo is shot in a white 'container' or on flat 'paper'.
    Container photos have:
      - White bowl walls extending flush to opposite edges of the frame, AND
      - Soil touching the top/bottom (or left/right) frame edge OR stronger L* shading across the curved white wall.
    """
    if not white.any():
        return "container"
    h, w = white.shape
    lab = rgb2lab(rgb)
    L = lab[..., 0]

    # Check how much of the image border is covered by the white mask vs dark soil
    top_w, bot_w = white[0, :].mean(), white[-1, :].mean()
    left_w, right_w = white[:, 0].mean(), white[:, -1].mean()

    # Check if soil cuts across the border between white walls (like in close-up container bowls)
    roi = convex_hull_image(white)
    paper_L = np.median(L[white])
    dark = roi & (L < paper_L - 12)
    border_dark = (
        dark[: max(2, h // 30), :].mean() > 0.25
        or dark[-max(2, h // 30) :, :].mean() > 0.25
        or dark[:, : max(2, w // 30)].mean() > 0.25
        or dark[:, -max(2, w // 30) :].mean() > 0.25
    )

    # Check brightness variation (IQR) on the white region (curved bowl walls have stronger shading gradients than flat paper)
    w_L = L[white]
    l_iqr = float(np.percentile(w_L, 90) - np.percentile(w_L, 10)) if w_L.size else 0.0
    flush_walls = (left_w > 0.75 and right_w > 0.75) or (top_w > 0.75 and bot_w > 0.75)

    if border_dark and (flush_walls or l_iqr > 11.0):
        return "container"
    if flush_walls and l_iqr > 14.0:
        return "container"
    return "paper"


def clean_reference(rgb, white, is_container=False):
    """Strict, chroma-consistent subset of the white region used only to estimate the illuminant."""
    if not white.any():
        return np.ones(rgb.shape[:2], dtype=bool)
    lab = rgb2lab(rgb)
    inner = op_erode(white, 10)
    if inner.sum() < 500:
        inner = white
    med = np.median(lab[..., 1:][inner], axis=0)
    close = inner & (np.hypot(lab[..., 1] - med[0], lab[..., 2] - med[1]) < 5)
    if close.sum() < 300:
        close = inner
    if close.any():
        # For curved containers, keep only the brighter top 40% (60th percentile) to avoid inner bowl shadows
        pct = 60 if is_container else 40
        close = close & (lab[..., 0] >= np.percentile(lab[..., 0][close], pct))
    return close if close.any() else white


def auto_soil(lin_wb, white):
    if white.sum() < 100:
        return np.zeros(lin_wb.shape[:2], dtype=bool), 1.0
    lab = rgb2lab(lin_to_srgb(lin_wb))
    roi, L = convex_hull_image(white), lab[..., 0]
    if not roi.any():
        return np.zeros(lin_wb.shape[:2], dtype=bool), 1.0
    inner = op_erode(white, 12)
    paper_L = np.median(L[inner]) if inner.any() else np.median(L[white])
    try:
        otsu_t = threshold_otsu(L[roi])
    except ValueError:
        otsu_t = paper_L - 12
    m = roi & (L < min(otsu_t, paper_L - 12))
    m = ndi.binary_fill_holes(largest_cc(op_open(m, 4)))
    m = op_erode(m, 8)                                    # trim blurry rim / shadow halo
    roi_edge = roi & ~ndi.binary_erosion(roi, iterations=sc(3))
    sb = m & ~ndi.binary_erosion(m)
    near = ndi.binary_dilation(roi_edge, iterations=sc(12))
    contact = float((sb & near).sum() / max(sb.sum(), 1))   # soil cut off by the ROI edge?
    return m, contact


def check_auto(white_frac, soil_frac, contact, is_container=False):
    why = []
    if white_frac < 0.15:
        why.append("little white reference found")
    if soil_frac < 0.03 or soil_frac > (0.75 if is_container else 0.60):
        why.append("implausible soil area")
    if contact > (0.45 if is_container else 0.20):
        why.append("soil mask cut off at ROI edge")
    return "; ".join(why)


# ---------------------------------------------------------------- annotations / masks
def ann_path(ann_dir, pat, stem):
    return Path(ann_dir) / pat / f"{stem}.json"


def load_ann(ann_dir, pat, stem):
    p = ann_path(ann_dir, pat, stem)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_ann(ann_dir, pat, stem, **keys):
    p = ann_path(ann_dir, pat, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d.update({k: v for k, v in keys.items() if v is not None})
    p.write_text(json.dumps(d), encoding="utf-8")


def mask_path(mask_dir, pat, stem):
    return Path(mask_dir) / pat / f"{stem}.png"


def save_mask(mask_dir, pat, stem, m):
    p = mask_path(mask_dir, pat, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((m * 255).astype(np.uint8)).save(p)

def save_isolated(iso_dir, pat, stem, rgb, m):
    """Save the white-balanced photo with everything outside the lasso mask removed (transparent PNG)."""
    p = Path(iso_dir) / pat / f"{stem}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    rgba[..., 3] = (m > 0).astype(np.uint8) * 255
    # Optional: crop tightly to the soil bounding box so the soil fills the frame
    ys, xs = np.where(m > 0)
    if len(ys) > 0 and len(xs) > 0:
        rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    Image.fromarray(rgba, mode="RGBA").save(p)    


def poly_mask(polys, h, w):
    m = np.zeros((h, w), bool)
    for poly in polys:
        pts = np.asarray(poly, float)
        if len(pts) < 3:
            continue
        rr, cc = draw_polygon(pts[:, 1] * h, pts[:, 0] * w, (h, w))
        m[rr, cc] = True
    return m


# ---------------------------------------------------------------- dataset listing / sheets
def list_images(root, patterns=None, n_per=5, seed=42):
    """Every sub-folder of `root` is a pattern; n_per images are sampled per pattern (0 = all)."""
    out = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if patterns and d.name not in patterns:
            continue
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in EXTS)
        if n_per and len(files) > n_per:
            files = sorted(random.Random(f"{seed}-{d.name}").sample(files, n_per))
        out += [(d.name, f) for f in files]
    return out


def npk(pattern):
    """'010' -> {'N':0,'P':1,'K':0}; None if the folder name is not a 3-digit 0/1/2 code."""
    if re.fullmatch(r"[0-2]{3}", str(pattern)):
        return {"N": int(pattern[0]), "P": int(pattern[1]), "K": int(pattern[2])}
    return None


def make_sheet(tiles, path, tw=170, th=300):
    """tiles = [(pattern, PIL image)] -> one column per pattern."""
    if not tiles:
        return
    pats = sorted({p for p, _ in tiles})
    per = {p: [t for q, t in tiles if q == p] for p in pats}
    rows = max(len(v) for v in per.values())
    sheet = Image.new("RGB", (tw * len(pats), th * rows + 18), "white")
    d = ImageDraw.Draw(sheet)
    for ci, p in enumerate(pats):
        d.text((ci * tw + 4, 2), str(p), fill="black")
        for ri, t in enumerate(per[p]):
            t = t.copy()
            t.thumbnail((tw - 4, th - 4))
            sheet.paste(t, (ci * tw, 18 + ri * th))
    sheet.save(path, quality=85)