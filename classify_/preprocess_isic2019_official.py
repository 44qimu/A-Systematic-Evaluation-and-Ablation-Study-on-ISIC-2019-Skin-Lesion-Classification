#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2


# ---------------------------
# 3x3 dilation / max filter with replicate border.
# ---------------------------
def dilation33(in2d: np.ndarray) -> np.ndarray:
    x = np.asarray(in2d)
    if x.ndim != 2:
        raise ValueError("dilation33 expects a 2D array")
    # vertical max (up, mid, down) with replicate
    up = np.vstack([x[1:, :], x[-1:, :]])
    mid = x
    down = np.vstack([x[:1, :], x[:-1, :]])
    out2 = np.maximum.reduce([up, mid, down])
    # horizontal max (left, mid, right) with replicate
    left = np.hstack([out2[:, 1:], out2[:, -1:]])
    mid2 = out2
    right = np.hstack([out2[:, :1], out2[:, :-1]])
    out = np.maximum.reduce([left, mid2, right])
    return out


# ---------------------------
# method=0 => border to 0
# method=1 => border filled with mean(interior)
# ---------------------------
def set_border(in2d: np.ndarray, width: int, method: int = 1) -> np.ndarray:
    x = np.asarray(in2d).astype(np.float64)
    if x.ndim != 2:
        raise ValueError("set_border expects a 2D array")
    h, w = x.shape
    width = int(width)
    if width <= 0:
        return x.copy()

    temp = np.ones((h, w), dtype=np.float64)
    temp[:width, :] = 0
    temp[-width:, :] = 0
    temp[:, :width] = 0
    temp[:, -width:] = 0

    out = temp * x
    if method == 1:
        denom = temp.sum()
        mean_inside = out.sum() / denom if denom > 0 else 0.0
        out = out + mean_inside * (1.0 - temp)
    return out


# ---------------------------
# replicate padding (edge)
# Included for completeness; we mostly use cv2 BORDER_REPLICATE.
# ---------------------------
def fill_border(x: np.ndarray, bw: int) -> np.ndarray:
    x = np.asarray(x)
    bw = int(bw)
    if bw <= 0:
        return x.copy()
    if x.ndim == 2:
        return np.pad(x, ((bw, bw), (bw, bw)), mode="edge")
    if x.ndim == 3:
        return np.pad(x, ((bw, bw), (bw, bw), (0, 0)), mode="edge")
    raise ValueError("fill_border expects 2D or 3D array")


# ---------------------------
# Needed only if you set sigma>0 or run Grey-Edge (njet>0).
# We use separable 1D Gaussian / derivative-of-Gaussian kernels.
# Border handling: replicate, matching fill_border behavior.
# ---------------------------
def _gauss_1d(sigma: float, order: int) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        # sigma==0: identity / finite-diff would be another choice; here keep simple.
        if order == 0:
            return np.array([1.0], dtype=np.float64)
        raise ValueError("sigma must be >0 for derivative orders >0")

    radius = int(math.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)

    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g / (g.sum() + 1e-12)

    if order == 0:
        return g

    if order == 1:
        # first derivative of Gaussian (unnormalized), then zero-mean
        dg = -x / (sigma * sigma) * g
        # ensure sum ~ 0
        dg = dg - dg.mean()
        return dg

    if order == 2:
        # second derivative
        ddg = ((x * x - sigma * sigma) / (sigma ** 4)) * g
        ddg = ddg - ddg.mean()
        return ddg

    raise ValueError("order must be 0,1,2")


def gDer(in2d: np.ndarray, sigma: float, dx: int, dy: int) -> np.ndarray:
    """Gaussian derivative filter of order (dx,dy) with replicate border."""
    img = np.asarray(in2d).astype(np.float64)

    if sigma == 0:
        # Provide a reasonable fallback for sigma==0 if someone calls it:
        if dx == 0 and dy == 0:
            return img.copy()
        raise ValueError("gDer with sigma==0 and derivative orders >0 is not supported in this minimal-equivalence version.")

    kx = _gauss_1d(sigma, dx).astype(np.float64)
    ky = _gauss_1d(sigma, dy).astype(np.float64)

    # cv2.sepFilter2D expects kernels shaped (k,1) and (1,k) but accepts 1D as well
    out = cv2.sepFilter2D(img, ddepth=cv2.CV_64F, kernelX=kx, kernelY=ky,
                         borderType=cv2.BORDER_REPLICATE)
    return out



def NormDerivative(in_rgb: np.ndarray, sigma: float, order: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.asarray(in_rgb).astype(np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("NormDerivative expects HxWx3")

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    if order == 1:
        Rx = gDer(R, sigma, 1, 0)
        Ry = gDer(R, sigma, 0, 1)
        Rw = np.sqrt(Rx * Rx + Ry * Ry)

        Gx = gDer(G, sigma, 1, 0)
        Gy = gDer(G, sigma, 0, 1)
        Gw = np.sqrt(Gx * Gx + Gy * Gy)

        Bx = gDer(B, sigma, 1, 0)
        By = gDer(B, sigma, 0, 1)
        Bw = np.sqrt(Bx * Bx + By * By)

        return Rw, Gw, Bw

    if order == 2:
        Rxx = gDer(R, sigma, 2, 0)
        Ryy = gDer(R, sigma, 0, 2)
        Rxy = gDer(R, sigma, 1, 1)
        Rw = np.sqrt(Rxx * Rxx + 4.0 * (Rxy * Rxy) + Ryy * Ryy)

        Gxx = gDer(G, sigma, 2, 0)
        Gyy = gDer(G, sigma, 0, 2)
        Gxy = gDer(G, sigma, 1, 1)
        Gw = np.sqrt(Gxx * Gxx + 4.0 * (Gxy * Gxy) + Gyy * Gyy)

        Bxx = gDer(B, sigma, 2, 0)
        Byy = gDer(B, sigma, 0, 2)
        Bxy = gDer(B, sigma, 1, 1)
        Bw = np.sqrt(Bxx * Bxx + 4.0 * (Bxy * Bxy) + Byy * Byy)

        return Rw, Gw, Bw

    raise ValueError("order must be 1 or 2")


def general_cc(input_data: np.ndarray,
               njet: int = 0,
               mink_norm: int = 1,
               sigma: float = 1.0,
               mask_im: Optional[np.ndarray] = None
               ) -> Tuple[float, float, float, np.ndarray]:
    img = np.asarray(input_data).astype(np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("general_cc expects HxWx3 image")

    h, w = img.shape[:2]
    if mask_im is None:
        mask_im = np.zeros((h, w), dtype=np.float64)
    else:
        mask_im = np.asarray(mask_im).astype(np.float64)
        if mask_im.shape != (h, w):
            raise ValueError("mask_im shape must be HxW")

    # remove all saturated points
    saturation_threshold = 255.0
    sat = (np.max(img, axis=2) >= saturation_threshold).astype(np.float64)  # HxW
    mask_im2 = mask_im + dilation33(sat)  # HxW
    mask_im2 = (mask_im2 == 0).astype(np.float64)
    mask_im2 = set_border(mask_im2, int(sigma + 1), 0)

    output_data = img.copy()

    # filtering / derivative selection
    if njet == 0:
        if sigma != 0:
            # gDer(..., sigma, 0, 0): Gaussian smoothing
            for ii in range(3):
                img[:, :, ii] = gDer(img[:, :, ii], sigma, 0, 0)

    if njet > 0:
        Rx, Gx, Bx = NormDerivative(img, sigma, njet)
        img[:, :, 0] = Rx
        img[:, :, 1] = Gx
        img[:, :, 2] = Bx

    img = np.abs(img)

    if mink_norm != -1:
        p = float(mink_norm)
        kleur = np.power(img, p)
        white_R = np.power(np.sum(kleur[:, :, 0] * mask_im2), 1.0 / p)
        white_G = np.power(np.sum(kleur[:, :, 1] * mask_im2), 1.0 / p)
        white_B = np.power(np.sum(kleur[:, :, 2] * mask_im2), 1.0 / p)

        som = math.sqrt(white_R * white_R + white_G * white_G + white_B * white_B)
        if som < 1e-12:
            white_R = white_G = white_B = 1.0 / math.sqrt(3.0)
        else:
            white_R /= som
            white_G /= som
            white_B /= som
    else:
        R = img[:, :, 0] * mask_im2
        G = img[:, :, 1] * mask_im2
        B = img[:, :, 2] * mask_im2
        white_R = float(np.max(R))
        white_G = float(np.max(G))
        white_B = float(np.max(B))
        som = math.sqrt(white_R * white_R + white_G * white_G + white_B * white_B)
        if som < 1e-12:
            white_R = white_G = white_B = 1.0 / math.sqrt(3.0)
        else:
            white_R /= som
            white_G /= som
            white_B /= som

    # apply correction (same as MATLAB)
    scale = math.sqrt(3.0)
    output_data[:, :, 0] = output_data[:, :, 0] / (white_R * scale)
    output_data[:, :, 1] = output_data[:, :, 1] / (white_G * scale)
    output_data[:, :, 2] = output_data[:, :, 2] / (white_B * scale)

    return float(white_R), float(white_G), float(white_B), output_data


# ---------------------------
# MATLAB crop_black block (Otsu + Gaussian + regionprops + heuristic reject)
# ---------------------------
def crop_black_matlab(im_rgb_uint8: np.ndarray,
                      margin: float = 0.1,
                      thresh: float = 0.3) -> Tuple[np.ndarray, bool]:
    """
    Returns (cropped_img, did_crop).
    """
    im = np.asarray(im_rgb_uint8)
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("crop_black_matlab expects HxWx3 uint8 RGB")

    h, w = im.shape[:2]

    # graythresh(rgb2gray(im)) -> Otsu threshold on grayscale in [0,1]
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Otsu via cv2 on 0..255 is easier; convert to 8-bit for exact thresholding
    gray_u8 = (gray * 255.0).astype(np.uint8)
    lvl_u8, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lvl = float(lvl_u8) / 255.0

    # BW = imbinarize(imgaussfilt(gray,2), lvl*0.2)
    gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REPLICATE)
    BW = (gray_blur > (lvl * 0.2)).astype(np.uint8)

    # regionprops: centroid + major/minor axis length
    # We'll label components then measure using OpenCV's moments and fitEllipse-like approximation.
    # For closer behavior to MATLAB's regionprops major/minor axis lengths, we use cv2.fitEllipse on contours when possible.
    num_labels, labels = cv2.connectedComponents(BW, connectivity=8)
    if num_labels <= 1:
        return im, False

    candidates = []
    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if ys.size < 20:
            continue

        # centroid (MATLAB stats.Centroid is [x,y] i.e., [col,row])
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))

        # estimate major/minor axis length via ellipse fit when possible
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        major = minor = None
        if pts.shape[0] >= 5:
            try:
                ell = cv2.fitEllipse(pts.reshape(-1, 1, 2))
                (ex, ey), (MA, ma), angle = ell  # MA>=ma but not guaranteed
                major = float(max(MA, ma))
                minor = float(min(MA, ma))
            except Exception:
                major = minor = None

        if major is None or minor is None:
            # fallback: use bbox as rough proxy
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            major = float(max(x1 - x0 + 1, y1 - y0 + 1))
            minor = float(min(x1 - x0 + 1, y1 - y0 + 1))

        diameter = (major + minor) / 2.0
        candidates.append((diameter, cx, cy))

    if not candidates:
        return im, False

    candidates.sort(key=lambda t: t[0], reverse=True)  # diameter desc

    def compute_box(diameter, cx, cy):
        radius = diameter / 2.0
        # MATLAB uses: x_min = center(2) ... (rows), y_min = center(1) ... (cols)
        x_min = int(cy - radius + margin * radius) + 1  # convert to 1-based like MATLAB int32
        x_max = int(cy + radius - margin * radius) + 1
        y_min = int(cx - radius + margin * radius) + 1
        y_max = int(cx + radius - margin * radius) + 1
        return x_min, x_max, y_min, y_max

    def valid_box(x_min, x_max, y_min, y_max):
        return (x_min >= 1 and y_min >= 1 and x_max <= h and y_max <= w and x_max >= x_min and y_max >= y_min)

    # try largest then second largest
    box = compute_box(*candidates[0])
    if not valid_box(*box) and len(candidates) > 1:
        box = compute_box(*candidates[1])

    if not valid_box(*box):
        return im, False

    x_min, x_max, y_min, y_max = box

    # heuristic reject: mean_outside/mean_inside > thresh
    inside = im[x_min - 1:x_max, y_min - 1:y_max, :].astype(np.float64)
    mean_inside = float(np.mean(inside)) if inside.size else 0.0
    if mean_inside <= 1e-12:
        return im, False

    top = im[0:x_min, :, :].astype(np.float64)                   # 1:x_min
    left = im[x_min - 1:x_max, 0:y_min, :].astype(np.float64)    # x_min:x_max, 1:y_min
    bottom = im[x_max - 1:, :, :].astype(np.float64)             # x_max:end
    right = im[x_min - 1:x_max, y_max - 1:, :].astype(np.float64)# x_min:x_max, y_max:end

    mean_outside = (np.mean(top) + np.mean(left) + np.mean(bottom) + np.mean(right)) / 4.0
    if (mean_outside / mean_inside) > thresh:
        return im, False

    cropped = im[x_min - 1:x_max, y_min - 1:y_max, :]
    return cropped, True


# ---------------------------
# preserve_ratio=True: if H>W, permute; resize so width==preserve_size
# preserve_ratio=False: permute if H>W; resize to std_size (450x600)
# ---------------------------
def resize_matlab(im_rgb: np.ndarray,
                  preserve_ratio: bool = True,
                  preserve_size: int = 600,
                  std_size: Tuple[int, int] = (450, 600)) -> np.ndarray:
    im = np.asarray(im_rgb)
    h, w = im.shape[:2]
    if h > w:
        im = np.transpose(im, (1, 0, 2))
        h, w = im.shape[:2]

    if preserve_ratio:
        if w != preserve_size:
            ratio = preserve_size / float(w)
            new_h = int(round(h * ratio))
            im = cv2.resize(im, (preserve_size, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        target_h, target_w = int(std_size[0]), int(std_size[1])
        if h != target_h or w != target_w:
            im = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    return im


def _iter_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def preprocess_folder(pathImSrc: Path,
                      pathImTar: Path,
                      std_size: Tuple[int, int] = (450, 600),
                      preserve_ratio: bool = True,
                      preserve_size: int = 600,
                      crop_black: bool = True,
                      margin: float = 0.1,
                      thresh: float = 0.3,
                      resize: bool = True,
                      use_cc: bool = True,
                      write_png: bool = False,
                      quality: int = 100,
                      overwrite: bool = False) -> None:
    pathImSrc = Path(pathImSrc)
    pathImTar = Path(pathImTar)
    pathImTar.mkdir(parents=True, exist_ok=True)

    files = _iter_image_files(pathImSrc)
    if not files:
        raise FileNotFoundError(f"No images found in: {pathImSrc}")

    for idx, p in enumerate(files, start=1):
        out_name = p.with_suffix(".png").name if write_png else p.name
        out_path = pathImTar / out_name
        if out_path.exists() and not overwrite:
            if idx % 1000 == 0:
                print(idx)
            continue

        try:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # MATLAB uses RGB
        except Exception:
            print(f"Image {p.name} failed.")
            continue

        # crop_black
        if crop_black:
            im, _ = crop_black_matlab(im, margin=margin, thresh=thresh)

        # resize
        if resize:
            im = resize_matlab(im, preserve_ratio=preserve_ratio, preserve_size=preserve_size, std_size=std_size)

        # color constancy
        if use_cc:
            _, _, _, im_new = general_cc(im, njet=0, mink_norm=6, sigma=0)
            im_new = np.clip(im_new, 0, 255).astype(np.uint8)
        else:
            im_new = im.astype(np.uint8)

        # write
        out_bgr = cv2.cvtColor(im_new, cv2.COLOR_RGB2BGR)
        if write_png:
            cv2.imwrite(str(out_path), out_bgr)
        else:
            cv2.imwrite(str(out_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

        if idx % 1000 == 0:
            print(idx)


def main():
    parser = argparse.ArgumentParser(description="ISIC2019 official preprocessing (MATLAB-equivalent).")

    # ---- default: in-place preprocess the already-prepared official folder ----
    # Expected folder (relative to project root):
    #   classify/data/dataDir/images/official
    # We auto-detect the project root by searching upwards from this script.
    script_dir = Path(__file__).resolve().parent
    project_root = None
    for p in [script_dir, *script_dir.parents]:
        if (p / "data" / "dataDir" / "images" / "official").exists():
            project_root = p
            break
    if project_root is None:
        project_root = script_dir

    default_official = project_root / "data" / "dataDir" / "images" / "official"

    # By default, read from official and overwrite back into official (src == dst).
    parser.add_argument("--src", type=str, default=str(default_official),
                        help="Input folder (pathImSrc). Default: classify/data/dataDir/images/official")
    parser.add_argument("--dst", type=str, default=str(default_official),
                        help="Output folder (pathImTar). Default: same as --src (in-place overwrite)")

    # Default overwrite=True; you can disable with --no_overwrite if you ever want to keep originals.
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True,
                        help="Overwrite existing outputs (default: True)")
    parser.add_argument("--no_overwrite", dest="overwrite", action="store_false",
                        help="Do not overwrite existing outputs")

    parser.add_argument("--preserve_ratio", action="store_true", default=True)
    parser.add_argument("--no_preserve_ratio", action="store_false", dest="preserve_ratio")
    parser.add_argument("--preserve_size", type=int, default=600)
    parser.add_argument("--std_h", type=int, default=450)
    parser.add_argument("--std_w", type=int, default=600)

    parser.add_argument("--crop_black", action="store_true", default=True)
    parser.add_argument("--no_crop_black", action="store_false", dest="crop_black")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--thresh", type=float, default=0.3)

    parser.add_argument("--resize", action="store_true", default=True)
    parser.add_argument("--no_resize", action="store_false", dest="resize")

    parser.add_argument("--use_cc", action="store_true", default=True)
    parser.add_argument("--no_cc", action="store_false", dest="use_cc")

    parser.add_argument("--write_png", action="store_true", default=False)
    parser.add_argument("--quality", type=int, default=100)

    args = parser.parse_args()

    preprocess_folder(
        pathImSrc=Path(args.src),
        pathImTar=Path(args.dst),
        std_size=(args.std_h, args.std_w),
        preserve_ratio=args.preserve_ratio,
        preserve_size=args.preserve_size,
        crop_black=args.crop_black,
        margin=args.margin,
        thresh=args.thresh,
        resize=args.resize,
        use_cc=args.use_cc,
        write_png=args.write_png,
        quality=args.quality,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
