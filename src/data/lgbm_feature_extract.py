"""
Feature extraction utilities for the LGBM pipeline.

This module provides functions to compute and preprocess features for the LGBM pipeline:
  - resize_image: resize images or masks to a consistent size
  - apply_mask: apply binary mask to an image
  - average_down_array: downsample arrays by averaging
  - get_center: detect circular region centers via Hough transform
  - generate_circle_mask: create a mask from center and radius
  - generate_masks: batch-generate masks for image directory
  - extract_color_histograms: per-image RGB histograms
  - extract_lbp_features: Local Binary Pattern histograms
  - extract_shape_descriptors: region-based shape metrics
  - extract_and_save_features: run all channels and save to .npz

Example:
    from data.feature_extraction import generate_masks, extract_and_save_features

    # 1) generate masks
    mask_paths = generate_masks(
        image_dir="data/images",
        mask_dir="data/masks",
    )

    # 2) extract and save features
    extract_and_save_features(
        image_dir="data/images",
        mask_dir="data/masks",
        labels_csv="data/labels.csv",
        output_path="data/features.npz",
        image_size=(256,256)
    )
"""
import os
from typing import List, Tuple
import numpy as np
import cv2
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label

# Parameters for LBP
LBP_POINTS = 24
LBP_RADIUS = 3
# Histogram bins
COLOR_BINS = 256


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image or mask to the specified size using linear interpolation.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an image. Background set to zero.
    """
    # Ensure mask single-channel and binary
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(img, img, mask=bin_mask)


def average_down_array(arr: np.ndarray, target_length: int = 10) -> np.ndarray:
    """
    Downsample a 1D or 2D array along axis 0 by averaging into target_length segments.
    """
    n = arr.shape[0]
    if target_length >= n:
        return arr
    chunk = n // target_length
    extra = n % target_length
    result = []
    start = 0
    for i in range(target_length):
        end = start + chunk + (1 if extra > 0 else 0)
        if extra > 0:
            extra -= 1
        segment = arr[start:end]
        result.append(np.mean(segment, axis=0))
        start = end
    return np.stack(result)


def get_center(image_path: str) -> Tuple[int, int, int]:
    """
    Detect the center (x, y) and radius of the main circular region via Hough transform.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=12,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=1475,
        maxRadius=1525
    )
    if circles is None:
        raise RuntimeError(f"No circles detected in {image_path}")
    x, y, r = circles[0][0]
    return int(x), int(y), int(r)


def generate_circle_mask(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    radius: int
) -> np.ndarray:
    """
    Generate a binary circular mask of given shape at center with radius.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def generate_masks(
    image_dir: str,
    mask_dir: str
) -> List[str]:
    """
    Generate circular masks for all images in image_dir and save to mask_dir.

    Returns list of mask file paths.
    """
    os.makedirs(mask_dir, exist_ok=True)
    masks = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue
        ipath = os.path.join(image_dir, fname)
        x, y, r = get_center(ipath)
        img = cv2.imread(ipath)
        mask = generate_circle_mask(img.shape[:2], (x, y), r)
        mname = os.path.splitext(fname)[0] + '_mask.png'
        mpath = os.path.join(mask_dir, mname)
        cv2.imwrite(mpath, mask)
        masks.append(mpath)
    return masks


def extract_color_histograms(
    image_paths: List[str],
    bins: int = COLOR_BINS,
    image_size: Tuple[int, int] = None,
    mask_paths: List[str] = None
) -> np.ndarray:
    """Compute color histograms, with optional resize and masking."""
    hist_list = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        if image_size:
            img = resize_image(img, image_size)
        if mask_paths and mask_paths[i]:
            mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            if image_size:
                mask = resize_image(mask, image_size)
            img = apply_mask(img, mask)
        hists = []
        for ch in range(3):
            hist = cv2.calcHist([img], [ch], None, [bins], [0, 256])
            hists.append(hist.flatten())
        hist_list.append(np.concatenate(hists))
    return np.vstack(hist_list)


def extract_lbp_features(
    image_paths: List[str],
    P: int = LBP_POINTS,
    R: int = LBP_RADIUS,
    image_size: Tuple[int, int] = None,
    mask_paths: List[str] = None
) -> np.ndarray:
    """Compute LBP histograms, with optional resize and masking."""
    lbp_list = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        if image_size:
            img = resize_image(img, image_size)
        if mask_paths and mask_paths[i]:
            mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            if image_size:
                mask = resize_image(mask, image_size)
            img = apply_mask(img, mask)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2), density=True)
        lbp_list.append(hist.astype(float))
    return np.vstack(lbp_list)


def extract_shape_descriptors(
    mask_paths: List[str],
    image_size: Tuple[int, int] = None
) -> np.ndarray:
    """Compute shape descriptors (area, perimeter, eccentricity) from masks."""
    shape_list = []
    for path in mask_paths:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        if image_size:
            m = resize_image(m, image_size)
        _, bw = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        lbl = label(bw)
        props = regionprops(lbl)
        if props:
            rp = props[0]
            shape_list.append([rp.area, rp.perimeter, rp.eccentricity])
        else:
            shape_list.append([0.0, 0.0, 0.0])
    return np.vstack(shape_list)


def extract_and_save_features(
    image_dir: str,
    mask_dir: str,
    labels_csv: str,
    output_path: str,
    image_size: Tuple[int, int] = None
) -> None:
    """Extract features for the LGBM pipeline and save to .npz."""
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ])
    mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in image_paths]
    df = pd.read_csv(labels_csv).set_index('file_name')
    y = [df.loc[os.path.basename(p), 'label'] for p in image_paths]
    color_feats = extract_color_histograms(
        image_paths, bins=COLOR_BINS, image_size=image_size, mask_paths=mask_paths)
    lbp_feats = extract_lbp_features(
        image_paths, image_size=image_size, mask_paths=mask_paths)
    shape_feats = extract_shape_descriptors(mask_paths, image_size=image_size)
    np.savez_compressed(
        output_path,
        color=color_feats,
        lbp=lbp_feats,
        shape=shape_feats,
        y=np.array(y)
    )
    print(f"Saved features to {output_path}")
