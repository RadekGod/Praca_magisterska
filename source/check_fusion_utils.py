import os
from typing import Tuple

import numpy as np
import rasterio
import torch

from source.dataset import load_multiband, load_grayscale, get_crs, save_img


# ImageNet stats (muszą być identyczne jak w source/transforms.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def logits_to_mask_rgb(logits: torch.Tensor, class_colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """(B,C,H,W) -> (3,H,W) uint8 z mapą kolorów. Zakładamy batch=1."""
    if logits.dim() == 4:
        logits = logits[0]
    pred = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)

    h, w = pred.shape
    rgb_mask = np.zeros((3, h, w), dtype=np.uint8)
    for cls_idx, color in class_colors.items():
        r, g, b = color
        mask = pred == cls_idx
        rgb_mask[0][mask] = r
        rgb_mask[1][mask] = g
        rgb_mask[2][mask] = b
    return rgb_mask


def print_pred_distribution(pred: np.ndarray, num_classes: int):
    """Procent pikseli każdej klasy (szybka diagnostyka collapse do tła)."""
    if pred.size == 0:
        return
    counts = np.bincount(pred.reshape(-1), minlength=num_classes).astype(np.int64)
    total = int(counts.sum())
    parts = []
    for i in range(num_classes):
        pct = 100.0 * float(counts[i]) / float(total)
        if pct >= 0.1:
            parts.append(f"c{i}:{pct:.1f}%")
    print("Pred class distribution:", ", ".join(parts) if parts else "(all <0.1%)")


def load_rgb_sar_pair_from_rgb_path(rgb_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Loader dopasowany do folderu `results/obrazy/fusion/oryginalne`.

    W tym folderze obrazy są w parach:
      - *_RGB.tif  (3 kanały)
      - *_SAR.tif  (1 kanał)

    Wejściem jest zawsze ścieżka do *_RGB.tif.

    Zwraca: (rgb [H,W,3], sar [H,W], georef_source_path)
    gdzie georef_source_path to plik RGB (dla CRS/transform).
    """
    if not rgb_path.lower().endswith("_rgb.tif"):
        raise ValueError(f"Obsługiwany jest tylko format *_RGB.tif. Dostałem: {rgb_path}")

    sar_path = rgb_path[:-8] + "_SAR.tif"  # zamiana sufiksu _RGB.tif -> _SAR.tif
    if not os.path.exists(sar_path):
        raise FileNotFoundError(f"Brak pasującego pliku SAR: {sar_path} (dla RGB: {rgb_path})")

    rgb = load_multiband(rgb_path)
    sar = load_grayscale(sar_path)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB file is not (H,W,3): {rgb_path} -> {getattr(rgb, 'shape', None)}")
    if sar.ndim != 2:
        raise ValueError(f"SAR file is not (H,W): {sar_path} -> {getattr(sar, 'shape', None)}")
    if rgb.shape[:2] != sar.shape[:2]:
        raise ValueError(f"RGB/SAR shape mismatch: rgb={rgb.shape} sar={sar.shape} for {rgb_path}")

    return rgb, sar, rgb_path


def preprocess_rgb_sar_to_4ch_tensor(
    rgb: np.ndarray,
    sar: np.ndarray,
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
) -> torch.Tensor:
    """Preprocessing zgodny z transforms.ToTensor dla fuzji.

    - RGB: /255 + ImageNet mean/std
    - SAR: /255 + global/per_sample/none

    Zwraca tensor (1,4,H,W) float32 w kolejności: [R,G,B,SAR].
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    sar_f = sar.astype(np.float32) / 255.0

    # RGB
    rgb_f[..., 0] = (rgb_f[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    rgb_f[..., 1] = (rgb_f[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    rgb_f[..., 2] = (rgb_f[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    # SAR
    if sar_normalize == "global" and sar_mean is not None and sar_std is not None:
        std = float(sar_std) if float(sar_std) > 0 else 1.0
        sar_f = (sar_f - float(sar_mean)) / std
    elif sar_normalize == "per_sample":
        m = float(sar_f.mean())
        s = float(sar_f.std())
        s = s if s > 0 else 1.0
        sar_f = (sar_f - m) / s

    x = np.dstack([rgb_f, sar_f[..., None]])
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()


def save_mask_geotiff(output_path: str, rgb_mask: np.ndarray, georef_path: str):
    """Zapisuje maskę (3,H,W) jako GeoTIFF z georeferencją z georef_path jeśli dostępna."""
    try:
        crs, transform = get_crs(georef_path)
    except Exception:
        crs, transform = None, None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if crs is not None and transform is not None:
        save_img(output_path, rgb_mask, crs, transform)
        return

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=rgb_mask.shape[1],
        width=rgb_mask.shape[2],
        count=3,
        dtype=rgb_mask.dtype,
    ) as dst:
        dst.write(rgb_mask)

