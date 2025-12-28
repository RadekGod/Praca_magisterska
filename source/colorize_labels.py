import argparse
import os
from pathlib import Path

import numpy as np
import rasterio

from source.check_model_rgb import CLASS_COLORS
from source.dataset import get_crs, save_img


def labels_to_color_mask(labels: np.ndarray) -> np.ndarray:
    """Konwertuje maskę etykiet (H, W) lub (1, H, W) na kolorową maskę (3, H, W).

    Używa mapowania klas z CLASS_COLORS.
    """
    # Upewniamy się, że mamy kształt (H, W)
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    elif labels.ndim != 2:
        raise ValueError(f"Oczekiwano maski o wymiarach (H, W) lub (1, H, W), dostałem {labels.shape}")

    labels = labels.astype(np.int64)
    h, w = labels.shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)

    for cls_idx, (r, g, b) in CLASS_COLORS.items():
        mask = labels == cls_idx
        if not np.any(mask):
            continue
        rgb[0][mask] = r
        rgb[1][mask] = g
        rgb[2][mask] = b

    return rgb


def process_file(input_path: Path, output_path: Path) -> None:
    """Wczytuje pojedynczy plik .tif z etykietami i zapisuje kolorową wersję."""
    with rasterio.open(input_path) as src:
        data = src.read()  # (bands, H, W) – oczekujemy 1 kanału z etykietami

    # Jeśli jest więcej kanałów, bierzemy pierwszy jako etykiety
    if data.shape[0] > 1:
        labels = data[0]
    else:
        labels = data[0]

    color_mask = labels_to_color_mask(labels)

    # Georeferencja (opcjonalnie, próbujemy pobrać z pliku)
    try:
        crs, transform = get_crs(str(input_path))
    except Exception:
        crs, transform = None, None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if crs is not None and transform is not None:
        save_img(str(output_path), color_mask, crs, transform)
    else:
        with rasterio.open(
            str(output_path),
            "w",
            driver="GTiff",
            height=color_mask.shape[1],
            width=color_mask.shape[2],
            count=3,
            dtype=color_mask.dtype,
        ) as dst:
            dst.write(color_mask)


def process_directory(input_dir: str, output_dir: str, suffix: str = "_color") -> None:
    """Przetwarza wszystkie pliki .tif w podanym katalogu.

    - input_dir: katalog z plikami etykiet (.tif), np. dataset/train/labels
    - output_dir: katalog wyjściowy; struktura nazw plików zostanie zachowana
    - suffix: dodawany przed rozszerzeniem, np. label.tif -> label_color.tif
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Katalog wejściowy nie istnieje lub nie jest katalogiem: {in_dir}")

    tif_files = sorted(in_dir.glob("*.tif"))
    if not tif_files:
        print(f"Brak plików .tif w katalogu: {in_dir}")
        return

    print(f"Znaleziono {len(tif_files)} plików .tif do przetworzenia w {in_dir}")

    for idx, tif_path in enumerate(tif_files, start=1):
        rel = tif_path.name
        stem = tif_path.stem
        out_name = f"{stem}{suffix}.tif"
        out_path = out_dir / out_name

        print(f"[{idx}/{len(tif_files)}] Przetwarzam {rel} -> {out_path}")
        process_file(tif_path, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Konwersja masek etykiet .tif na kolorowe maski wg CLASS_COLORS z check_model_rgb.py"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/obrazy/etykiety/",
        help="Ścieżka do katalogu z plikami etykiet .tif (np. dataset/train/labels)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="results/obrazy/rgb/etykiety/",
        help=(
            "Katalog wyjściowy na kolorowe maski. Jeżeli nie podano, "
            "zostanie utworzony podkatalog 'color' w input_dir."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_color",
        help="Przyrostek dodawany do nazwy pliku przed rozszerzeniem (domyślnie '_color').",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "color")

    print(f"Katalog wejściowy: {input_dir}")
    print(f"Katalog wyjściowy: {output_dir}")
    print(f"Przyrostek nazw plików: {args.suffix}")

    process_directory(input_dir, output_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()

