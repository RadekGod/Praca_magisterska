import argparse
import os

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

from collections import OrderedDict

from source.dataset import load_multiband, load_grayscale, get_crs, save_img
from source import transforms as T


# Mapa kolorów: indeks klasy -> (R, G, B)
# 0 traktujemy jako tło (czarne), kolejne klasy jako widoczne kolory.
CLASS_COLORS = {
    0: (0, 0, 0),        # tło
    1: (255, 0, 0),      # klasa 1 - czerwony
    2: (0, 255, 0),      # klasa 2 - zielony
    3: (0, 0, 255),      # klasa 3 - niebieski
    4: (255, 255, 0),    # klasa 4 - żółty
    5: (255, 0, 255),    # klasa 5 - magenta
    6: (0, 255, 255),    # klasa 6 - cyjan
    7: (255, 128, 0),    # klasa 7 - pomarańczowy
    8: (128, 0, 255),    # klasa 8 - fioletowy
}


# ImageNet stats (muszą być identyczne jak w source/transforms.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(model_type: str, num_classes: int, device: str = "cuda"):
    """Tworzy model U-Net zgodny z tymi z train_*.

    model_type: 'rgb', 'sar' lub 'fusion'.
    """
    if model_type == "rgb":
        in_channels = 3
        encoder_weights = None  # wagi i tak wczytamy z pliku .pth
    elif model_type == "sar":
        in_channels = 1
        encoder_weights = None
    elif model_type == "fusion":
        in_channels = 4
        encoder_weights = None
    else:
        raise ValueError(f"Nieznany model_type: {model_type}")

    model = smp.Unet(
        classes=num_classes,
        in_channels=in_channels,
        activation=None,
        encoder_weights=encoder_weights,
        encoder_name=encoder_name,
        decoder_attention_type="scse",
    )
    model.to(device)
    model.eval()
    return model


def logits_to_mask_rgb(logits: torch.Tensor) -> np.ndarray:
    """Konwertuje wyjście modelu (B, C, H, W) na kolorową maskę (3, H, W) w uint8.

    Zakładamy batch=1.
    """
    # (B, C, H, W) -> (C, H, W)
    if logits.dim() == 4:
        logits = logits[0]
    # predykcja klas pikseli
    pred = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)  # (H, W)

    h, w = pred.shape
    rgb_mask = np.zeros((3, h, w), dtype=np.uint8)

    for cls_idx, color in CLASS_COLORS.items():
        r, g, b = color
        mask = pred == cls_idx
        rgb_mask[0][mask] = r
        rgb_mask[1][mask] = g
        rgb_mask[2][mask] = b

    return rgb_mask


def _preprocess_image(
    img: np.ndarray,
    model_type: str,
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
) -> torch.Tensor:
    """Preprocessing zgodny z treningiem.

    - RGB: /255 + normalizacja ImageNet mean/std
    - SAR: /255 + (opcjonalnie) normalizacja SAR jak w `source.transforms.ToTensor`

    Zwraca tensor (1, C, H, W) float32.
    """
    if img.ndim == 2:
        img = img[..., None]

    x = img.astype(np.float32) / 255.0  # H,W,C

    if model_type in ("rgb", "fusion") and x.shape[-1] >= 3:
        # normalizujemy pierwsze 3 kanały jako RGB
        x[..., 0] = (x[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        x[..., 1] = (x[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        x[..., 2] = (x[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    # Normalizacja SAR (dla modelu SAR i dla kanału SAR w fusion)
    if model_type in ("sar", "fusion"):
        # zakładamy: SAR jest jedynym kanałem (sar) albo ostatnim kanałem (fusion)
        sar_ch = -1 if x.shape[-1] > 1 else 0
        if sar_normalize == "global" and sar_mean is not None and sar_std is not None:
            std = sar_std if sar_std > 0 else 1.0
            x[..., sar_ch] = (x[..., sar_ch] - float(sar_mean)) / float(std)
        elif sar_normalize == "per_sample":
            m = float(x[..., sar_ch].mean())
            s = float(x[..., sar_ch].std())
            s = s if s > 0 else 1.0
            x[..., sar_ch] = (x[..., sar_ch] - m) / s
        # 'none' -> bez zmian

    # (H,W,C) -> (1,C,H,W)
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    return x


def _print_pred_distribution(pred: np.ndarray, num_classes: int):
    """Wypisuje procent pikseli każdej klasy w predykcji (diagnostyka szybkiego 'collapse')."""
    if pred.size == 0:
        return
    counts = np.bincount(pred.reshape(-1), minlength=num_classes).astype(np.int64)
    total = int(counts.sum())
    parts = []
    for i in range(num_classes):
        pct = 100.0 * float(counts[i]) / float(total)
        if pct >= 0.1:  # nie spamuj bardzo małymi
            parts.append(f"c{i}:{pct:.1f}%")
    print("Pred class distribution:", ", ".join(parts) if parts else "(all <0.1%)")


def run_inference(
    model,
    model_path: str,
    image_path: str,
    output_path: str,
    device: str = None,
    model_type: str = "rgb",
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
):
    """Wykonuje inferencję dla pojedynczego obrazu i zapisuje kolorową maskę."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- wczytanie obrazu ---
    if model_type == "rgb":
        img = load_multiband(image_path)          # (H, W, 3)
    elif model_type == "sar":
        sar = load_grayscale(image_path)          # (H, W)
        img = sar[..., None]                      # (H, W, 1)
    elif model_type == "fusion":
        img = load_multiband(image_path)          # (H, W, 4)
    else:
        raise ValueError(f"Nieznany model_type: {model_type}")

    # --- preprocessing identyczny jak w treningu ---
    x = _preprocess_image(
        img,
        model_type=model_type,
        sar_normalize=sar_normalize,
        sar_mean=sar_mean,
        sar_std=sar_std,
    ).to(device)  # (1, C, H, W)

    # --- inferencja ---
    with torch.no_grad():
        logits = model(x)  # (1, C, H, W)

    # --- diagnostyka: rozkład klas ---
    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype(np.uint8)
    _print_pred_distribution(pred, num_classes=len(CLASS_COLORS))

    # --- konwersja na kolorową maskę ---
    rgb_mask = logits_to_mask_rgb(logits)  # (3, H, W)

    # --- zapis GeoTIFF z georeferencją z oryginału, jeśli jest ---
    try:
        crs, transform = get_crs(image_path)
    except Exception:
        crs, transform = None, None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if crs is not None and transform is not None:
        save_img(output_path, rgb_mask, crs, transform)
    else:
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


def main():
    parser = argparse.ArgumentParser(description="Inferencja segmentacji i zapis kolorowej maski")
    parser.add_argument("--image_path", type=str,
                        help="Ścieżka do obrazu wejściowego (RGB lub SAR, GeoTIFF). Jeśli jest to folder,"
                             " przetworzone zostaną wszystkie pliki .tif w tym folderze.",
                        default="results/obrazy/sar/oryginalne")
    parser.add_argument("--cpu", action="store_true", help="Wymuś użycie CPU zamiast GPU", default=False)
    parser.add_argument("--model_type", type=str, default="sar", choices=["rgb", "sar", "fusion"],
                        help="Typ modelu do użycia: 'rgb' (train_rgb.py), 'sar' (train_sar.py) lub 'fusion'.")
    parser.add_argument("--model_path", type=str,
                        help="Ścieżka do wytrenowanego modelu .pth, np. model/RGB_Unet_resnet34_WCEPlusDice_ce1.0_dice1.0_lr_0.001_augmT2V1.pth",
                        default="model/sar/" + main_model_name + model_name_variant + ".pth")
    parser.add_argument("--output_path", type=str,
                        help="Ścieżka wyjściowa.\n"
                             "Jeśli --image_path jest plikiem, to jest to dokładna ścieżka do wyjścia.\n"
                             "Jeśli --image_path jest folderem, to jest to folder wyjściowy,"
                             " w którym zostaną zapisane maski o tych samych nazwach plików.",
                        default = "results/obrazy/sar/model/" + encoder_name + "/" + model_name_variant)
    parser.add_argument("--sar_normalize", type=str, default="global", choices=["global", "per_sample", "none"],
                        help="Normalizacja SAR jak w treningu: global (mean/std z datasetu), per_sample, none")
    parser.add_argument("--sar_mean", type=float, default=None, help="Mean SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument("--sar_std", type=float, default=None, help="Std SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument("--sar_stats_root", type=str, default="../dataset/train",
                        help="Jeśli podasz folder treningowy (np. ../dataset/train), to mean/std SAR zostaną policzone automatycznie z labeli.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Ładuję wagi modelu z: {args.model_path}")

    # --- wylicz mean/std SAR jeśli trzeba ---
    sar_mean = args.sar_mean
    sar_std = args.sar_std
    if args.model_type in ("sar", "fusion") and args.sar_normalize == "global" and (sar_mean is None or sar_std is None):
        if args.sar_stats_root is not None:
            from pathlib import Path
            label_paths = [str(f) for f in Path(args.sar_stats_root).rglob("*.tif") if "labels" in f.parts]
            m, s = T.compute_sar_stats(label_paths, load_fn=None)
            sar_mean = m if sar_mean is None else sar_mean
            sar_std = s if sar_std is None else sar_std
            print(f"SAR stats (computed): mean={sar_mean} std={sar_std} (scale [0,1])")
        else:
            print("[WARN] SAR global normalization wybrana, ale nie podano --sar_mean/--sar_std ani --sar_stats_root. Inferencja będzie bez normalizacji SAR.")

    # --- zbudowanie modelu i załadowanie wag tylko raz ---
    num_classes = len(CLASS_COLORS)
    model = build_model(model_type=args.model_type, num_classes=num_classes, device=device)

    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    # Nie ładujmy wag "po cichu" – wypiszemy co nie pasuje.
    incompatible = model.load_state_dict(new_state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print("[WARN] Missing keys when loading state_dict (model weights not found in checkpoint):")
        for k in missing[:50]:
            print("  -", k)
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    if unexpected:
        print("[WARN] Unexpected keys when loading state_dict (checkpoint contains weights not used by model):")
        for k in unexpected[:50]:
            print("  -", k)
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more")

    # --- rozróżnienie: pojedynczy plik vs folder ---
    if os.path.isdir(args.image_path):
        in_dir = args.image_path
        out_dir = args.output_path
        os.makedirs(out_dir, exist_ok=True)

        # przetworzenie wszystkich plików .tif w folderze (bez rekurencji)
        for fn in sorted(os.listdir(in_dir)):
            if not fn.lower().endswith(".tif"):
                continue
            in_path = os.path.join(in_dir, fn)
            out_path = os.path.join(out_dir, fn)
            print(f"Przetwarzam: {in_path} -> {out_path}")
            run_inference(
                model=model,
                model_path=args.model_path,
                image_path=in_path,
                output_path=out_path,
                device=device,
                model_type=args.model_type,
                sar_normalize=args.sar_normalize,
                sar_mean=sar_mean,
                sar_std=sar_std,
            )
    else:
        run_inference(
            model=model,
            model_path=args.model_path,
            image_path=args.image_path,
            output_path=args.output_path,
            device=device,
            model_type=args.model_type,
            sar_normalize=args.sar_normalize,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )


if __name__ == "__main__":
    model_name_variant = "T3V1"
    encoder_name = "tu-convnext_tiny"
    main_model_name = "SAR_Unet_" + encoder_name +"_WCEPlusDice_ce1.0_dice1.0_lr_0.001_augm"
    main()
