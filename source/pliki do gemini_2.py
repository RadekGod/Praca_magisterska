import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice loss dla segmentacji wieloklasowej.

    Oczekuje:
    - logits: [B, C, H, W]
    - target: one-hot [B, C, H, W] (tak jak u Ciebie po ToTensor)

    include_background=False zwykle pomaga, gdy background dominuje.
    """

    def __init__(self, smooth: float = 1.0, include_background: bool = False):
        super().__init__()
        self.smooth = float(smooth)
        self.include_background = bool(include_background)
        self.name = "DiceLoss"

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        if target.ndim != 4:
            raise ValueError(f"DiceLoss expects target one-hot [B,C,H,W], got {tuple(target.shape)}")

        if not self.include_background and probs.size(1) > 1:
            probs = probs[:, 1:]
            target = target[:, 1:]

        dims = (0, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        denom = (probs + target).sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class WeightedCEPlusDice(nn.Module):
    """Kombinacja: weighted CrossEntropy + Dice.

    - WCE: liczone po indeksach klas (target.argmax(1))
    - Dice: liczone na softmax i one-hot
    """

    def __init__(self, class_weights, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 dice_smooth: float = 1.0, dice_include_background: bool = False, device: str = "cuda"):
        super().__init__()
        w = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.dice = DiceLoss(smooth=dice_smooth, include_background=dice_include_background)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.name = f"WCEPlusDice_ce{self.ce_weight}_dice{self.dice_weight}"

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_idx = target.argmax(dim=1)
        ce = self.ce(logits, y_idx)
        dice = self.dice(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice


class CEWithLogitsLoss(nn.Module):
    def __init__(self, weights, device="cuda"):
        super().__init__()
        self.weight = torch.from_numpy(weights).float().to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.name = "CELoss"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        return loss



import numpy as np
import rasterio
from torch.utils.data import Dataset as BaseDataset
from typing import Optional

from . import transforms as T


def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


def get_crs(path):
    src = rasterio.open(path, "r")
    return src.crs, src.transform


def save_img(path, img, crs, transform):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=img.shape[1],
        width=img.shape[2],
        count=img.shape[0],
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(img)
        dst.close()


def _pick_augm(train: bool, train_augm: Optional[int], valid_augm: Optional[int]):
    """Zwraca funkcję augmentacji z modułu transforms na podstawie numeru trybu.

    train_augm/valid_augm: 1,2,3 lub None (None => domyślnie 1).
    """
    if train:
        idx = train_augm or 1
        if idx == 1:
            return T.train_augm1
        if idx == 2:
            return T.train_augm2
        if idx == 3:
            return T.train_augm3
        # fallback
        return T.train_augm1
    else:
        idx = valid_augm or 1
        if idx == 1:
            return T.valid_augm1
        if idx == 2:
            return T.valid_augm2
        if idx == 3:
            return T.valid_augm3
        return T.valid_augm1


class Dataset(BaseDataset):
    def __init__(
        self,
        label_list,
        classes=None,
        size=128,
        train=False,
        sar_mean=None,
        sar_std=None,
        compute_stats=False,
        sar_normalize='global',
        train_augm: Optional[int] = None,
        valid_augm: Optional[int] = None,
        # --- class-aware crop / oversampling ---
        class_aware_crop: bool = False,
        oversample_class: int = 1,
        oversample_p: float = 0.0,
        oversample_min_pixels: int = 20,
        oversample_max_tries: int = 30,
    ):
        """label_list: list of label file paths (contains 'labels' in path)
        If compute_stats=True and train=True, dataset will compute global SAR mean/std
        using T.compute_sar_stats and the module-level load_grayscale.

        train_augm / valid_augm - numery trybów augmentacji (1..3) dla treningu / walidacji.
        """
        self.fns = label_list
        self.augm = _pick_augm(train=train, train_augm=train_augm, valid_augm=valid_augm)
        self.size = size
        self.train = train

        # konfiguracja cropów (działa na masce int, przed one-hot)
        self.crop_cfg = {
            "enabled": bool(class_aware_crop) and bool(train) and float(oversample_p) > 0.0,
            "target_class": int(oversample_class),
            "p": float(oversample_p),
            "min_pixels": int(oversample_min_pixels),
            "max_tries": int(oversample_max_tries),
        }

        # store stats for diagnostics
        self.sar_mean = sar_mean
        self.sar_std = sar_std
        self.sar_normalize = sar_normalize

        # create ToTensor with SAR normalization info
        self.to_tensor = T.ToTensor(
            classes=classes,
            sar_mean=sar_mean,
            sar_std=sar_std,
            sar_normalize=sar_normalize,
        )
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_grayscale(self.fns[idx].replace("labels", "sar_images"))
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size, self.crop_cfg)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)

# =============================================
#   Dataset dla obrazów SAR (1 kanał)
# =============================================
class SARDataset(Dataset):
    def __getitem__(self, idx):
        img = self.load_grayscale(self.fns[idx].replace("labels", "sar_images"))
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size, self.crop_cfg)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

# =============================================
#   Dataset dla obrazów RGB (3 kanały)
# =============================================
class RGBDataset(Dataset):
    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "rgb_images"))
        msk = self.load_grayscale(self.fns[idx])
        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size, self.crop_cfg)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)
        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

# =============================================
#   Dataset dla fuzji SAR+RGB (4 kanały: R,G,B,SAR)
# =============================================
class FusionDataset(Dataset):
    def __getitem__(self, idx):
        # Ścieżki obrazów bazując na etykiecie
        rgb_path = self.fns[idx].replace("labels", "rgb_images")
        sar_path = self.fns[idx].replace("labels", "sar_images")
        # Ładowanie RGB (H,W,3) i SAR (H,W)
        rgb = self.load_multiband(rgb_path)
        sar = self.load_grayscale(sar_path)
        # Połączenie kanałów -> (H,W,4)
        img = np.dstack([rgb, sar[..., None]])
        msk = self.load_grayscale(self.fns[idx])
        # Augmentacje zgodne z SAR/RGB (bez kolorowych specyficznych operacji)
        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size, self.crop_cfg)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)  # image -> [4,H,W]
        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    from pathlib import Path
    import random
    from torch.utils.data import DataLoader
    from . import transforms as T

    # =============================================
    #   Build Data Loaders
    # Funkcja zwraca dwa DataLoadery: train oraz validation.
    # =============================================
    def build_data_loaders(args, DatasetClass):
        """Zwraca train_loader i valid_loader dla podanej klasy DatasetClass.
        DatasetClass powinien akceptować (fns, classes=..., size=..., train=..., sar_mean=..., sar_std=..., sar_normalize=..., train_augm=..., valid_augm=...)
        """

        # =============================================
        #   Scan i split dataset
        # =============================================
        # 1) Znajdź wszystkie pliki .tif w katalogu args.data_root, które znajdują się w katalogu "labels".
        # 2) Pomieszaj listę (random.shuffle) i podziel w proporcji 90%/10% na zbiór treningowy i walidacyjny.
        image_paths = [f for f in Path(args.data_root).rglob("*.tif") if "labels" in f.parts]
        random.shuffle(image_paths)
        split_index = int(0.9 * len(image_paths))
        train_paths = image_paths[:split_index]
        validate_paths = image_paths[split_index:]
        train_paths = [str(f) for f in train_paths]
        validate_paths = [str(f) for f in validate_paths]

        print("Total samples      :", len(image_paths))
        print("Training samples   :", len(train_paths))
        print("Validation samples :", len(validate_paths))

        # =============================================
        #   Obliczanie uśrednionych statystyk SAR
        # =============================================

        sar_mean, sar_std = (None, None)
        sar_normalize = getattr(args, 'sar_normalize', 'global')

        needs_sar = DatasetClass.__name__ in ("SARDataset", "FusionDataset")

        if needs_sar and sar_normalize == 'global':
            try:
                sar_mean, sar_std = T.compute_sar_stats(train_paths, load_fn=None)
            except Exception as e:
                print("Warning: failed to compute SAR stats in build_data_loaders:", e)

        # Tryby augmentacji przekazywane z args; mogą nie istnieć w starszych skryptach, więc używamy getattr.
        train_augm = getattr(args, 'train_augm')
        valid_augm = getattr(args, 'valid_augm')

        # --- class-aware crop / oversampling (tylko train) ---
        class_aware_crop = bool(getattr(args, 'class_aware_crop', 0))
        oversample_class = int(getattr(args, 'oversample_class', 1))
        oversample_p = float(getattr(args, 'oversample_p', 0.0))
        oversample_min_pixels = int(getattr(args, 'oversample_min_pixels', 20))
        oversample_max_tries = int(getattr(args, 'oversample_max_tries', 30))

        # =============================================
        #   Tworzenie datasetów i loaderów
        # =============================================
        trainset = DatasetClass(
            train_paths,
            classes=args.classes,
            size=getattr(args, 'crop_size', None),
            train=True,
            sar_mean=sar_mean,
            sar_std=sar_std,
            sar_normalize=sar_normalize,
            train_augm=train_augm,
            valid_augm=valid_augm,
            class_aware_crop=class_aware_crop,
            oversample_class=oversample_class,
            oversample_p=oversample_p,
            oversample_min_pixels=oversample_min_pixels,
            oversample_max_tries=oversample_max_tries,
        )
        validset = DatasetClass(
            validate_paths,
            classes=args.classes,
            train=False,
            sar_mean=sar_mean,
            sar_std=sar_std,
            sar_normalize=sar_normalize,
            train_augm=train_augm,
            valid_augm=valid_augm,
            class_aware_crop=False,
            oversample_class=oversample_class,
            oversample_p=0.0,
            oversample_min_pixels=oversample_min_pixels,
            oversample_max_tries=oversample_max_tries,
        )

        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(int(args.num_workers), 2),
            pin_memory=False,
        )
        valid_loader = DataLoader(
            validset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        return train_loader, valid_loader



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



import argparse
import os

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

from collections import OrderedDict

from source.dataset import load_multiband, load_grayscale, get_crs, save_img


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


def _preprocess_image(img: np.ndarray, model_type: str) -> torch.Tensor:
    """Preprocessing zgodny z treningiem.

    - RGB: /255 + normalizacja ImageNet mean/std
    - SAR/fusion: tylko /255 (tu brak globalnych statystyk w check_model_rgb.py)

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

    # (H,W,C) -> (1,C,H,W)
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    return x


def run_inference(model, model_path: str, image_path: str, output_path: str, device: str = None, model_type: str = "rgb"):
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
    x = _preprocess_image(img, model_type=model_type).to(device)  # (1, C, H, W)

    # --- inferencja ---
    with torch.no_grad():
        logits = model(x)  # (1, C, H, W)

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
                        default="results/obrazy/rgb/oryginalne")
    parser.add_argument("--cpu", action="store_true", help="Wymuś użycie CPU zamiast GPU", default=False)
    parser.add_argument("--model_type", type=str, default="rgb", choices=["rgb", "sar", "fusion"],
                        help="Typ modelu do użycia: 'rgb' (train_rgb.py), 'sar' (train_sar.py) lub 'fusion'.")
    parser.add_argument("--model_path", type=str,
                        help="Ścieżka do wytrenowanego modelu .pth, np. model/RGB_Unet_resnet34_WCEPlusDice_ce1.0_dice1.0_lr_0.001_augmT1V1.pth",
                        default="model/rgb/" + main_model_name + model_name_variant + ".pth")
    parser.add_argument("--output_path", type=str,
                        help="Ścieżka wyjściowa.\n"
                             "Jeśli --image_path jest plikiem, to jest to dokładna ścieżka do wyjścia.\n"
                             "Jeśli --image_path jest folderem, to jest to folder wyjściowy,"
                             " w którym zostaną zapisane maski o tych samych nazwach plików.",
                        default="results/obrazy/rgb/model/" + encoder_name + "/" + model_name_variant)
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Ładuję wagi modelu z: {args.model_path}")

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
            )
    else:
        # tryb pojedynczego pliku jak wcześniej
        run_inference(
            model=model,
            model_path=args.model_path,
            image_path=args.image_path,
            output_path=args.output_path,
            device=device,
            model_type=args.model_type,
        )


if __name__ == "__main__":
    model_name_variant = "T1V1"
    encoder_name = "efficientnet-b4"
    main_model_name = "RGB_Unet_" + encoder_name +"_WCEPlusDice_ce1.0_dice1.0_lr_0.001_augm"
    main()





import argparse
import os
from collections import OrderedDict

import torch

from source import transforms as T
from source.train_late_fusion import LateFusionUnet
from source.check_fusion_utils import (
    load_rgb_sar_pair_from_rgb_path,
    preprocess_rgb_sar_to_4ch_tensor,
    logits_to_mask_rgb,
    print_pred_distribution,
    save_mask_geotiff,
)


# Mapa kolorów: indeks klasy -> (R, G, B)
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


def build_model(num_classes: int, device: str = "cuda", fusion_mode: str = "mean"):
    """Buduje model LateFusion zgodny z train_late_fusion.py (fuzja na logitach)."""
    model = LateFusionUnet(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights_rgb=None,   # wagi i tak wczytamy z checkpointu .pth
        encoder_weights_sar=None,
        decoder_attention_type="scse",
        fusion_mode=fusion_mode,
    )
    model.to(device)
    model.eval()
    return model


def run_inference(
    model,
    rgb_path: str,
    output_path: str,
    device: str,
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
):
    rgb, sar, georef_path = load_rgb_sar_pair_from_rgb_path(rgb_path)

    x = preprocess_rgb_sar_to_4ch_tensor(
        rgb,
        sar,
        sar_normalize=sar_normalize,
        sar_mean=sar_mean,
        sar_std=sar_std,
    ).to(device)

    with torch.no_grad():
        logits = model(x)  # (1,C,H,W)

    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype('uint8')
    print_pred_distribution(pred, num_classes=len(CLASS_COLORS))

    rgb_mask = logits_to_mask_rgb(logits, class_colors=CLASS_COLORS)
    save_mask_geotiff(output_path, rgb_mask, georef_path)


def main():
    parser = argparse.ArgumentParser(
        description="check_model_late_fusion: LateFusion (logit fusion) inferencja + zapis kolorowej maski"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="results/obrazy/fusion/oryginalne",
        help=(
            "Ścieżka do FOLDERU z parami plików: *_RGB.tif oraz *_SAR.tif. "
            "Tryb pojedynczego pliku nie jest obsługiwany."
        ),
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Wymuś CPU zamiast GPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/fusion/" + main_model_name + model_name_variant + ".pth",
        help="Ścieżka do checkpointu late-fusion (.pth) zapisanego przez train_late_fusion.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/obrazy/fusion/model/" + encoder_name + "/" + model_name_variant + "/" + model_fusion_variant,
        help="Folder wyjściowy na maski .tif",
    )
    parser.add_argument("--fusion_mode", type=str, default="mean", choices=["mean", "sum", "weighted"], help="Tryb fuzji logitów")

    parser.add_argument(
        "--sar_normalize",
        type=str,
        default="global",
        choices=["global", "per_sample", "none"],
        help="Normalizacja SAR: global (mean/std z datasetu), per_sample, none",
    )
    parser.add_argument("--sar_mean", type=float, default=None, help="Mean SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument("--sar_std", type=float, default=None, help="Std SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument(
        "--sar_stats_root",
        type=str,
        default="../dataset/train",
        help="Folder treningowy (np. ../dataset/train), z którego policzymy mean/std SAR jeśli nie podasz ręcznie.",
    )

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Ładuję wagi modelu z: {args.model_path}")

    # SAR mean/std (jeśli global)
    sar_mean = args.sar_mean
    sar_std = args.sar_std
    if args.sar_normalize == "global" and (sar_mean is None or sar_std is None):
        try:
            from pathlib import Path

            label_paths = [str(f) for f in Path(args.sar_stats_root).rglob("*.tif") if "labels" in f.parts]
            m, s = T.compute_sar_stats(label_paths, load_fn=None)
            sar_mean = m if sar_mean is None else sar_mean
            sar_std = s if sar_std is None else sar_std
            print(f"SAR stats (computed): mean={sar_mean} std={sar_std} (scale [0,1])")
        except Exception as e:
            print("[WARN] Nie udało się policzyć SAR mean/std, inferencja może być bez normalizacji SAR.")
            print("       Błąd:", e)

    # build + load
    num_classes = len(CLASS_COLORS)
    model = build_model(num_classes=num_classes, device=device, fusion_mode=args.fusion_mode)

    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print("[WARN] Missing keys when loading state_dict:")
        for k in missing[:50]:
            print("  -", k)
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    if unexpected:
        print("[WARN] Unexpected keys when loading state_dict:")
        for k in unexpected[:50]:
            print("  -", k)
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more")

    # Wejściem musi być folder
    if not os.path.isdir(args.image_path):
        raise ValueError(f"Wymagany jest folder z parami *_RGB.tif + *_SAR.tif. Podano: {args.image_path}")

    in_dir = args.image_path
    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    # Przetwarzamy tylko *_RGB.tif
    rgb_files = [fn for fn in sorted(os.listdir(in_dir)) if fn.lower().endswith("_rgb.tif")]
    if not rgb_files:
        raise FileNotFoundError(f"Nie znaleziono plików *_RGB.tif w folderze: {in_dir}")

    # walidacja par
    missing_sar = []
    for fn in rgb_files:
        sar_fn = fn[:-8] + "_SAR.tif"
        if not os.path.exists(os.path.join(in_dir, sar_fn)):
            missing_sar.append(sar_fn)
    if missing_sar:
        raise FileNotFoundError(
            "Brakuje pasujących plików *_SAR.tif dla części *_RGB.tif. Brakujące (pierwsze 20): "
            + ", ".join(missing_sar[:20])
        )

    for fn in rgb_files:
        in_path = os.path.join(in_dir, fn)
        out_path = os.path.join(out_dir, fn.replace("_RGB.tif", ".tif"))
        print(f"Przetwarzam: {in_path} -> {out_path}")
        run_inference(
            model=model,
            rgb_path=in_path,
            output_path=out_path,
            device=device,
            sar_normalize=args.sar_normalize,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )


if __name__ == "__main__":
    model_name_variant = "T1V1"
    model_fusion_variant = "LATE_MEAN"
    encoder_name = "efficientnet-b4"
    main_model_name = "LATE_FUSION_Unet_" + encoder_name + "_WCEPlusDice_ce1.0_dice1.0_lr_0.001_fusion_mean_augm"
    main()



import argparse
import os
from collections import OrderedDict

import torch

# --- Umożliwia uruchomienie skryptu jako pliku: python path\to\check_model_joint_fusion.py
# (wtedy katalog projektu może nie być na PYTHONPATH).
try:
    from source import transforms as T
    from source.train_joint_fusion import JointFusionUnet
    from source.check_fusion_utils import (
        load_rgb_sar_pair_from_rgb_path,
        preprocess_rgb_sar_to_4ch_tensor,
        logits_to_mask_rgb,
        print_pred_distribution,
        save_mask_geotiff,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from source import transforms as T
    from source.train_joint_fusion import JointFusionUnet
    from source.check_fusion_utils import (
        load_rgb_sar_pair_from_rgb_path,
        preprocess_rgb_sar_to_4ch_tensor,
        logits_to_mask_rgb,
        print_pred_distribution,
        save_mask_geotiff,
    )


# Mapa kolorów: indeks klasy -> (R, G, B)
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


def build_model(
    num_classes: int,
    device: str = "cuda",
    feature_fusion: str = "concat",
):
    """Buduje model JointFusion zgodny z train_joint_fusion.py."""
    model = JointFusionUnet(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights_rgb=None,  # wagi i tak wczytamy z checkpointu .pth
        encoder_weights_sar=None,
        decoder_attention_type="scse",
        feature_fusion=feature_fusion,
        encoder_depth=5,
        safe_nan_to_num=True,
    )
    model.to(device)
    model.eval()
    return model


def run_inference(
    model,
    rgb_path: str,
    output_path: str,
    device: str,
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
):
    """Inferencja dla pojedynczej pary RGB+SAR (wejście: ścieżka do *_RGB.tif)."""
    rgb, sar, georef_path = load_rgb_sar_pair_from_rgb_path(rgb_path)

    x = preprocess_rgb_sar_to_4ch_tensor(
        rgb,
        sar,
        sar_normalize=sar_normalize,
        sar_mean=sar_mean,
        sar_std=sar_std,
    ).to(device)

    with torch.no_grad():
        logits = model(x)  # (1,C,H,W)

    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype("uint8")
    print_pred_distribution(pred, num_classes=len(CLASS_COLORS))

    rgb_mask = logits_to_mask_rgb(logits, class_colors=CLASS_COLORS)
    save_mask_geotiff(output_path, rgb_mask, georef_path)


def main():
    parser = argparse.ArgumentParser(
        description="check_model_joint_fusion: Joint/Intermediate Fusion (feature-level) inferencja + zapis kolorowej maski"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="results/obrazy/fusion/oryginalne",
        help=(
            "Ścieżka do FOLDERU z parami plików: *_RGB.tif oraz *_SAR.tif. "
            "Tryb pojedynczego pliku nie jest obsługiwany."
        ),
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Wymuś CPU zamiast GPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/fusion/" + main_model_name + model_name_variant + ".pth",
        help="Ścieżka do checkpointu joint-fusion (.pth) zapisanego przez train_joint_fusion.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/obrazy/fusion/model/" + encoder_name + "/" + model_name_variant + "/" + model_fusion_variant,
        help="Folder wyjściowy na maski .tif",
    )
    parser.add_argument(
        "--feature_fusion",
        type=str,
        default="concat",
        choices=["concat", "sum", "mean"],
        help="Jak łączyć feature maps RGB i SAR na każdym poziomie (musi pasować do treningu)",
    )

    parser.add_argument(
        "--sar_normalize",
        type=str,
        default="global",
        choices=["global", "per_sample", "none"],
        help="Normalizacja SAR: global (mean/std z datasetu), per_sample, none",
    )
    parser.add_argument("--sar_mean", type=float, default=None, help="Mean SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument("--sar_std", type=float, default=None, help="Std SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument(
        "--sar_stats_root",
        type=str,
        default="../dataset/train",
        help="Folder treningowy (np. ../dataset/train), z którego policzymy mean/std SAR jeśli nie podasz ręcznie.",
    )

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Ładuję wagi modelu z: {args.model_path}")

    # SAR mean/std (jeśli global)
    sar_mean = args.sar_mean
    sar_std = args.sar_std
    if args.sar_normalize == "global" and (sar_mean is None or sar_std is None):
        try:
            from pathlib import Path

            label_paths = [str(f) for f in Path(args.sar_stats_root).rglob("*.tif") if "labels" in f.parts]
            m, s = T.compute_sar_stats(label_paths, load_fn=None)
            sar_mean = m if sar_mean is None else sar_mean
            sar_std = s if sar_std is None else sar_std
            print(f"SAR stats (computed): mean={sar_mean} std={sar_std} (scale [0,1])")
        except Exception as e:
            print("[WARN] Nie udało się policzyć SAR mean/std, inferencja może być bez normalizacji SAR.")
            print("       Błąd:", e)

    # build + load
    num_classes = len(CLASS_COLORS)
    model = build_model(num_classes=num_classes, device=device, feature_fusion=args.feature_fusion)

    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print("[WARN] Missing keys when loading state_dict:")
        for k in missing[:50]:
            print("  -", k)
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    if unexpected:
        print("[WARN] Unexpected keys when loading state_dict:")
        for k in unexpected[:50]:
            print("  -", k)
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more")

    # Wejściem musi być folder
    if not os.path.isdir(args.image_path):
        raise ValueError(f"Wymagany jest folder z parami *_RGB.tif + *_SAR.tif. Podano: {args.image_path}")

    in_dir = args.image_path
    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    # Przetwarzamy tylko *_RGB.tif
    rgb_files = [fn for fn in sorted(os.listdir(in_dir)) if fn.lower().endswith("_rgb.tif")]
    if not rgb_files:
        raise FileNotFoundError(f"Nie znaleziono plików *_RGB.tif w folderze: {in_dir}")

    # walidacja par
    missing_sar = []
    for fn in rgb_files:
        sar_fn = fn[:-8] + "_SAR.tif"
        if not os.path.exists(os.path.join(in_dir, sar_fn)):
            missing_sar.append(sar_fn)
    if missing_sar:
        raise FileNotFoundError(
            "Brakuje pasujących plików *_SAR.tif dla części *_RGB.tif. Brakujące (pierwsze 20): "
            + ", ".join(missing_sar[:20])
        )

    for fn in rgb_files:
        in_path = os.path.join(in_dir, fn)
        out_path = os.path.join(out_dir, fn.replace("_RGB.tif", ".tif"))
        print(f"Przetwarzam: {in_path} -> {out_path}")
        run_inference(
            model=model,
            rgb_path=in_path,
            output_path=out_path,
            device=device,
            sar_normalize=args.sar_normalize,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )


if __name__ == "__main__":
    model_name_variant = "T1V1"
    model_fusion_variant = "JOINT_CONCAT"
    encoder_name = "efficientnet-b4"
    main_model_name = "JOINT_FUSION_Unet_" + encoder_name + "_WCEPlusDice_ce1.0_dice1.0_lr_0.001_feat_concat_augm"
    main()



import argparse
import os
from collections import OrderedDict

import torch
import segmentation_models_pytorch as smp

from source import transforms as T
from source.check_fusion_utils import (
    load_rgb_sar_pair_from_rgb_path,
    preprocess_rgb_sar_to_4ch_tensor,
    logits_to_mask_rgb,
    print_pred_distribution,
    save_mask_geotiff,
)


# Mapa kolorów: indeks klasy -> (R, G, B)
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


def build_model(num_classes: int, device: str = "cuda"):
    """Buduje model early-fusion (RGB+SAR = 4 kanały) zgodny z train_early_fusion.py."""
    model = smp.Unet(
        classes=num_classes,
        in_channels=4,
        activation=None,
        encoder_weights=None,  # wagi i tak wczytamy z .pth
        encoder_name=encoder_name,
        decoder_attention_type="scse",
    )
    model.to(device)
    model.eval()
    return model


def _print_first_conv_diagnostics(model: torch.nn.Module):
    """Wypisuje diagnostykę pierwszej konwolucji (czy faktycznie 4 kanały, statystyki wag)."""
    target = model.module if hasattr(model, "module") else model
    for name, p in target.named_parameters():
        if p.dim() == 4 and any(k in name.lower() for k in ["conv_stem", "conv1", "stem", "conv_first"]):
            w = p.detach().cpu()
            print(f"First conv param: {name} shape: {tuple(w.shape)}")
            print(f"  mean: {float(w.mean().item()):.6f} std: {float(w.std().item()):.6f}")
            ww = w.numpy()
            if ww.ndim == 4:
                ch_means = ww.mean(axis=(0, 2, 3))
                print("  Per-channel means:", ", ".join(f"{v:.6f}" for v in ch_means))
            return
    print("[WARN] Nie znaleziono parametru pierwszej konwolucji do diagnostyki.")


def run_inference(
    model,
    image_path: str,
    output_path: str,
    device: str,
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
):
    """Inferencja dla pojedynczego obrazu fusion (para RGB+SAR) i zapis kolorowej maski."""
    rgb, sar, georef_path = load_rgb_sar_pair_from_rgb_path(image_path)

    x = preprocess_rgb_sar_to_4ch_tensor(
        rgb,
        sar,
        sar_normalize=sar_normalize,
        sar_mean=sar_mean,
        sar_std=sar_std,
    ).to(device)

    with torch.no_grad():
        logits = model(x)

    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype('uint8')
    print_pred_distribution(pred, num_classes=len(CLASS_COLORS))

    rgb_mask = logits_to_mask_rgb(logits, class_colors=CLASS_COLORS)
    save_mask_geotiff(output_path, rgb_mask, georef_path)


def main():
    parser = argparse.ArgumentParser(description="check_model_fusion: inferencja + zapis kolorowej maski (RGB+SAR)")
    parser.add_argument(
        "--image_path",
        type=str,
        default="results/obrazy/fusion/oryginalne",
        help=(
            "Ścieżka do FOLDERU z parami plików: *_RGB.tif oraz *_SAR.tif (np. TrainArea_3902_RGB.tif i TrainArea_3902_SAR.tif). "
            "Tryb pojedynczego pliku nie jest obsługiwany."
        ),
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Wymuś CPU zamiast GPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/fusion/" + main_model_name + model_name_variant + ".pth",
        help="Ścieżka do modelu .pth",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/obrazy/fusion/model/" + encoder_name + "/" + model_name_variant,
        help="Ścieżka wyjściowa (folder).",
    )

    parser.add_argument(
        "--sar_normalize",
        type=str,
        default="global",
        choices=["global", "per_sample", "none"],
        help="Normalizacja SAR: global (mean/std z datasetu), per_sample, none",
    )
    parser.add_argument("--sar_mean", type=float, default=None, help="Mean SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument("--sar_std", type=float, default=None, help="Std SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument(
        "--sar_stats_root",
        type=str,
        default="../dataset/train",
        help="Folder treningowy (np. ../dataset/train), z którego policzymy mean/std SAR jeśli nie podasz ręcznie.",
    )

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Ładuję wagi modelu z: {args.model_path}")

    # SAR mean/std (jeśli global)
    sar_mean = args.sar_mean
    sar_std = args.sar_std
    if args.sar_normalize == "global" and (sar_mean is None or sar_std is None):
        try:
            from pathlib import Path

            label_paths = [str(f) for f in Path(args.sar_stats_root).rglob("*.tif") if "labels" in f.parts]
            m, s = T.compute_sar_stats(label_paths, load_fn=None)
            sar_mean = m if sar_mean is None else sar_mean
            sar_std = s if sar_std is None else sar_std
            print(f"SAR stats (computed): mean={sar_mean} std={sar_std} (scale [0,1])")
        except Exception as e:
            print("[WARN] Nie udało się policzyć SAR mean/std, inferencja może być bez normalizacji SAR.")
            print("       Błąd:", e)

    # build + load
    num_classes = len(CLASS_COLORS)
    model = build_model(num_classes=num_classes, device=device)

    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print("[WARN] Missing keys when loading state_dict:")
        for k in missing[:50]:
            print("  -", k)
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    if unexpected:
        print("[WARN] Unexpected keys when loading state_dict:")
        for k in unexpected[:50]:
            print("  -", k)
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more")

    _print_first_conv_diagnostics(model)
    if sar_mean is not None and sar_std is not None:
        print(f"SAR mean/std used: mean={float(sar_mean):.6f} std={float(sar_std):.6f} (scale [0,1])")

    # Wejściem musi być folder
    if not os.path.isdir(args.image_path):
        raise ValueError(
            "Wymagany jest folder z parami *_RGB.tif + *_SAR.tif. "
            f"Podano: {args.image_path}"
        )

    in_dir = args.image_path
    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    rgb_files = [fn for fn in sorted(os.listdir(in_dir)) if fn.lower().endswith("_rgb.tif")]
    if not rgb_files:
        raise FileNotFoundError(f"Nie znaleziono plików *_RGB.tif w folderze: {in_dir}")

    missing_sar = []
    for fn in rgb_files:
        sar_fn = fn[:-8] + "_SAR.tif"
        if not os.path.exists(os.path.join(in_dir, sar_fn)):
            missing_sar.append(sar_fn)
    if missing_sar:
        raise FileNotFoundError(
            "Brakuje pasujących plików *_SAR.tif dla części *_RGB.tif. Brakujące (pierwsze 20): "
            + ", ".join(missing_sar[:20])
        )

    for fn in rgb_files:
        in_path = os.path.join(in_dir, fn)
        out_path = os.path.join(out_dir, fn.replace("_RGB.tif", ".tif"))
        print(f"Przetwarzam: {in_path} -> {out_path}")
        run_inference(
            model=model,
            image_path=in_path,
            output_path=out_path,
            device=device,
            sar_normalize=args.sar_normalize,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )


if __name__ == "__main__":
    model_name_variant = "T1V1"
    encoder_name = "efficientnet-b4"
    main_model_name = "FUSION_Unet_" + encoder_name + "_WCEPlusDice_ce1.0_dice1.0_lr_0.001_augm"
    main()



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




import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza rozkładu klas w maskach etykiet. "
            "Przechodzi po wszystkich plikach .tif w podanym katalogu, "
            "zlicza wartości pikseli (klasy) i wyświetla ich liczność oraz udział procentowy."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="../dataset/train/labels",
        help=(
            "Ścieżka do katalogu z maskami etykiet. "
            "Domyślnie ../dataset/train/labels względem katalogu source."
        ),
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".tif",
        help="Rozszerzenie plików etykiet (np. .tif). Domyślnie .tif.",
    )
    return parser.parse_args()


def find_label_files(labels_dir: Path, ext: str) -> List[Path]:
    """Zwraca posortowaną listę plików etykiet o zadanym rozszerzeniu.

    Nie przeszukujemy rekurencyjnie ani nie filtrujemy po plikach ukrytych,
    bo scenariusz jest prosty: pojedynczy katalog z ~4000 plików .tif.
    """

    if not labels_dir.exists() or not labels_dir.is_dir():
        print(f"[BŁĄD] Katalog z etykietami nie istnieje lub nie jest katalogiem: {labels_dir}")
        return []

    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext

    files = [
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ext
    ]
    files.sort()
    return files


def load_mask_pil(path: Path) -> np.ndarray:
    """Wczytuje maskę jako obraz 2D (H, W) z zachowaniem wartości pikseli.

    Używamy PIL i konwertujemy do trybu "L" (8-bit szarości), jeśli to konieczne.
    To w zupełności wystarczy, jeśli klasy są zakodowane jako wartości 0..255.
    """

    with Image.open(path) as img:
        if img.mode not in ("L", "I"):
            img = img.convert("L")
        mask = np.array(img)
    # Upewniamy się, że mamy 2D (bez kanału koloru)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def update_class_counts(counts: Dict[int, int], mask: np.ndarray) -> None:
    """Aktualizuje słownik `counts` o zliczenia wartości pikseli w `mask`.

    Używa np.unique, więc jest bardzo szybkie i pamięciooszczędne
    w porównaniu z iterowaniem po każdym pikselu w Pythonie.
    """

    values, pix_counts = np.unique(mask, return_counts=True)
    for v, c in zip(values, pix_counts):
        v_int = int(v)
        counts[v_int] = counts.get(v_int, 0) + int(c)


def main() -> None:
    args = parse_args()

    labels_dir = Path(args.labels_dir).resolve()
    print(f"Katalog z etykietami: {labels_dir}")
    files = find_label_files(labels_dir, args.ext)

    if not files:
        print("[INFO] Nie znaleziono żadnych plików etykiet o zadanym rozszerzeniu.")
        return

    print(f"Liczba znalezionych plików: {len(files)}")

    class_counts: Dict[int, int] = {}
    num_errors = 0

    for idx, path in enumerate(files, start=1):
        try:
            mask = load_mask_pil(path)
        except Exception as e:  # noqa: BLE001
            num_errors += 1
            print(f"[OSTRZEŻENIE] Problem z wczytaniem pliku {path.name}: {e}")
            continue

        update_class_counts(class_counts, mask)

        if idx % 500 == 0:
            print(f"Przetworzono {idx} / {len(files)} plików...")

    if not class_counts:
        print("[INFO] Nie udało się zliczyć żadnych pikseli (być może wszystkie pliki były błędne).")
        return

    total_pixels = sum(class_counts.values())
    if total_pixels == 0:
        print("[INFO] Łączna liczba pikseli wynosi 0.")
        return

    print("\n=== PODSUMOWANIE KLAS ===")
    print(f"Łączna liczba plików: {len(files)}")
    print(f"Liczba plików z błędami: {num_errors}")
    print(f"Łączna liczba pikseli (wszystkie klasy): {total_pixels}")
    print()

    print(f"{'Klasa':>10} | {'Piksele':>15} | {'Udział [%]':>10}")
    print("-" * 45)

    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_pixels) * 100.0
        print(f"{class_id:>10d} | {count:>15d} | {percentage:>9.4f}")

    print("-" * 45)
    print(f"{'SUMA':>10} | {total_pixels:>15d} | {100.0:>9.4f}")


if __name__ == "__main__":
    main()
