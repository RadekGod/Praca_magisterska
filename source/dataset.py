"""Datasety i narzędzia do ładowania obrazów i masek.

Ten moduł zawiera:
- funkcje pomocnicze do ładowania tiff-ów: `load_multiband`, `load_grayscale`,
  `get_crs`, `save_img`.
- funkcję `_pick_augm` wybierającą tryb augmentacji z modułu `transforms`.
- klasę `Dataset` (rozszerzenie torch.utils.data.Dataset) oraz wyspecjalizowane
  dataset'y: `SARDataset`, `RGBDataset`, `FusionDataset`.

Konwencje i ważne uwagi:
- Maski (labels) na wejściu w większości operacji są w formacie integer, tzn.
  mapa etykiet z wartościami [0..C-1]. Transformacje i oversampling (class-aware crop)
  działają na tej formie PRZED konwersją do one-hot.
- One-hot w tym kontekście oznacza tensor o kształcie [B, C, H, W] (lub [C, H, W]
  dla pojedynczego przykładu), gdzie dla każdej klasy jest oddzielny kanał z wartościami 0/1.
  Większość strat (lossów) oczekuje target w formie one-hot lub konwertuje go przez argmax
  przy użyciu CrossEntropy.
- Opis działania "class-aware crop" (oversampling):
  Działa na masce w formacie integer (przed one-hot). Jeśli żądanej klasy nie ma
  w oknie lub nie uda się znaleźć odpowiedniego wycinka w `max_tries`, wykonany
  zostanie zwykły losowy crop.

Obsługiwane typy obrazów:
- SAR: jednoskanalowy (H, W) — używane przez `SARDataset`.
- RGB: wielokanałowy (H, W, 3) — używane przez `RGBDataset`.
- Fuzja SAR+RGB: kanały sklejone wzdłuż osi kanałów (H, W, 4) — używane przez `FusionDataset`.

"""

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
    """Zwraca funkcję augmentacji z modułu `transforms` na podstawie numeru trybu.

    Parametry:
    - train (bool): czy dataset jest w trybie treningowym.
    - train_augm / valid_augm: numer trybu augmentacji (1,2,3) lub None (domyślnie 1).

    Zwraca:
    - callable: funkcję augmentacji (np. T.train_augm1 / T.valid_augm2).
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
        """Inicjalizacja datasetu.

        Parametry wejściowe:
        - label_list: lista ścieżek do plików z etykietami (ścieżka powinna zawierać
          segment "labels" aby funkcje loadera mogły z niej wywnioskować ścieżki do
          obrazów: "sar_images" i/lub "rgb_images").
        - classes: lista klas (przekazywana do ToTensor, opcjonalnie używana przy one-hot).
        - size: rozmiar cropu używany podczas treningu (domyślnie 128).
        - train: czy dataset jest w trybie treningowym (wpływa na augmentacje i oversampling).
        - sar_mean, sar_std: statystyki SAR (jeżeli compute_stats=True można je obliczyć automatycznie).
        - compute_stats: jeżeli True i train=True, dataset policzy globalne średnie/std dla SAR
          używając `T.compute_sar_stats` i funkcji `load_grayscale`.
        - sar_normalize: tryb normalizacji SAR (np. 'global' lub 'per_image').
        - train_augm / valid_augm: numer trybu augmentacji (1..3) dla treningu / walidacji.

        Class-aware crop / oversampling (parametry):
        - class_aware_crop (bool): włącza próbę cropów skoncentrowanych na danej klasie.
        - oversample_class (int): indeks klasy docelowej, której próbujemy "oversamplować".
        - oversample_p (float): prawdopodobieństwo, że dany sample będzie poddany class-aware crop.
        - oversample_min_pixels (int): minimalna liczba pikseli klasy w oknie, żeby przyjąć crop.
        - oversample_max_tries (int): maksymalna liczba prób znalezienia okna spełniającego warunek.

        Uwaga:
        - Konfiguracja cropów (`self.crop_cfg`) działa na masce w formacie integer (przed one-hot).
          Jeśli żądanej klasy nie ma w oknie lub nie uda się znaleźć odpowiedniego wycinka
          w `max_tries`, wykonywany jest zwykły losowy crop.

        Metoda tworzy pomocnicze obiekty:
        - self.augm: wybrana funkcja augmentacji z modułu `transforms`.
        - self.to_tensor: obiekt `T.ToTensor` z informacjami o normalizacji SAR i listą klas.

        Przykładowe kształty/formaty:
        - obraz SAR ładowany przez `load_grayscale`: (H, W) -> po `ToTensor`: [1, H, W]
        - obraz RGB ładowany przez `load_multiband`: (H, W, 3) -> po `ToTensor`: [3, H, W]
        - maska ładowana jako (H, W) z wartościami int (0..C-1); `ToTensor` może zamienić
          ją na one-hot [C, H, W] jeśli to wymagane przez dalsze przetwarzanie.
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
    """Dataset dla obrazów SAR (1 kanał)."""
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
    """Dataset dla obrazów RGB (3 kanały)."""
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
    """Dataset dla fuzji SAR+RGB; zwraca obraz z kanałami [R,G,B,SAR] (4 kanały)."""
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