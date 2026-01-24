"""Moduł `transforms`.

Zawiera narzędzia do przetwarzania obrazów i masek przed trenowaniem i ewaluacją.

Główne elementy:
- ToTensor: konwertuje próbkę (słownik z kluczami 'image' i 'mask') na tensory PyTorch
  oraz normalizuje kanały. Obsługuje obrazy RGB, SAR (jednokanałowe) oraz kombinacje
  (np. RGB + dodatkowy kanał SAR).

- compute_sar_stats(paths, ...): oblicza globalne mean/std dla kanału SAR pośród podanych plików.

- class_aware_random_crop / _random_crop_np: funkcje do losowego cropowania, z opcją
  preferowania regionów zawierających daną klasę maski.

- train_augm1/2/3 i valid_augm1/2/3: predefiniowanie pipeline'y augmentacji o różnym stopniu
  agresywności (od minimalnego do mocnego). Zwracają obiekt z kluczami 'image' i 'mask'
  zgodny ze zwracanym formatem przez albumentations.

Format wejściowy `sample`:
    sample: dict z kluczami:
        - 'image': numpy.ndarray, kształt HxW lub HxWXC, wartości [0..255] lub float
        - 'mask': numpy.ndarray, kształt HxW, typ integer (klasy)

Zwracane formaty:
    Większość funkcji zwraca słownik {'image': image, 'mask': mask} (tak jak albumentations).

Uwaga:
    Ten plik skupia się na kompatybilności pipeline'u z danymi satelitarnymi (RGB + SAR)
    i zawiera konwencje normalizacji: ImageNet dla RGB, opcjonalnie globalne lub per-sample
    dla SAR.
"""

import warnings
import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF
import torch
import rasterio

warnings.simplefilter("ignore")

# ImageNet stats (float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ToTensor:
    """Konwerter próbki do tensorów PyTorch i normalizator kanałów.

    Przeznaczenie:
        Przygotowuje dane wejściowe modelu: konwertuje maski do tensorów (opcjonalnie one-hot z
        backgroundem na pierwszym kanale), konwertuje obrazy do postaci CxHxW oraz normalizuje
        kanały:
          - RGB: standard ImageNet (mean/std zdefiniowane w module)
          - SAR: opcje normalizacji: 'global' (używa podanych sar_mean/sar_std), 'per_sample' (liczy
                 mean/std na próbce), lub 'none'

    Argumenty konstruktora (przekazywane do __init__):
        classes: opcjonalna lista wartości klas (np. [1,2,3]) używana do konwersji maski na one-hot
        sar_mean, sar_std: opcjonalne wartości float do globalnej normalizacji SAR (skala [0,1])
        sar_normalize: 'global' | 'per_sample' | 'none'

    Format wejściowy:
        sample: dict z 'image' (H,W lub H,W,C) i 'mask' (H,W). Wartości obrazów w skali 0..255

    Zwracane wartości:
        Zmodyfikowany sample (dict) z polami:
          - 'image': torch.FloatTensor CxHxW
          - 'mask': torch.FloatTensor (jednokanałowa float lub one-hot CxHxW jeśli classes podane)
    """

    def __init__(self, classes=None, sar_mean=None, sar_std=None, sar_normalize='global'):
        """Inicjalizuje konwerter."""
        self.classes = classes
        self.sar_mean = float(sar_mean) if sar_mean is not None else None
        self.sar_std = float(sar_std) if sar_std is not None else None
        self.sar_normalize = sar_normalize

    def __call__(self, sample):
        """Wykonuje konwersję i normalizację na pojedynczej próbce.

        Argumenty:
            sample (dict): wejściowa próbka z kluczami 'image' (ndarray) i 'mask' (ndarray)

        Zwraca:
            dict: próbka z 'image' i 'mask' przekonwertowanymi na tensory PyTorch.
        """
        # mask -> one-hot with background first if classes provided
        if self.classes:
            msks = [(sample["mask"] == v) for v in self.classes]
            msk = np.stack(msks, axis=-1).astype(np.float32)
            background = 1 - msk.sum(axis=-1, keepdims=True)
            sample["mask"] = TF.to_tensor(np.concatenate((background, msk), axis=-1))
        else:
            sample["mask"] = TF.to_tensor(sample["mask"].astype(np.float32))

        img = np.asarray(sample["image"]).astype(np.float32)

        # single-channel SAR
        if img.ndim == 2:
            img = img / 255.0
            if self.sar_normalize == 'global' and self.sar_mean is not None and self.sar_std is not None:
                img = (img - self.sar_mean) / self.sar_std
            elif self.sar_normalize == 'per_sample':
                m = img.mean()
                s = img.std()
                s = s if s > 0 else 1.0
                img = (img - m) / s
            img = img[np.newaxis, ...]
            sample["image"] = torch.from_numpy(img.astype(np.float32))
            return sample

        # multi-channel H,W,C (RGB or RGB+SAR)
        if img.ndim == 3:
            img = img / 255.0
            img = img.transpose(2, 0, 1)  # C,H,W
            C = img.shape[0]

            # normalize RGB channels if present
            if C >= 3:
                for c in range(3):
                    img[c] = (img[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
                # if additional channels exist, assume last is SAR
                if C > 3:
                    if self.sar_normalize == 'global' and self.sar_mean is not None and self.sar_std is not None:
                        img[-1] = (img[-1] - self.sar_mean) / self.sar_std
                    elif self.sar_normalize == 'per_sample':
                        m = img[-1].mean()
                        s = img[-1].std()
                        s = s if s > 0 else 1.0
                        img[-1] = (img[-1] - m) / s
            else:
                # fewer than 3 channels: treat all channels like SAR if requested
                if self.sar_normalize == 'global' and self.sar_mean is not None and self.sar_std is not None:
                    for c in range(C):
                        img[c] = (img[c] - self.sar_mean) / self.sar_std
                elif self.sar_normalize == 'per_sample':
                    for c in range(C):
                        m = img[c].mean()
                        s = img[c].std()
                        s = s if s > 0 else 1.0
                        img[c] = (img[c] - m) / s

            sample["image"] = torch.from_numpy(img.astype(np.float32))
            return sample

        # fallback
        sample["image"] = TF.to_tensor(img.astype(np.float32) / 255.0)
        return sample


def compute_sar_stats(paths, load_fn=None, replace_from='labels', replace_to='sar_images', max_files=None,
                      verbose=False):
    """
    Oblicza globalne wartości średniej i odchylenia standardowego (mean/std) dla kanału SAR
    na podstawie podanej listy ścieżek.
    - paths: lista ścieżek do plików etykiet lub do plików SAR
    - load_fn: opcjonalna funkcja load_fn(path) -> ndarray; jeśli None, używany jest rasterio
    Zwraca (mean, std) w skali [0,1].
    """
    total = 0
    sum_ = 0.0
    sumsq = 0.0
    n = len(paths)
    if max_files is not None:
        n = min(n, max_files)

    for i, p in enumerate(paths[:n]):
        sar_path = p
        if replace_from in p and replace_to not in p:
            sar_path = p.replace(replace_from, replace_to)
        arr = None
        try:
            if load_fn is not None:
                arr = load_fn(sar_path)
            else:
                with rasterio.open(sar_path, 'r') as src:
                    arr = src.read(1)
        except Exception as e:
            if verbose:
                print(f"compute_sar_stats: could not read {sar_path}: {e}")
            continue
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]
        # normalize to [0,1] according to dtype
        if arr.dtype == np.uint8:
            arrf = arr.astype(np.float32) / 255.0
        else:
            info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
            if info is not None:
                arrf = arr.astype(np.float32) / float(info.max)
            else:
                arrf = arr.astype(np.float32)
        total += arrf.size
        sum_ += float(arrf.sum())
        sumsq += float((arrf ** 2).sum())
        if verbose and (i + 1) % 50 == 0:
            print(f'compute_sar_stats: processed {i + 1}/{n}')

    if total == 0:
        return None, None
    mean = sum_ / total
    var = sumsq / total - mean * mean
    std = float(np.sqrt(var)) if var > 0 else 1.0
    return float(mean), float(std)


def _random_crop_np(image: np.ndarray, mask: np.ndarray, size: int, rng: np.random.RandomState):
    """Prosty crop (H,W[,C]) i (H,W) zwracany jako dict dla zgodności z resztą pipeline."""
    h, w = mask.shape[:2]
    if h == size and w == size:
        return {"image": image, "mask": mask}
    if h < size or w < size:
        # zakładamy, że A.PadIfNeeded wcześniej dopasował rozmiar; tu tylko asekuracja
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w)) + (() if image.ndim == 2 else ((0, 0),)), mode="constant")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        h, w = mask.shape[:2]
    y0 = int(rng.randint(0, h - size + 1))
    x0 = int(rng.randint(0, w - size + 1))
    if image.ndim == 2:
        img_c = image[y0:y0 + size, x0:x0 + size]
    else:
        img_c = image[y0:y0 + size, x0:x0 + size, ...]
    msk_c = mask[y0:y0 + size, x0:x0 + size]
    return {"image": img_c, "mask": msk_c}


def class_aware_random_crop(sample, size: int, target_class: int = 1, p: float = 0.5,
                           max_tries: int = 30, min_pixels: int = 20, seed: int = None):
    """Class-aware crop: z prawdopodobieństwem p próbuje wylosować crop zawierający >= min_pixels pikseli target_class.

    Działa na masce w formacie integer (przed one-hot). Jeśli klasa nie występuje / nie uda się w max_tries,
    robi zwykły losowy crop.

    Parametry:
    - target_class: wartość w masce (np. 1 dla c1)
    - p: prawdopodobieństwo użycia trybu class-aware dla danej próbki
    - max_tries: ile losowych cropów maksymalnie sprawdzić
    - min_pixels: minimalna liczba pikseli target_class w cropie
    """
    img = sample["image"]
    msk = sample["mask"]
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # jeśli nie włączamy, szybko wyjdź
    if p <= 0 or rng.rand() > p:
        return _random_crop_np(img, msk, size, rng)

    # jeśli klasa w ogóle nie występuje, fallback
    if not np.any(msk == target_class):
        return _random_crop_np(img, msk, size, rng)

    best = None
    best_cnt = -1
    for _ in range(max(1, int(max_tries))):
        out = _random_crop_np(img, msk, size, rng)
        cnt = int(np.sum(out["mask"] == target_class))
        if cnt > best_cnt:
            best_cnt = cnt
            best = out
        if cnt >= int(min_pixels):
            return out

    return best if best is not None else _random_crop_np(img, msk, size, rng)


# --- augmentations: proste, sensowne dla RGB i SAR ---
def train_augm1(sample, size=512, crop_cfg=None):
    """Minimalne augmentacje treningowe: tylko pad + crop.

    Przydatne jako baseline – nie wprowadza losowych transformacji poza przycięciem.

    Argumenty:
        sample (dict): wejściowy słownik z 'image' i 'mask'
        size (int): docelowy rozmiar cropu/resize
        crop_cfg (dict|None): konfiguracja cropu (opcjonalne klucze: enabled, target_class, p, max_tries, min_pixels)

    Zwraca:
        dict: wynik działania pipeline albumentations z polami 'image' i 'mask'
    """
    crop_cfg = crop_cfg or {}
    augms = [
        A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
    ]
    out = A.Compose(augms)(image=sample['image'], mask=sample['mask'])

    if crop_cfg.get("enabled", False):
        out = class_aware_random_crop(
            out,
            size=size,
            target_class=int(crop_cfg.get("target_class", 1)),
            p=float(crop_cfg.get("p", 0.5)),
            max_tries=int(crop_cfg.get("max_tries", 30)),
            min_pixels=int(crop_cfg.get("min_pixels", 20)),
        )
    else:
        out = A.RandomCrop(size, size, p=1.0)(image=out["image"], mask=out["mask"])
    return out


def valid_augm1(sample, size=512):
    """Funkcja augmentacji dla walidacji odpowiadająca `train_augm1` bez losowości.

    Zwraca obraz i maskę przeskalowane do `size` x `size`.
    """
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


# train_augm2: łagodne, uniwersalne augmentacje (geometria + lekka degradacja)
def train_augm2(sample, size=512, crop_cfg=None):
    """Łagodne augmentacje treningowe: geometria + lekka degradacja jakości.

    Zawiera: przesunięcia, skalowanie, rotację, flipy, oraz opcjonalny blur/gauss noise.
    Przydatne do zwiększenia różnorodności danych bez silnej zmiany kolorystyki.
    """
    crop_cfg = crop_cfg or {}
    augms = [
        # prosta geometria
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            value=0,
            p=0.7,
        ),
        A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        # lekka degradacja jakości (szum / blur)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
        ], p=0.3),
    ]
    out = A.Compose(augms)(image=sample['image'], mask=sample['mask'])

    if crop_cfg.get("enabled", False):
        out = class_aware_random_crop(
            out,
            size=size,
            target_class=int(crop_cfg.get("target_class", 1)),
            p=float(crop_cfg.get("p", 0.5)),
            max_tries=int(crop_cfg.get("max_tries", 30)),
            min_pixels=int(crop_cfg.get("min_pixels", 20)),
        )
    else:
        out = A.RandomCrop(size, size, p=1.0)(image=out["image"], mask=out["mask"])
    return out


def valid_augm2(sample, size=512):
    """Walidacyjne odpowiedniki `train_augm2` – deterministyczne (tylko resize).
    """
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


# train_augm3: mocniejsze augmentacje, nadal bez operacji typowo kolorystycznych
def train_augm3(sample, size=512, crop_cfg=None):
    """Mocniejsze augmentacje treningowe: większe transformacje geometryczne i degradacja obrazu.

    Zawiera: silniejsze ShiftScaleRotate, Downscale, MaskDropout (usuwanie obiektów), oraz mieszankę
    szumu/blur/sharpen. Nadaje się gdy chcemy mocniej uodpornić model na zakłócenia.
    """
    crop_cfg = crop_cfg or {}
    augms = [
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.15,
            rotate_limit=25,
            border_mode=0,
            value=0,
            p=0.8,
        ),
        A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        # sporadyczne pogorszenie jakości (downscale)
        A.Downscale(scale_min=0.6, scale_max=0.9, p=0.2),
        # okazjonalny dropout obiektów (jeśli maski na to pozwalają)
        A.MaskDropout(
            max_objects=2,
            image_fill_value=0,
            mask_fill_value=0,
            p=0.15,
        ),
        # trochę szumu / blur / sharpen
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=1.0),
        ], p=0.4),
    ]
    out = A.Compose(augms)(image=sample['image'], mask=sample['mask'])

    if crop_cfg.get("enabled", False):
        out = class_aware_random_crop(
            out,
            size=size,
            target_class=int(crop_cfg.get("target_class", 1)),
            p=float(crop_cfg.get("p", 0.5)),
            max_tries=int(crop_cfg.get("max_tries", 30)),
            min_pixels=int(crop_cfg.get("min_pixels", 20)),
        )
    else:
        out = A.RandomCrop(size, size, p=1.0)(image=out["image"], mask=out["mask"])
    return out


def valid_augm3(sample, size=512):
    """Walidacyjne odpowiedniki `train_augm3` – deterministyczne (tylko resize).
    """
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])