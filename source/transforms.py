import warnings
import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF
import torch
import rasterio

# reference: https://albumentations.ai/

warnings.simplefilter("ignore")

# ImageNet stats (float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ToTensor:
    def __init__(self, classes=None, sar_mean=None, sar_std=None, sar_normalize='global'):
        """
        Convert sample dict to tensors and apply normalization.

        - classes: optional list of class values for mask one-hot encoding (keeps backward compat)
        - sar_mean, sar_std: optional floats (on [0,1] scale) to normalize SAR channel globally
        - sar_normalize: 'global' | 'per_sample' | 'none'
        """
        self.classes = classes
        self.sar_mean = float(sar_mean) if sar_mean is not None else None
        self.sar_std = float(sar_std) if sar_std is not None else None
        self.sar_normalize = sar_normalize

    def __call__(self, sample):
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
    Compute global mean/std for SAR channel across provided list of paths.
    - paths: list of label paths or SAR paths
    - load_fn: optional function(path)->ndarray; if None, rasterio is used
    Returns (mean, std) on [0,1] scale.
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


# --- augmentations: proste, sensowne dla RGB i SAR ---

# train_augm1: minimalne augmentacje – tylko pad + crop (baseline)
def train_augm1(sample, size=512):
    augms = [
        A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
        A.RandomCrop(size, size, p=1.0),
    ]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


def valid_augm1(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])

# train_augm2: łagodne, uniwersalne augmentacje (geometria + lekka degradacja)
def train_augm2(sample, size=512):
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
        A.RandomCrop(size, size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        # lekka degradacja jakości (szum / blur)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
        ], p=0.3),
    ]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


def valid_augm2(sample, size=512):
    # Walidacja: brak losowych augmentacji, tylko zmiana rozmiaru.
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


# train_augm3: mocniejsze augmentacje, nadal bez operacji typowo kolorystycznych
def train_augm3(sample, size=512):
    augms = [
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.15,
            rotate_limit=25,
            border_mode=0,
            value=0,
            p=0.8,
        ),
        A.RandomCrop(size, size, p=1.0),
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
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


def valid_augm3(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])