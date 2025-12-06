import numpy as np
import rasterio
from torch.utils.data import Dataset as BaseDataset
import os
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
            data = self.augm({"image": img, "mask": msk}, self.size)
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
            data = self.augm({"image": img, "mask": msk}, self.size)
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
            data = self.augm({"image": img, "mask": msk}, self.size)
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
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)  # image -> [4,H,W]
        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}