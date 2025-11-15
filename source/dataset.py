import numpy as np
import rasterio
from torch.utils.data import Dataset as BaseDataset
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


class Dataset(BaseDataset):
    def __init__(self, label_list, classes=None, size=128, train=False, sar_mean=None, sar_std=None, compute_stats=False, sar_normalize='global'):
        """label_list: list of label file paths (contains 'labels' in path)
        If compute_stats=True and train=True, dataset will compute global SAR mean/std
        using T.compute_sar_stats and the module-level load_grayscale.
        """
        self.fns = label_list
        self.augm = T.train_augm3 if train else T.valid_augm
        self.size = size
        self.train = train

        # optionally compute SAR stats from training files
        if train and compute_stats and (sar_mean is None or sar_std is None):
            try:
                sar_mean_calc, sar_std_calc = T.compute_sar_stats(self.fns, load_fn=load_grayscale, verbose=True)
                if sar_mean_calc is not None:
                    sar_mean = sar_mean_calc
                    sar_std = sar_std_calc
            except Exception as e:
                print("Warning: failed to compute SAR stats:", e)

        # store stats for diagnostics
        self.sar_mean = sar_mean
        self.sar_std = sar_std
        self.sar_normalize = sar_normalize

        # create ToTensor with SAR normalization info
        self.to_tensor = T.ToTensor(classes=classes, sar_mean=sar_mean, sar_std=sar_std, sar_normalize=sar_normalize)
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
