import numpy as np
import matplotlib.pyplot as plt
import os


def progress(train_logs, valid_logs, loss_nm, metric_nm, nepochs, outdir, fn_out):
    loss_t = [dic[loss_nm] for dic in train_logs]
    loss_v = [dic[loss_nm] for dic in valid_logs]
    score_t = [dic[metric_nm] for dic in train_logs]
    score_v = [dic[metric_nm] for dic in valid_logs]

    epochs = range(0, len(score_t))
    plt.figure(figsize=(12, 5))

    # Train and validation metric
    # ---------------------------
    plt.subplot(1, 2, 1)

    idx = np.nonzero(score_t == max(score_t))[0][0]
    label = f"Train, {metric_nm}={max(score_t):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_t, "b", label=label)

    idx = np.nonzero(score_v == max(score_v))[0][0]
    label = f"Valid, {metric_nm}={max(score_v):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_v, "r", label=label)

    plt.title("Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel(metric_nm)
    plt.ylim(0, 1)
    plt.legend()

    # Train and validation loss
    # -------------------------
    plt.subplot(1, 2, 2)
    ymax = max(max(loss_t), max(loss_v))
    ymin = min(min(loss_t), min(loss_v))
    ymax = 1 if ymax <= 1 else ymax + 0.5
    ymin = 0 if ymin <= 0.5 else ymin - 0.5

    idx = np.nonzero(loss_t == min(loss_t))[0][0]
    label = f"Train {loss_nm}={min(loss_t):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_t, "b", label=label)

    idx = np.nonzero(loss_v == min(loss_v))[0][0]
    label = f"Valid {loss_nm}={min(loss_v):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_v, "r", label=label)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel("Loss")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(f"{outdir}/{fn_out}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    return


def log_epoch_results(log_path: str, epoch_idx: int, lr_str: str, train_logs: dict, valid_logs: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _g(d: dict, k: str):
        return float(d.get(k, float('nan')))

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Epoch: {epoch_idx} (lr: {lr_str})\n")
        f.write("Train Loss: {:.6f}, Valid Loss: {:.6f}\n".format(_g(train_logs, "loss"), _g(valid_logs, "loss")))
        f.write("Train IoU: {:.6f}, Valid IoU: {:.6f}\n".format(_g(train_logs, "iou"), _g(valid_logs, "iou")))
        f.write("Train Dice: {:.6f}, Valid Dice: {:.6f}\n".format(_g(train_logs, "dice"), _g(valid_logs, "dice")))
        f.write("Train Acc: {:.6f}, Valid Acc: {:.6f}\n".format(_g(train_logs, "acc"), _g(valid_logs, "acc")))
        f.write("Train Prec: {:.6f}, Valid Prec: {:.6f}\n".format(_g(train_logs, "prec"), _g(valid_logs, "prec")))
        f.write("Train Rec: {:.6f}, Valid Rec: {:.6f}\n".format(_g(train_logs, "rec"), _g(valid_logs, "rec")))
        f.write("Train F1: {:.6f}, Valid F1: {:.6f}\n".format(_g(train_logs, "f1"), _g(valid_logs, "f1")))

        # Opcjonalne: per-class IoU i F1 jeśli są dostępne
        def _get_per_class(d: dict, key: str):
            v = d.get(key, None)
            if v is None:
                return None
            # oczekujemy listy/ndarray/tensora 1D
            try:
                arr = np.asarray(v, dtype=float).ravel()
            except Exception:
                return None
            if arr.size == 0:
                return None
            return arr

        iou_t = _get_per_class(train_logs, "iou_per_class")
        iou_v = _get_per_class(valid_logs, "iou_per_class")
        f1_t = _get_per_class(train_logs, "f1_per_class")
        f1_v = _get_per_class(valid_logs, "f1_per_class")

        if iou_t is not None and iou_v is not None:
            f.write("Per-class IoU (Train):\n")
            for cid, val in enumerate(iou_t):
                f.write(f"  class_{cid}: {val:.6f}\n")
            f.write("Per-class IoU (Valid):\n")
            for cid, val in enumerate(iou_v):
                f.write(f"  class_{cid}: {val:.6f}\n")

        if f1_t is not None and f1_v is not None:
            f.write("Per-class F1 (Train):\n")
            for cid, val in enumerate(f1_t):
                f.write(f"  class_{cid}: {val:.6f}\n")
            f.write("Per-class F1 (Valid):\n")
            for cid, val in enumerate(f1_v):
                f.write(f"  class_{cid}: {val:.6f}\n")

        f.write("\n")

        print(f"\nEpoch: {epoch_idx} (lr: {lr_str})")
        print(f"Train Loss: {train_logs.get('loss'):.6f}, Valid Loss: {valid_logs.get('loss'):.6f}")
        print(f"Train IoU: {train_logs.get('iou'):.6f}, Valid IoU: {valid_logs.get('iou'):.6f}")
        print(f"Train Dice: {train_logs.get('dice'):.6f}, Valid Dice: {valid_logs.get('dice'):.6f}")
        print(f"Train Acc: {train_logs.get('acc'):.6f}, Valid Acc: {valid_logs.get('acc'):.6f}")
        print(f"Train Prec: {train_logs.get('prec'):.6f}, Valid Prec: {valid_logs.get('prec'):.6f}")
        print(f"Train Rec: {train_logs.get('rec'):.6f}, Valid Rec: {valid_logs.get('rec'):.6f}")
        print(f"Train F1: {train_logs.get('f1'):.6f}, Valid F1: {valid_logs.get('f1'):.6f}")
        print("Per-class IoU (Train):", ", ".join(f"c{cid}={val:.3f}" for cid, val in enumerate(iou_t)))
        print("Per-class IoU (Valid):", ", ".join(f"c{cid}={val:.3f}" for cid, val in enumerate(iou_v)))
        print("Per-class F1 (Train):", ", ".join(f"c{cid}={val:.3f}" for cid, val in enumerate(f1_t)))
        print("Per-class F1 (Valid):", ", ".join(f"c{cid}={val:.3f}" for cid, val in enumerate(f1_v)))


def _get_lr(optimizer):
    return ", ".join(f"{g['lr']:.6g}" for g in optimizer.param_groups)


def log_number_of_parameters(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)





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

# train_augm1: minimalne augmentacje – tylko pad + crop (baseline)
def train_augm1(sample, size=512, crop_cfg=None):
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
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


# train_augm2: łagodne, uniwersalne augmentacje (geometria + lekka degradacja)
def train_augm2(sample, size=512, crop_cfg=None):
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
    # Walidacja: brak losowych augmentacji, tylko zmiana rozmiaru.
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])


# train_augm3: mocniejsze augmentacje, nadal bez operacji typowo kolorystycznych
def train_augm3(sample, size=512, crop_cfg=None):
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
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample['image'], mask=sample['mask'])



import argparse
import os
import time
import warnings

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

import source
from source import streaming as S
from source.data_loader import build_data_loaders
from source.dataset import SARDataset
from source.utils import log_epoch_results, _get_lr, log_number_of_parameters

warnings.filterwarnings("ignore")


# =============================================
#   Main
# =============================================

def main(args):
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.9, device=0)
        except Exception:
            pass
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        classes=len(args.classes) + 1,
        in_channels=1,
        activation=None,
        encoder_weights=None if str(args.encoder_weights).lower() == "none" else args.encoder_weights,
        encoder_name=args.encoder_name,
        decoder_attention_type="scse",
    )
    log_number_of_parameters(model)

    # Te same wagi klas jak w treningu RGB (założenie: identyczny rozkład klas masek)
    classes_wt = np.array(
        [
            3.0,
            5.0,
            1.0,
            1.0,
            1.5,
            1.0,
            1.0,
            1.5,
            1.0
        ],
        dtype=np.float32,
    )
    if classes_wt.shape[0] != len(args.classes) + 1:
        raise ValueError(f"classes_wt length {classes_wt.shape[0]} != num_classes {len(args.classes) + 1}")

    loss_name = str(getattr(args, 'loss', 'ce')).lower()
    if loss_name in ("wce_dice", "wce+dice", "ce_dice"):
        criterion = source.losses.WeightedCEPlusDice(
            class_weights=classes_wt,
            ce_weight=float(getattr(args, 'ce_weight', 1.0)),
            dice_weight=float(getattr(args, 'dice_weight', 1.0)),
            dice_smooth=float(getattr(args, 'dice_smooth', 1.0)),
            dice_include_background=bool(int(getattr(args, 'dice_include_background', 0))),
            device=device,
        )
    else:
        criterion = source.losses.CEWithLogitsLoss(weights=classes_wt, device=device)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # build parameter groups: exclude biases and normalization params from weight decay
    target = model.module if hasattr(model, "module") else model
    decay_params = []
    no_decay_params = []
    for name, param in target.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if name.endswith('.bias') or 'bn' in lname or 'norm' in lname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(args.learning_rate),
    )

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=int(args.lr_patience),
            threshold=float(args.min_delta),
            min_lr=float(args.min_lr),
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.t_max),
            eta_min=float(args.min_lr),
        )

    # Weights & Biases (opcjonalnie)
    wandb_run = None
    if getattr(args, "use_wandb", False):
        wandb_config = {
            "seed": int(args.seed),
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "crop_size": int(args.crop_size),
            "learning_rate": float(args.learning_rate),
            "classes": args.classes,
            "data_root": args.data_root,
            "optimizer": "AdamW",
            "scheduler": args.scheduler,
            "lr_patience": int(args.lr_patience),
            "early_patience": int(args.early_patience),
            "min_delta": float(args.min_delta),
            "min_lr": float(args.min_lr),
            "t_max": int(args.t_max),
            "amp": bool(args.amp),
            "grad_clip": float(args.grad_clip) if args.grad_clip is not None else None,
            "model_name": f"Unet_{args.encoder_name}_sar",
            "criterion": criterion.name if hasattr(criterion, "name") else type(criterion).__name__,
            "model_type": "sar",
            "encoder_name": args.encoder_name,
            "encoder_weights": args.encoder_weights,
            "class_weights": classes_wt.tolist(),
        }

        run_name = (
            f"SAR_Unet_{args.encoder_name}_"
            f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
            f"_lr-{args.learning_rate}_"
            f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
        )

        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "SAR-train"),
            entity=getattr(args, "wandb_entity", None),
            config=wandb_config,
            name=run_name,
        )

        print("Number of epochs   :", args.n_epochs)
        print("Number of classes  :", len(args.classes) + 1)
        print("Batch size         :", args.batch_size)
        print("Device             :", device)
        print("AMP                :", args.amp)
        print("Grad clip          :", args.grad_clip)
        print("Encoder name       :", args.encoder_name)
        print("Encoder weights    :", args.encoder_weights)
        print("Class weights      :", classes_wt)

    train_model(args, model, optimizer, criterion, device, scheduler, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


# =============================================
#   Główna pętla treningowa z early stopping + scheduler
# =============================================

def train_model(args, model, optimizer, criterion, device, scheduler=None, wandb_run=None):
    train_loader, valid_loader = build_data_loaders(args, SARDataset)
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_results, exist_ok=True)

    model_name = (
        f"SAR_Unet_{args.encoder_name}_"
        f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
        f"_lr_{args.learning_rate}_"
        f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
    )
    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_results, "train_" + model_name)

    for epoch in range(args.n_epochs):
        logs_train = S.train_epoch_streaming(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
            grad_clip=float(args.grad_clip) if args.grad_clip is not None else None,
        )
        logs_valid = S.valid_epoch_streaming(
            model=model,
            criterion=criterion,
            dataloader=valid_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
        )

        # =============================================
        #   Logowanie wyników epoki (plik + konsola)
        # =============================================
        log_epoch_results(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        # =============================================
        #   Logowanie do Weights & Biases (jeśli włączone)
        # Zapisujemy wszystkie metryki z train/valid oraz aktualny learning rate.
        # =============================================
        if wandb_run is not None:
            log_dict = {"epoch": epoch + 1}
            # aktualny learning rate (pierwsza grupa parametrów)
            try:
                log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
            except Exception:
                pass

            def _log_metrics(prefix: str, logs: dict):
                for k, v in logs.items():
                    if k in ("iou_per_class", "f1_per_class"):
                        arr = np.asarray(v, dtype=float).ravel()
                        for cid, val in enumerate(arr):
                            if k == "iou_per_class":
                                log_dict[f"{prefix}/iou_class_{cid}"] = float(val)
                            elif k == "f1_per_class":
                                log_dict[f"{prefix}/f1_class_{cid}"] = float(val)
                    else:
                        try:
                            log_dict[f"{prefix}/{k}"] = float(v)
                        except Exception:
                            continue

            _log_metrics("train", logs_train)
            _log_metrics("valid", logs_valid)
            wandb_run.log(log_dict)

        score = logs_valid["iou"]
        if scheduler is not None:
            previous_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

            new_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if new_learning_rates != previous_learning_rates:
                print(f"LR changed: {previous_learning_rates} -> {new_learning_rates}")

        improvement = score - (max_score if max_score != -float("inf") else score)

        # zawsze zapisujemy poprawny state_dict (bez względu na DataParallel)
        state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

        best_path = os.path.join(args.save_model, f"{model_name}.pth")
        last_path = os.path.join(args.save_model, f"{model_name}_last.pth")

        if max_score == -float("inf") or improvement > float(args.min_delta):
            max_score = score
            bad_epochs = 0
            os.makedirs(args.save_model, exist_ok=True)
            torch.save(state_dict, best_path)
            print("Model saved:", os.path.abspath(best_path))
        else:
            bad_epochs += 1

        # Zapis awaryjny: zawsze zapisuj ostatni checkpoint (łatwiej debugować i nic nie ginie)
        try:
            os.makedirs(args.save_model, exist_ok=True)
            torch.save(state_dict, last_path)
        except Exception as e:
            print("Warning: failed to save last checkpoint:", e)

        if bad_epochs >= int(args.early_patience):
            print(f"Early stopping: brak poprawy IoU >= {args.min_delta} przez {args.early_patience} epok.")
            break

        if args.scheduler == "plateau":
            min_lr = float(args.min_lr)
            lrs = [g["lr"] for g in optimizer.param_groups]
            if all(lr <= min_lr + 1e-12 for lr in lrs) and improvement <= float(args.min_delta):
                print("LR osiągnął minimum i brak poprawy – zatrzymuję trening.")
                break


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    parser = argparse.ArgumentParser(description='Training SAR (streaming, memory-efficient)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', type=str, default='../dataset/train')
    parser.add_argument('--save_model', type=str, default='model/sar')
    parser.add_argument('--save_results', type=str, default="results")
    # --- AUGMENTACJE ---
    parser.add_argument('--train_augm', type=int, choices=[1, 2, 3], default=3,
                        help='Wybór trybu augmentacji dla treningu (train_augm1/2/3)')
    parser.add_argument('--valid_augm', type=int, choices=[1, 2, 3], default=1,
                        help='Wybór trybu augmentacji dla walidacji (valid_augm1/2/3)')
    # --- ENKODER ---
    parser.add_argument('--encoder_name', type=str, default='tu-convnext_tiny',
                        help='Nazwa enkodera z segmentation_models_pytorch, np. efficientnet-b4, resnet34, tu-convnext_tiny, swin_tiny_patch4_window7_224')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='Wagi enkodera, np. imagenet, ssl, swsl lub none (brak wag)')
    # --- SCHEDULER / EARLY STOPPING PARAMS ---
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine', 'none'], default='plateau')
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--early_patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--t_max', type=int, default=30)
    parser.add_argument('--amp', type=int, default=0, help='Włącz / wyłącz mixed precision (1/0)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maks. norm gradientu; None aby wyłączyć')

    # --- Loss ---
    parser.add_argument('--loss', type=str, default='wce_dice', choices=['ce', 'wce_dice'],
                        help='Funkcja straty: ce (weighted CE) lub wce_dice (weighted CE + Dice)')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Waga składnika CE w loss wce_dice')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Waga składnika Dice w loss wce_dice')
    parser.add_argument('--dice_smooth', type=float, default=1.0, help='Smooth dla DiceLoss')
    parser.add_argument('--dice_include_background', type=int, default=0, choices=[0, 1],
                        help='Czy uwzględniać tło (klasa 0) w Dice (0/1)')

    # --- Class-aware crop / oversampling (pod rzadkie klasy, np. c1) ---
    parser.add_argument('--class_aware_crop', type=int, default=1, choices=[0, 1],
                        help='Włącz class-aware cropping na treningu (0/1)')
    parser.add_argument('--oversample_class', type=int, default=1, help='Wartość klasy w masce, którą preferujemy w crop (np. 1 dla c1)')
    parser.add_argument('--oversample_p', type=float, default=0.5, help='Prawdopodobieństwo użycia class-aware crop dla próbki (np. 0.5)')
    parser.add_argument('--oversample_min_pixels', type=int, default=200, help='Minimalna liczba pikseli target_class w crop')
    parser.add_argument('--oversample_max_tries', type=int, default=50, help='Ile losowych cropów próbować zanim fallback')

    # --- Weights & Biases ---
    parser.add_argument('--use_wandb', type=bool, default=True, help='Włącz logowanie do Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='SAR-train', help='Nazwa projektu w W&B')
    parser.add_argument('--wandb_entity', type=str, default='radoslaw-godlewski00-politechnika-warszawska',
                        help='Nazwa entity w W&B')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)



import argparse
import os
import time
import warnings

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

import source
from source import streaming as S
from source.data_loader import build_data_loaders
from source.dataset import RGBDataset
from source.utils import log_epoch_results, _get_lr, log_number_of_parameters

# Filtrujemy ostrzeżenia by log był czytelniejszy.
(warnings.filterwarnings("ignore"))


# =============================================
#   Main
# =============================================
# Funkcja `main` inicjalizuje wszystkie komponenty potrzebne do treningu:
# - ustawia ziarno losowości (reproducibility), opcje cudnn
# - tworzy model (UNet z EfficientNet-B4) i kryterium strat
# - przenosi model na GPU(y) i opakowuje w DataParallel jeśli jest więcej niż 1 GPU
# - tworzy optimizer (AdamW z grupowaniem parametrów) oraz scheduler
# - opcjonalnie inicjalizuje Weights & Biases (wandb)
# - uruchamia pętlę treningową `train_model`

def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # wymuszamy deterministyczne zachowanie cudnn (może spowolnić trening, ale daje powtarzalność)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ograniczamy zużycie GPU (opcjonalne, ustawienie specyficzne dla Twojego środowiska)
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =============================================
    #   Model i criterion
    # Tworzymy architekturę UNet z parametryzowanym enkoderem.
    # =============================================
    model = smp.Unet(
        classes=len(args.classes) + 1,
        in_channels=3,
        activation=None,
        encoder_weights=None if str(args.encoder_weights).lower() == "none" else args.encoder_weights,
        encoder_name=args.encoder_name,
        decoder_attention_type="scse",
    )
    log_number_of_parameters(model)
    classes_wt = np.array([
        3.0,
        5.0,
        1.0,
        1.0,
        1.5,
        1.0,
        1.0,
        1.5,
        1.0
    ],
        dtype=np.float32,
    )
    if classes_wt.shape[0] != len(args.classes) + 1:
        raise ValueError(f"classes_wt length {classes_wt.shape[0]} != num_classes {len(args.classes) + 1}")

    loss_name = str(getattr(args, 'loss', 'ce')).lower()
    if loss_name in ("wce_dice", "wce+dice", "ce_dice"):
        criterion = source.losses.WeightedCEPlusDice(
            class_weights=classes_wt,
            ce_weight=float(getattr(args, 'ce_weight', 1.0)),
            dice_weight=float(getattr(args, 'dice_weight', 1.0)),
            dice_smooth=float(getattr(args, 'dice_smooth', 1.0)),
            dice_include_background=bool(int(getattr(args, 'dice_include_background', 0))),
            device=device,
        )
    else:
        criterion = source.losses.CEWithLogitsLoss(weights=classes_wt, device=device)

    # =============================================
    #   Device i DataParallel
    # Przenosimy model na urządzenie (GPU/CPU). Jeśli jest więcej GPU, opakowujemy model
    # w torch.nn.DataParallel aby trenować równolegle na wielu kartach.
    # =============================================
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # =============================================
    #   Optimizer i grupowanie parametrów (AdamW)
    #   Poniżej tworzymy dwie grupy parametrów:
    # - decay_params: parametry podlegające weight decay (zwykle wagi konwolucji/linearnych)
    # - no_decay_params: biasy i parametry normalizacji, które zwykle nie dostają weight decay
    # Dzięki temu regularyzacja (weight decay) działa poprawnie i nie niszczy parametrów takich jak bias/Norm.
    # =============================================
    target = model.module if hasattr(model, "module") else model
    decay_params = []
    no_decay_params = []
    for name, param in target.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        # reguła prostego wykrywania parametrów normalizacyjnych i biasów
        if name.endswith('.bias') or 'bn' in lname or 'norm' in lname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Tworzymy optimizer AdamW z dwiema grupami parametrów.
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(args.learning_rate),
    )

    # =============================================
    #   Scheduler
    # =============================================
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=int(args.lr_patience),
            threshold=float(args.min_delta),
            min_lr=float(args.min_lr)
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.t_max),
            eta_min=float(args.min_lr),
        )

    # =============================================
    #   Weights & Biases (opcjonalnie)
    # Inicjalizujemy run w wandb tylko jeśli użytkownik poda --use_wandb.
    # Do config wrzucamy najważniejsze hiperparametry i opis modelu.
    # =============================================
    wandb_run = None
    if getattr(args, "use_wandb", False):
        wandb_config = {
            "seed": args.seed,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "crop_size": args.crop_size,
            "learning_rate": float(args.learning_rate),
            "classes": args.classes,
            "data_root": args.data_root,
            "optimizer": "AdamW",
            "scheduler": args.scheduler,
            "lr_patience": int(args.lr_patience),
            "early_patience": int(args.early_patience),
            "min_delta": float(args.min_delta),
            "min_lr": float(args.min_lr),
            "t_max": int(args.t_max),
            "amp": bool(args.amp),
            "grad_clip": float(args.grad_clip) if args.grad_clip is not None else None,
            "model_name": f"Unet_{args.encoder_name}_rgb",
            "criterion": criterion.name if hasattr(criterion, "name") else type(criterion).__name__,
            "model_type": "rgb",
            "encoder_name": args.encoder_name,
            "encoder_weights": args.encoder_weights,
            "class_weights": classes_wt.tolist(),
        }
        run_name = (
            f"RGB_Unet_{args.encoder_name}_"
            f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
            f"_lr:{args.learning_rate}_"
            f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
        )
        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "RGB-train"),
            entity=getattr(args, "wandb_entity", None),
            config=wandb_config,
            name=run_name,
        )

    print("Number of epochs   :", args.n_epochs)
    print("Number of classes  :", len(args.classes) + 1)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)
    print("AMP                :", args.amp)
    print("Grad clip          :", args.grad_clip)
    print("Encoder name       :", args.encoder_name)
    print("Encoder weights    :", args.encoder_weights)
    print("Class weights      :", classes_wt)
    # Uruchamiamy pętlę treningową
    train_model(args, model, optimizer, criterion, device, scheduler, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


# =============================================
#   Główna pętla treningowa z early stopping + scheduler
# Funkcja `train_model` odpowiada za cały przebieg treningu po epokach:
# - tworzy dataloadery przez `build_data_loaders`
# - dla każdej epoki: trening (streaming), walidacja (streaming), logowanie wyników
# - aktualizuje scheduler (ReduceLROnPlateau z metryką lub Cosine bez metryki)
# - realizuje logiczkę checkpointów (zapisywanie najlepszych wag) i early stopping
# - opcjonalnie loguje metryki do wandb
# =============================================
def train_model(args, model, optimizer, criterion, device, scheduler=None, wandb_run=None):
    train_data_loader, val_data_loader = build_data_loaders(args, RGBDataset)
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_results, exist_ok=True)
    model_name = (
        f"RGB_Unet_{args.encoder_name}_"
        f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
        f"_lr_{args.learning_rate}_"
        f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
    )
    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_results, "train_rgb.txt")

    for epoch in range(args.n_epochs):
        # --- trening strumieniowy ---
        # Korzystamy z funkcji streaming.train_epoch_streaming, która wykonuje trening na batchach
        # w sposób memory-safe (strumieniowo): forward, backward oraz optimizer.step wywoływane dla batcha.
        logs_train = S.train_epoch_streaming(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
            grad_clip=float(args.grad_clip) if args.grad_clip is not None else None,
        )
        # --- walidacja strumieniowa ---
        logs_valid = S.valid_epoch_streaming(
            model=model,
            criterion=criterion,
            dataloader=val_data_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
        )

        # =============================================
        #   Logowanie wyników epoki (plik + konsola)
        # =============================================
        log_epoch_results(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        # =============================================
        #   Logowanie do Weights & Biases (jeśli włączone)
        # Zapisujemy wszystkie metryki z train/valid oraz aktualny learning rate.
        # =============================================
        if wandb_run is not None:
            log_dict = {"epoch": epoch + 1}
            # aktualny learning rate (pierwsza grupa parametrów)
            try:
                log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
            except Exception:
                pass

            def _log_metrics(prefix: str, logs: dict):
                for k, v in logs.items():
                    if k in ("iou_per_class", "f1_per_class"):
                        arr = np.asarray(v, dtype=float).ravel()
                        for cid, val in enumerate(arr):
                            if k == "iou_per_class":
                                log_dict[f"{prefix}/iou_class_{cid}"] = float(val)
                            elif k == "f1_per_class":
                                log_dict[f"{prefix}/f1_class_{cid}"] = float(val)
                    else:
                        try:
                            log_dict[f"{prefix}/{k}"] = float(v)
                        except Exception:
                            continue

            _log_metrics("train", logs_train)
            _log_metrics("valid", logs_valid)
            wandb_run.log(log_dict)

        # --- LR scheduler ---
        score = logs_valid["iou"]
        if scheduler is not None:
            previous_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

            new_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if new_learning_rates != previous_learning_rates:
                print(f"LR changed: {previous_learning_rates} -> {new_learning_rates}")

        # --- EARLY STOPPING + checkpoint ---
        # Jeśli nastąpi poprawa (większe IoU) zapisujemy checkpoint i resetujemy licznik "złych" epok.
        improvement = score - (max_score if max_score != -float("inf") else score)

        # zawsze zapisujemy poprawny state_dict (bez względu na DataParallel)
        state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

        best_path = os.path.join(args.save_model, f"{model_name}.pth")
        last_path = os.path.join(args.save_model, f"{model_name}_last.pth")

        if max_score == -float("inf") or improvement > float(args.min_delta):
            max_score = score
            bad_epochs = 0
            os.makedirs(args.save_model, exist_ok=True)
            torch.save(state_dict, best_path)
            print("Model saved:", os.path.abspath(best_path))
        else:
            bad_epochs += 1

        # Zapis awaryjny: zawsze zapisuj ostatni checkpoint (łatwiej debugować i nic nie ginie)
        try:
            os.makedirs(args.save_model, exist_ok=True)
            torch.save(state_dict, last_path)
        except Exception as e:
            print("Warning: failed to save last checkpoint:", e)

        # Early stopping: jeśli przez `early_patience` epok nie ma poprawy, przerywamy trening
        if bad_epochs >= int(args.early_patience):
            print(f"Early stopping: brak poprawy IoU >= {args.min_delta} przez {args.early_patience} epok.")
            break

        # Dodatkowa logika dla scheduler="plateau": jeśli LR osiągnął min i brak poprawy -> stop
        if args.scheduler == "plateau":
            min_lr = float(args.min_lr)
            lrs = [g["lr"] for g in optimizer.param_groups]
            if all(lr <= min_lr + 1e-12 for lr in lrs) and improvement <= float(args.min_delta):
                print("LR osiągnął minimum i brak poprawy – zatrzymuję trening.")
                break


# =============================================
#   Punkt wejścia skryptu
# Parametry domyślne można nadpisać przez CLI.
# =============================================
if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    parser = argparse.ArgumentParser(description='Model Training RGB')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', type=str, default="../dataset/train")
    parser.add_argument('--save_model', type=str, default="model/rgb")
    parser.add_argument('--save_results', type=str, default="results")
    # --- AUGMENTACJE ---
    parser.add_argument('--train_augm', type=int, choices=[1, 2, 3], default=3,
                        help='Wybór trybu augmentacji dla treningu (train_augm1/2/3)')
    parser.add_argument('--valid_augm', type=int, choices=[1, 2, 3], default=1,
                        help='Wybór trybu augmentacji dla walidacji (valid_augm1/2/3)')
    # --- ENKODER ---
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b4',
                        help='Nazwa enkodera z segmentation_models_pytorch, np. efficientnet-b4, resnet34, tu-convnext_tiny')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='Wagi enkodera, np. imagenet, ssl, swsl lub none (brak wag)')
    # --- SCHEDULER / EARLY STOPPING PARAMS ---
    parser.add_argument('--scheduler', choices=['plateau', 'cosine', 'none'], default='plateau')
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--early_patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--t_max', type=int, default=30)
    parser.add_argument('--amp', type=int, default=1, help='Włącz / wyłącz mixed precision (1/0)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maks. norm gradientu; None aby wyłączyć')

    # --- Loss ---
    parser.add_argument('--loss', type=str, default='wce_dice', choices=['ce', 'wce_dice'],
                        help='Funkcja straty: ce (weighted CE) lub wce_dice (weighted CE + Dice)')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Waga składnika CE w loss wce_dice')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Waga składnika Dice w loss wce_dice')
    parser.add_argument('--dice_smooth', type=float, default=1.0, help='Smooth dla DiceLoss')
    parser.add_argument('--dice_include_background', type=int, default=0, choices=[0, 1],
                        help='Czy uwzględniać tło (klasa 0) w Dice (0/1)')

    # --- Class-aware crop / oversampling (pod rzadkie klasy, np. c1) ---
    parser.add_argument('--class_aware_crop', type=int, default=1, choices=[0, 1],
                        help='Włącz class-aware cropping na treningu (0/1)')
    parser.add_argument('--oversample_class', type=int, default=1, help='Wartość klasy w masce, którą preferujemy w crop (np. 1 dla c1)')
    parser.add_argument('--oversample_p', type=float, default=0.5, help='Prawdopodobieństwo użycia class-aware crop dla próbki (np. 0.5)')
    parser.add_argument('--oversample_min_pixels', type=int, default=200, help='Minimalna liczba pikseli target_class w crop')
    parser.add_argument('--oversample_max_tries', type=int, default=50, help='Ile losowych cropów próbować zanim fallback')

    # --- Weights & Biases ---
    parser.add_argument('--use_wandb', type=bool, default=True, help='Włącz logowanie do Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='RGB-train', help='Nazwa projektu w W&B')
    parser.add_argument('--wandb_entity', type=str, default='radoslaw-godlewski00-politechnika-warszawska',
                        help='Nazwa entity w W&B')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)


import os
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

import source
from source import streaming as S
from source.data_loader import build_data_loaders
from source.dataset import FusionDataset
from source.utils import log_epoch_results, _get_lr, log_number_of_parameters

warnings.filterwarnings("ignore")


# =============================================
#   Late Fusion model
# =============================================

class LateFusionUnet(nn.Module):
    """Late Fusion dla segmentacji:

    - oddzielny model RGB (3ch) i SAR (1ch)
    - fuzja na poziomie LOGITÓW: (logits_rgb + logits_sar) / 2 lub ważona suma

    Kontrakt wejścia:
    - `x` ma mieć 4 kanały w kolejności dokładnie jak `FusionDataset` w `source/dataset.py`:
        img = np.dstack([rgb, sar[..., None]])  -> [R,G,B,SAR]
      czyli w tensorze: x[:, 0:3] = RGB, x[:, 3:4] = SAR.

    Wejście: x (B,4,H,W) gdzie kanały: [R,G,B,SAR]
    Wyjście: logits_fused (B,C,H,W)
    """

    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "efficientnet-b4",
        encoder_weights_rgb: str | None = "imagenet",
        encoder_weights_sar: str | None = None,
        decoder_attention_type: str | None = "scse",
        fusion_mode: str = "mean",  # mean | sum | weighted
    ):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.rgb_model = smp.Unet(
            classes=num_classes,
            in_channels=3,
            activation=None,
            encoder_name=encoder_name,
            encoder_weights=None if str(encoder_weights_rgb).lower() == "none" else encoder_weights_rgb,
            decoder_attention_type=decoder_attention_type,
        )

        self.sar_model = smp.Unet(
            classes=num_classes,
            in_channels=1,
            activation=None,
            encoder_name=encoder_name,
            encoder_weights=None if str(encoder_weights_sar).lower() == "none" else encoder_weights_sar,
            decoder_attention_type=decoder_attention_type,
        )

        # Uczone wagi dla fuzji (tylko dla fusion_mode='weighted')
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))  # po sigmoidzie: waga RGB w [0,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 4:
            raise ValueError(f"Expected input (B,4,H,W), got {tuple(x.shape)}")

        x_rgb = x[:, :3]
        x_sar = x[:, 3:4]

        logits_rgb = self.rgb_model(x_rgb)
        logits_sar = self.sar_model(x_sar)

        if self.fusion_mode == "sum":
            return logits_rgb + logits_sar

        if self.fusion_mode == "weighted":
            w_rgb = torch.sigmoid(self.logit_alpha)  # 0..1
            return w_rgb * logits_rgb + (1.0 - w_rgb) * logits_sar

        # default: mean
        return 0.5 * (logits_rgb + logits_sar)


# =============================================
#   Main
# =============================================

def main(args):
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        except Exception:
            pass
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = len(args.classes) + 1

    model = LateFusionUnet(
        num_classes=num_classes,
        encoder_name=args.encoder_name,
        encoder_weights_rgb=args.encoder_weights,
        encoder_weights_sar=None,  # SAR: domyślnie bez imagenet
        decoder_attention_type="scse",
        fusion_mode=args.fusion_mode,
    )

    log_number_of_parameters(model)

    classes_wt = np.array(
        [
            3.0,
            5.0,
            1.0,
            1.0,
            1.5,
            1.0,
            1.0,
            1.5,
            1.0,
        ],
        dtype=np.float32,
    )
    if classes_wt.shape[0] != num_classes:
        raise ValueError(f"classes_wt length {classes_wt.shape[0]} != num_classes {num_classes}")

    loss_name = str(getattr(args, "loss", "ce")).lower()
    if loss_name in ("wce_dice", "wce+dice", "ce_dice"):
        criterion = source.losses.WeightedCEPlusDice(
            class_weights=classes_wt,
            ce_weight=float(getattr(args, "ce_weight", 1.0)),
            dice_weight=float(getattr(args, "dice_weight", 1.0)),
            dice_smooth=float(getattr(args, "dice_smooth", 1.0)),
            dice_include_background=bool(int(getattr(args, "dice_include_background", 0))),
            device=device,
        )
    else:
        criterion = source.losses.CEWithLogitsLoss(weights=classes_wt, device=device)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # Optimizer - wspólne grupowanie parametrów jak w innych skryptach
    target = model.module if hasattr(model, "module") else model
    decay_params = []
    no_decay_params = []
    for name, param in target.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if name.endswith(".bias") or "bn" in lname or "norm" in lname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(args.learning_rate),
    )

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=int(args.lr_patience),
            threshold=float(args.min_delta),
            min_lr=float(args.min_lr),
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.t_max),
            eta_min=float(args.min_lr),
        )

    # W&B
    wandb_run = None
    if getattr(args, "use_wandb", False):
        wandb_config = {
            "seed": int(args.seed),
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "crop_size": int(args.crop_size),
            "learning_rate": float(args.learning_rate),
            "classes": args.classes,
            "data_root": args.data_root,
            "optimizer": "AdamW",
            "scheduler": args.scheduler,
            "lr_patience": int(args.lr_patience),
            "early_patience": int(args.early_patience),
            "min_delta": float(args.min_delta),
            "min_lr": float(args.min_lr),
            "t_max": int(args.t_max),
            "amp": bool(args.amp),
            "grad_clip": float(args.grad_clip) if args.grad_clip is not None else None,
            "model_name": f"LateFusion_Unet_{args.encoder_name}",
            "criterion": criterion.name if hasattr(criterion, "name") else type(criterion).__name__,
            "model_type": "late_fusion",
            "encoder_name": args.encoder_name,
            "encoder_weights_rgb": args.encoder_weights,
            "encoder_weights_sar": None,
            "class_weights": classes_wt.tolist(),
            "fusion_mode": args.fusion_mode,
        }

        run_name = (
            f"LATEFUSION_Unet_{args.encoder_name}_"
            f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
            f"_lr-{args.learning_rate}_"
            f"fusion-{args.fusion_mode}_"
            f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
        )

        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "LATEFUSION-train"),
            entity=getattr(args, "wandb_entity", None),
            config=wandb_config,
            name=run_name,
        )

    print("Number of epochs   :", args.n_epochs)
    print("Number of classes  :", num_classes)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)
    print("AMP                :", args.amp)
    print("Grad clip          :", args.grad_clip)
    print("Encoder name       :", args.encoder_name)
    print("Encoder weights(RGB):", args.encoder_weights)
    print("Fusion mode        :", args.fusion_mode)

    train_model(args, model, optimizer, criterion, device, scheduler, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


# =============================================
#   Train loop
# =============================================

def train_model(args, model, optimizer, criterion, device, scheduler=None, wandb_run=None):
    # Używamy FusionDataset, bo zwraca x z 4 kanałami [R,G,B,SAR]
    train_loader, valid_loader = build_data_loaders(args, FusionDataset)
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_results, exist_ok=True)

    model_name = (
        f"LATE_FUSION_Unet_{args.encoder_name}_"
        f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
        f"_lr_{args.learning_rate}_"
        f"fusion_{args.fusion_mode}_"
        f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
    )

    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_results, "train_late_fusion.txt")

    for epoch in range(args.n_epochs):
        logs_train = S.train_epoch_streaming(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
            grad_clip=float(args.grad_clip) if args.grad_clip is not None else None,
        )
        logs_valid = S.valid_epoch_streaming(
            model=model,
            criterion=criterion,
            dataloader=valid_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
        )

        log_epoch_results(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        if wandb_run is not None:
            log_dict = {"epoch": epoch + 1}
            try:
                log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
            except Exception:
                pass

            def _log_metrics(prefix: str, logs: dict):
                for k, v in logs.items():
                    if k in ("iou_per_class", "f1_per_class"):
                        arr = np.asarray(v, dtype=float).ravel()
                        for cid, val in enumerate(arr):
                            if k == "iou_per_class":
                                log_dict[f"{prefix}/iou_class_{cid}"] = float(val)
                            elif k == "f1_per_class":
                                log_dict[f"{prefix}/f1_class_{cid}"] = float(val)
                    else:
                        try:
                            log_dict[f"{prefix}/{k}"] = float(v)
                        except Exception:
                            continue

            _log_metrics("train", logs_train)
            _log_metrics("valid", logs_valid)
            wandb_run.log(log_dict)

        score = logs_valid["iou"]
        if scheduler is not None:
            previous_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

            new_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if new_learning_rates != previous_learning_rates:
                print(f"LR changed: {previous_learning_rates} -> {new_learning_rates}")

        improvement = score - (max_score if max_score != -float("inf") else score)
        state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

        best_path = os.path.join(args.save_model, f"{model_name}.pth")
        last_path = os.path.join(args.save_model, f"{model_name}_last.pth")

        if max_score == -float("inf") or improvement > float(args.min_delta):
            max_score = score
            bad_epochs = 0
            os.makedirs(args.save_model, exist_ok=True)
            torch.save(state_dict, best_path)
            print("Model saved:", os.path.abspath(best_path))
        else:
            bad_epochs += 1

        try:
            torch.save(state_dict, last_path)
        except Exception as e:
            print("Warning: failed to save last checkpoint:", e)

        if bad_epochs >= int(args.early_patience):
            print(f"Early stopping: brak poprawy IoU >= {args.min_delta} przez {args.early_patience} epok.")
            break

        if args.scheduler == "plateau":
            min_lr = float(args.min_lr)
            lrs = [g["lr"] for g in optimizer.param_groups]
            if all(lr <= min_lr + 1e-12 for lr in lrs) and improvement <= float(args.min_delta):
                print("LR osiągnął minimum i brak poprawy – zatrzymuję trening.")
                break


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    parser = argparse.ArgumentParser(description="Training Late Fusion (RGB + SAR) - streaming")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--classes", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--data_root", type=str, default="../dataset/train")
    parser.add_argument("--save_model", type=str, default="model/fusion")
    parser.add_argument("--save_results", type=str, default="results")

    # AUGMENTACJE
    parser.add_argument("--train_augm", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--valid_augm", type=int, choices=[1, 2, 3], default=1)

    # ENKODER
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")

    # LATE FUSION SPECIFIC
    parser.add_argument("--fusion_mode", type=str, default="mean", choices=["mean", "sum", "weighted"],
        help="Jak łączyć logity: mean (średnia), sum (suma), weighted (uczony blending).")

    # SCHEDULER / EARLY STOPPING
    parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--early_patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--t_max", type=int, default=30)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # LOSS
    parser.add_argument("--loss", type=str, default="wce_dice", choices=["ce", "wce_dice"])
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--dice_smooth", type=float, default=1.0)
    parser.add_argument("--dice_include_background", type=int, default=0, choices=[0, 1])

    # Class-aware crop / oversampling
    parser.add_argument("--class_aware_crop", type=int, default=1, choices=[0, 1])
    parser.add_argument("--oversample_class", type=int, default=1)
    parser.add_argument("--oversample_p", type=float, default=0.5)
    parser.add_argument("--oversample_min_pixels", type=int, default=200)
    parser.add_argument("--oversample_max_tries", type=int, default=50)

    # W&B
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="FUSION-train")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="radoslaw-godlewski00-politechnika-warszawska",
    )

    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print("Processing time:", end - start)


import os
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead

import source
from source import streaming as S
from source.data_loader import build_data_loaders
from source.dataset import FusionDataset
from source.utils import log_epoch_results, _get_lr, log_number_of_parameters

warnings.filterwarnings("ignore")


# =============================================
#   Joint Fusion (feature-level) model
# =============================================

class JointFusionUnet(nn.Module):
    """Joint/Intermediate Fusion: 2 encodery + wspólny dekoder.

    - encoder_rgb: EfficientNet-B4 (lub inny z SMP) dla RGB (3 kanały)
    - encoder_sar: EfficientNet-B4 (lub inny z SMP) dla SAR (1 kanał)
    - na każdym poziomie (skip + bottleneck) łączymy feature maps
    - wspólny decoder UNet odtwarza maskę

    Wejście:
      x: (B,4,H,W) w kolejności [R,G,B,SAR] (zgodnie z FusionDataset)

    feature_fusion:
      - 'concat' (domyślnie): cat po kanałach + 1x1 conv do kanałów RGB
      - 'sum': sumujemy (po wyrównaniu kanałów paddingiem) + 1x1 conv do kanałów RGB
      - 'mean': jak sum, ale /2

    Dzięki wspólnemu decoderowi gradienty przechodzą do obu encoderów jednocześnie.
    """

    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "efficientnet-b4",
        encoder_weights_rgb: str | None = "imagenet",
        encoder_weights_sar: str | None = None,
        decoder_attention_type: str | None = "scse",
        feature_fusion: str = "concat",
        encoder_depth: int = 5,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        decoder_use_norm: str | bool | dict = "batchnorm",
        decoder_add_center_block: bool = False,
        safe_nan_to_num: bool = True,
    ):
        super().__init__()

        self.safe_nan_to_num = bool(safe_nan_to_num)

        self.feature_fusion = str(feature_fusion).lower()
        if self.feature_fusion not in ("concat", "sum", "mean"):
            raise ValueError(f"feature_fusion must be one of concat/sum/mean, got {feature_fusion}")

        # 2 encodery
        self.encoder_rgb = get_encoder(
            encoder_name,
            in_channels=3,
            depth=int(encoder_depth),
            weights=None if str(encoder_weights_rgb).lower() == "none" else encoder_weights_rgb,
        )
        self.encoder_sar = get_encoder(
            encoder_name,
            in_channels=1,
            depth=int(encoder_depth),
            weights=None if str(encoder_weights_sar).lower() == "none" else encoder_weights_sar,
        )

        encoder_channel_rgb = list(self.encoder_rgb.out_channels)
        encoder_channel_sar = list(self.encoder_sar.out_channels)
        if len(encoder_channel_rgb) != len(encoder_channel_sar):
            raise ValueError(
                f"Encoder channel lists differ: rgb={len(encoder_channel_rgb)} sar={len(encoder_channel_sar)}"
            )

        # warstwy fuzji per-level -> sprowadzamy do kanałów RGB (żeby decoder był spójny)
        # UWAGA: channel_sar NIE jest ignorowany — wpływa na in_channels tej projekcji.
        # out_channels ustawiamy na channel_rgb celowo: UnetDecoder dostaje encoder_channels=encoder_channel_rgb,
        # więc fused feature maps muszą mieć takie same liczby kanałów jak gałąź RGB.
        # 1x1 conv uczy się "wymieszać" (połączyć) informację z RGB+SAR do tej wspólnej przestrzeni.
        self.fuse_convs = nn.ModuleList()
        for channel_rgb, channel_sar in zip(encoder_channel_rgb, encoder_channel_sar):
            in_channel = (channel_rgb + channel_sar) if self.feature_fusion == "concat" else max(channel_rgb, channel_sar)
            # GroupNorm jest stabilniejszy niż BatchNorm przy batch_size=1 (valid) i AMP.
            # Dobieramy liczbę grup tak, aby dzieliła channel_rgb.
            g = 32
            while g > 1 and (channel_rgb % g) != 0:
                g //= 2
            if g < 1:
                g = 1
            self.fuse_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, channel_rgb, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=g, num_channels=channel_rgb),
                    nn.ReLU(inplace=True),
                )
            )

        # wspólny decoder
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channel_rgb,
            decoder_channels=decoder_channels,
            n_blocks=int(encoder_depth),
            use_norm=decoder_use_norm,
            attention_type=decoder_attention_type,
            add_center_block=bool(decoder_add_center_block),
            interpolation_mode="nearest",
        )

        self.segmentation_head = SegmentationHead(
            in_channels=int(decoder_channels[-1]),
            out_channels=int(num_classes),
            activation=None,
            kernel_size=3,
        )

    @staticmethod
    def _pad_channels(t: torch.Tensor, c: int) -> torch.Tensor:
        if t.size(1) == c:
            return t
        if t.size(1) > c:
            return t[:, :c]
        pad = torch.zeros(
            (t.size(0), c - t.size(1), t.size(2), t.size(3)),
            device=t.device,
            dtype=t.dtype,
        )
        return torch.cat([t, pad], dim=1)

    def _fuse(self, fr: torch.Tensor, fs: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        if self.feature_fusion == "concat":
            x = torch.cat([fr, fs], dim=1)
        else:
            max_c = max(fr.size(1), fs.size(1))
            frp = self._pad_channels(fr, max_c)
            fsp = self._pad_channels(fs, max_c)
            x = frp + fsp
            if self.feature_fusion == "mean":
                x = 0.5 * x
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 4:
            raise ValueError(f"Expected input (B,4,H,W), got {tuple(x.shape)}")

        x_rgb = x[:, :3]
        x_sar = x[:, 3:4]

        feats_rgb = self.encoder_rgb(x_rgb)
        feats_sar = self.encoder_sar(x_sar)

        fused_feats = [self._fuse(fr, fs, conv) for fr, fs, conv in zip(feats_rgb, feats_sar, self.fuse_convs)]

        # SMP 0.5.0: UnetDecoder.forward(features: list[Tensor])
        dec = self.decoder(fused_feats)
        logits = self.segmentation_head(dec)
        if self.safe_nan_to_num:
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        return logits


# =============================================
#   Main
# =============================================

def main(args):
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        except Exception:
            pass
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = len(args.classes) + 1

    model = JointFusionUnet(
        num_classes=num_classes,
        encoder_name=args.encoder_name,
        encoder_weights_rgb=args.encoder_weights,
        encoder_weights_sar=None,
        decoder_attention_type="scse",
        feature_fusion=args.feature_fusion,
        encoder_depth=5,
    )

    log_number_of_parameters(model)

    classes_wt = np.array(
        [3.0, 5.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0],
        dtype=np.float32,
    )
    if classes_wt.shape[0] != num_classes:
        raise ValueError(f"classes_wt length {classes_wt.shape[0]} != num_classes {num_classes}")

    loss_name = str(getattr(args, "loss", "ce")).lower()
    if loss_name in ("wce_dice", "wce+dice", "ce_dice"):
        criterion = source.losses.WeightedCEPlusDice(
            class_weights=classes_wt,
            ce_weight=float(getattr(args, "ce_weight", 1.0)),
            dice_weight=float(getattr(args, "dice_weight", 1.0)),
            dice_smooth=float(getattr(args, "dice_smooth", 1.0)),
            dice_include_background=bool(int(getattr(args, "dice_include_background", 0))),
            device=device,
        )
    else:
        criterion = source.losses.CEWithLogitsLoss(weights=classes_wt, device=device)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # Optimizer - grupowanie parametrów jak w innych skryptach
    target = model.module if hasattr(model, "module") else model
    decay_params = []
    no_decay_params = []
    for name, param in target.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if name.endswith(".bias") or "bn" in lname or "norm" in lname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(args.learning_rate),
    )

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=int(args.lr_patience),
            threshold=float(args.min_delta),
            min_lr=float(args.min_lr),
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.t_max),
            eta_min=float(args.min_lr),
        )

    # W&B
    wandb_run = None
    if getattr(args, "use_wandb", False):
        wandb_config = {
            "seed": int(args.seed),
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "crop_size": int(args.crop_size),
            "learning_rate": float(args.learning_rate),
            "classes": args.classes,
            "data_root": args.data_root,
            "optimizer": "AdamW",
            "scheduler": args.scheduler,
            "lr_patience": int(args.lr_patience),
            "early_patience": int(args.early_patience),
            "min_delta": float(args.min_delta),
            "min_lr": float(args.min_lr),
            "t_max": int(args.t_max),
            "amp": bool(args.amp),
            "grad_clip": float(args.grad_clip) if args.grad_clip is not None else None,
            "model_name": f"JointFusion_Unet_{args.encoder_name}",
            "criterion": criterion.name if hasattr(criterion, "name") else type(criterion).__name__,
            "model_type": "joint_fusion",
            "encoder_name": args.encoder_name,
            "encoder_weights_rgb": args.encoder_weights,
            "encoder_weights_sar": None,
            "feature_fusion": args.feature_fusion,
        }

        run_name = (
            f"JOINTFUSION_Unet_{args.encoder_name}_"
            f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
            f"_lr-{args.learning_rate}_"
            f"feat-{args.feature_fusion}_"
            f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
        )

        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "FUSION-train"),
            entity=getattr(args, "wandb_entity", None),
            config=wandb_config,
            name=run_name,
        )

    print("Number of epochs   :", args.n_epochs)
    print("Number of classes  :", num_classes)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)
    print("AMP                :", args.amp)
    print("Grad clip          :", args.grad_clip)
    print("Encoder name       :", args.encoder_name)
    print("Encoder weights(RGB):", args.encoder_weights)
    print("Feature fusion     :", args.feature_fusion)

    train_model(args, model, optimizer, criterion, device, scheduler, wandb_run=wandb_run)

    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


# =============================================
#   Train loop
# =============================================

def train_model(args, model, optimizer, criterion, device, scheduler=None, wandb_run=None):
    train_loader, valid_loader = build_data_loaders(args, FusionDataset)
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_results, exist_ok=True)

    model_name = (
        f"JOINT_FUSION_Unet_{args.encoder_name}_"
        f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
        f"_lr_{args.learning_rate}_"
        f"feat_{args.feature_fusion}_"
        f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
    )

    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_results, "train_joint_fusion.txt")

    for epoch in range(args.n_epochs):
        logs_train = S.train_epoch_streaming(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
            grad_clip=float(args.grad_clip) if args.grad_clip is not None else None,
        )
        logs_valid = S.valid_epoch_streaming(
            model=model,
            criterion=criterion,
            dataloader=valid_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
        )

        log_epoch_results(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        if wandb_run is not None:
            log_dict: dict[str, float] = {"epoch": float(epoch + 1)}
            try:
                log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
            except Exception:
                pass

            def _log_metrics(prefix: str, logs: dict):
                for k, v in logs.items():
                    if k in ("iou_per_class", "f1_per_class"):
                        arr = np.asarray(v, dtype=float).ravel()
                        for cid, val in enumerate(arr):
                            if k == "iou_per_class":
                                log_dict[f"{prefix}/iou_class_{cid}"] = float(val)
                            elif k == "f1_per_class":
                                log_dict[f"{prefix}/f1_class_{cid}"] = float(val)
                    else:
                        try:
                            log_dict[f"{prefix}/{k}"] = float(v)
                        except Exception:
                            continue

            _log_metrics("train", logs_train)
            _log_metrics("valid", logs_valid)
            wandb_run.log(log_dict)

        score = logs_valid["iou"]
        if scheduler is not None:
            previous_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

            new_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if new_learning_rates != previous_learning_rates:
                print(f"LR changed: {previous_learning_rates} -> {new_learning_rates}")

        improvement = score - (max_score if max_score != -float("inf") else score)
        state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

        best_path = os.path.join(args.save_model, f"{model_name}.pth")
        last_path = os.path.join(args.save_model, f"{model_name}_last.pth")

        if max_score == -float("inf") or improvement > float(args.min_delta):
            max_score = score
            bad_epochs = 0
            os.makedirs(args.save_model, exist_ok=True)
            torch.save(state_dict, best_path)
            print("Model saved:", os.path.abspath(best_path))
        else:
            bad_epochs += 1

        try:
            torch.save(state_dict, last_path)
        except Exception as e:
            print("Warning: failed to save last checkpoint:", e)

        if bad_epochs >= int(args.early_patience):
            print(f"Early stopping: brak poprawy IoU >= {args.min_delta} przez {args.early_patience} epok.")
            break

        if args.scheduler == "plateau":
            min_lr = float(args.min_lr)
            lrs = [g["lr"] for g in optimizer.param_groups]
            if all(lr <= min_lr + 1e-12 for lr in lrs) and improvement <= float(args.min_delta):
                print("LR osiągnął minimum i brak poprawy – zatrzymuję trening.")
                break


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    parser = argparse.ArgumentParser(description="Training Joint Fusion (2 encoders + shared UNet decoder)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--classes", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--data_root", type=str, default="../dataset/train")
    parser.add_argument("--save_model", type=str, default="model/fusion")
    parser.add_argument("--save_results", type=str, default="results")

    # AUGMENTACJE
    parser.add_argument("--train_augm", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--valid_augm", type=int, choices=[1, 2, 3], default=1)

    # ENKODER
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")

    # JOINT FUSION SPECIFIC
    parser.add_argument(
        "--feature_fusion",
        type=str,
        default="concat",
        choices=["concat", "sum", "mean"],
        help="Jak łączyć cechy (skip+bottleneck): concat / sum / mean",
    )

    # SAR normalizacja (dla FusionDataset)
    parser.add_argument(
        "--sar_normalize",
        type=str,
        default="global",
        choices=["global", "per_sample", "none"],
        help="Normalizacja SAR: global (mean/std z datasetu), per_sample, none",
    )

    # SCHEDULER / EARLY STOPPING
    parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--early_patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--t_max", type=int, default=30)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # LOSS
    parser.add_argument("--loss", type=str, default="wce_dice", choices=["ce", "wce_dice"])
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--dice_smooth", type=float, default=1.0)
    parser.add_argument("--dice_include_background", type=int, default=0, choices=[0, 1])

    # Class-aware crop / oversampling
    parser.add_argument("--class_aware_crop", type=int, default=1, choices=[0, 1])
    parser.add_argument("--oversample_class", type=int, default=1)
    parser.add_argument("--oversample_p", type=float, default=0.5)
    parser.add_argument("--oversample_min_pixels", type=int, default=200)
    parser.add_argument("--oversample_max_tries", type=int, default=50)

    # W&B
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="FUSION-train")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="radoslaw-godlewski00-politechnika-warszawska",
    )

    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print("Processing time:", end - start)


import os
import time
import numpy as np
import torch
import source
import segmentation_models_pytorch as smp
import argparse
import warnings
from source import streaming as S
from source.data_loader import build_data_loaders
from source.dataset import FusionDataset
from source.utils import log_epoch_results, log_number_of_parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

warnings.filterwarnings("ignore")

# =============================================
#   Main
# =============================================

def main(args):
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # wymuszamy deterministyczne zachowanie cudnn (może spowolnić trening, ale daje powtarzalność)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ograniczamy zużycie GPU (opcjonalne, ustawienie specyficzne dla Twojego środowiska)
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =============================================
    #   Model i criterion
    # Tworzymy architekturę UNet z parametryzowanym enkoderem.
    # =============================================
    model = smp.Unet(
        classes=len(args.classes) + 1,
        in_channels=4,
        activation=None,
        encoder_weights=None if str(args.encoder_weights).lower() == "none" else args.encoder_weights,
        encoder_name=args.encoder_name,
        decoder_attention_type="scse",
    )

    log_number_of_parameters(model)
    classes_wt = np.array(
        [
            3.0,
            5.0,
            1.0,
            1.0,
            1.5,
            1.0,
            1.0,
            1.5,
            1.0,
        ],
        dtype=np.float32,
    )
    if classes_wt.shape[0] != len(args.classes) + 1:
        raise ValueError(f"classes_wt length {classes_wt.shape[0]} != num_classes {len(args.classes) + 1}")

    loss_name = str(getattr(args, 'loss', 'ce')).lower()
    if loss_name in ("wce_dice", "wce+dice", "ce_dice"):
        criterion = source.losses.WeightedCEPlusDice(
            class_weights=classes_wt,
            ce_weight=float(getattr(args, 'ce_weight', 1.0)),
            dice_weight=float(getattr(args, 'dice_weight', 1.0)),
            dice_smooth=float(getattr(args, 'dice_smooth', 1.0)),
            dice_include_background=bool(int(getattr(args, 'dice_include_background', 0))),
            device=device,
        )
    else:
        criterion = source.losses.CEWithLogitsLoss(weights=classes_wt, device=device)

    # =============================================
    #   Device i DataParallel
    # Przenosimy model na urządzenie (GPU/CPU). Jeśli jest więcej GPU, opakowujemy model
    # w torch.nn.DataParallel aby trenować równolegle na wielu kartach.
    # =============================================
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # =============================================
    #   Optimizer i grupowanie parametrów (AdamW)
    #   Poniżej tworzymy dwie grupy parametrów:
    # - decay_params: parametry podlegające weight decay (zwykle wagi konwolucji/linearnych)
    # - no_decay_params: biasy i parametry normalizacji, które zwykle nie dostają weight decay
    # Dzięki temu regularyzacja (weight decay) działa poprawnie i nie niszczy parametrów takich jak bias/Norm.
    # =============================================
    target = model.module if hasattr(model, "module") else model
    decay_params = []
    no_decay_params = []
    for name, param in target.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if name.endswith('.bias') or 'bn' in lname or 'norm' in lname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(args.learning_rate),
    )

    # =============================================
    #   Scheduler
    # =============================================
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=int(args.lr_patience),
            threshold=float(args.min_delta),
            min_lr=float(args.min_lr),
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.t_max),
            eta_min=float(args.min_lr),
        )


    # =============================================
    #   Weights & Biases (opcjonalnie)
    # Inicjalizujemy run w wandb tylko jeśli użytkownik poda --use_wandb.
    # Do config wrzucamy najważniejsze hiperparametry i opis modelu.
    # =============================================
    wandb_run = None
    if getattr(args, "use_wandb", False):
        wandb_config = {
            "seed": int(args.seed),
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "crop_size": int(args.crop_size),
            "learning_rate": float(args.learning_rate),
            "classes": args.classes,
            "data_root": args.data_root,
            "optimizer": "AdamW",
            "scheduler": args.scheduler,
            "lr_patience": int(args.lr_patience),
            "early_patience": int(args.early_patience),
            "min_delta": float(args.min_delta),
            "min_lr": float(args.min_lr),
            "t_max": int(args.t_max),
            "amp": bool(args.amp),
            "grad_clip": float(args.grad_clip) if args.grad_clip is not None else None,
            "model_name": f"Unet_{args.encoder_name}_fusion",
            "criterion": criterion.name if hasattr(criterion, "name") else type(criterion).__name__,
            "model_type": "fusion",
            "encoder_name": args.encoder_name,
            "encoder_weights": args.encoder_weights,
            "class_weights": classes_wt.tolist(),
            "in_channels": 4,
        }

        run_name = (
            f"FUSION_Unet_{args.encoder_name}_"
            f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
            f"_lr-{args.learning_rate}_"
            f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
        )

        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "FUSION-train"),
            entity=getattr(args, "wandb_entity", None),
            config=wandb_config,
            name=run_name,
        )

    print("Number of epochs   :", args.n_epochs)
    print("Number of classes  :", len(args.classes) + 1)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)
    print("AMP                :", args.amp)
    print("Grad clip          :", args.grad_clip)
    print("Encoder name       :", args.encoder_name)
    print("Encoder weights    :", args.encoder_weights)
    print("Class weights      :", classes_wt)
    # Uruchamiamy pętlę treningową
    train_model(args, model, optimizer, criterion, device, scheduler, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


# =============================================
#   Główna pętla treningowa z early stopping + scheduler
# =============================================

def train_model(args, model, optimizer, criterion, device, scheduler=None, wandb_run=None):
    train_loader, valid_loader = build_data_loaders(args, FusionDataset)
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_results, exist_ok=True)

    model_name = (
        f"FUSION_Unet_{args.encoder_name}_"
        f"{criterion.name if hasattr(criterion, 'name') else type(criterion).__name__}"
        f"_lr_{args.learning_rate}_"
        f"augmT{getattr(args, 'train_augm', 'NA')}V{getattr(args, 'valid_augm', 'NA')}"
    )

    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_results, "train_fusion.txt")

    # Zapis diagnostyki modelu i danych na początek logu
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("Model diagnostics:\n")
            try:
                for name, p in model.named_parameters():
                    if p.dim() == 4 and any(k in name.lower() for k in ["conv_stem", "conv1", "stem", "conv_first"]):
                        f.write(f"First conv param: {name} shape: {tuple(p.shape)}\n")
                        f.write(f"  mean: {float(p.mean().item()):.6f} std: {float(p.std().item()):.6f}\n")
                        w = p.detach().cpu().numpy()
                        if w.ndim == 4:
                            ch_means = w.mean(axis=(0, 2, 3))
                            f.write("  Per-channel means: " + ", ".join(f"{v:.6f}" for v in ch_means) + "\n")
                        break
            except Exception as e:
                f.write(f"Could not compute first conv stats: {e}\n")

            try:
                sar_mean = getattr(train_loader.dataset, "sar_mean", None)
                sar_std = getattr(train_loader.dataset, "sar_std", None)
                if sar_mean is not None:
                    try:
                        f.write(f"SAR dataset mean: {float(sar_mean):.6f}, std: {float(sar_std):.6f}\n")
                    except Exception:
                        f.write(f"SAR dataset mean/std: {sar_mean} / {sar_std}\n")
                else:
                    f.write("SAR dataset mean/std: not available\n")
            except Exception as e:
                f.write(f"Could not write SAR mean/std: {e}\n")

            f.write("\n")
    except Exception:
        pass

    def _get_lr(opt):
        return ", ".join(f"{g['lr']:.6g}" for g in opt.param_groups)

    for epoch in range(args.n_epochs):
        logs_train = S.train_epoch_streaming(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
            grad_clip=float(args.grad_clip) if args.grad_clip is not None else None,
        )
        logs_valid = S.valid_epoch_streaming(
            model=model,
            criterion=criterion,
            dataloader=valid_loader,
            device=device,
            num_classes=num_classes,
            use_amp=bool(args.amp),
        )
        log_epoch_results(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        if wandb_run is not None:
            log_dict = {"epoch": epoch + 1}
            try:
                log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
            except Exception:
                pass

            def _log_metrics(prefix: str, logs: dict):
                for k, v in logs.items():
                    if k in ("iou_per_class", "f1_per_class"):
                        arr = np.asarray(v, dtype=float).ravel()
                        for cid, val in enumerate(arr):
                            if k == "iou_per_class":
                                log_dict[f"{prefix}/iou_class_{cid}"] = float(val)
                            elif k == "f1_per_class":
                                log_dict[f"{prefix}/f1_class_{cid}"] = float(val)
                    else:
                        try:
                            log_dict[f"{prefix}/{k}"] = float(v)
                        except Exception:
                            continue

            _log_metrics("train", logs_train)
            _log_metrics("valid", logs_valid)
            wandb_run.log(log_dict)

        score = logs_valid["iou"]
        if scheduler is not None:
            previous_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

            new_learning_rates = [g["lr"] for g in optimizer.param_groups]
            if new_learning_rates != previous_learning_rates:
                print(f"LR changed: {previous_learning_rates} -> {new_learning_rates}")

        # --- EARLY STOPPING + checkpoint ---
        # Jeśli nastąpi poprawa (większe IoU) zapisujemy checkpoint i resetujemy licznik "złych" epok.
        improvement = score - (max_score if max_score != -float("inf") else score)

        # zawsze zapisujemy poprawny state_dict (bez względu na DataParallel)
        state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

        best_path = os.path.join(args.save_model, f"{model_name}.pth")
        last_path = os.path.join(args.save_model, f"{model_name}_last.pth")

        if max_score == -float("inf") or improvement > float(args.min_delta):
            max_score = score
            bad_epochs = 0
            os.makedirs(args.save_model, exist_ok=True)
            torch.save(state_dict, best_path)
            print("Model saved:", os.path.abspath(best_path))
        else:
            bad_epochs += 1

        # Zapis awaryjny: zawsze zapisuj ostatni checkpoint (łatwiej debugować i nic nie ginie)
        try:
            torch.save(state_dict, last_path)
        except Exception as e:
            print("Warning: failed to save last checkpoint:", e)

        # Early stopping: jeśli przez `early_patience` epok nie ma poprawy, przerywamy trening
        if bad_epochs >= int(args.early_patience):
            print(f"Early stopping: brak poprawy IoU >= {args.min_delta} przez {args.early_patience} epok.")
            break

        # Dodatkowa logika dla scheduler="plateau": jeśli LR osiągnął min i brak poprawy -> stop
        if args.scheduler == "plateau":
            min_lr = float(args.min_lr)
            lrs = [g["lr"] for g in optimizer.param_groups]
            if all(lr <= min_lr + 1e-12 for lr in lrs) and improvement <= float(args.min_delta):
                print("LR osiągnął minimum i brak poprawy – zatrzymuję trening.")
                break


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    parser = argparse.ArgumentParser(description='Training Fusion (RGB+SAR) - streaming, memory-efficient')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', type=str, default="../dataset/train")
    parser.add_argument('--save_model', type=str, default="model/fusion")
    parser.add_argument('--save_results', type=str, default="results")

    # --- AUGMENTACJE ---
    parser.add_argument('--train_augm', type=int, choices=[1, 2, 3], default=1,
                        help='Wybór trybu augmentacji dla treningu (train_augm1/2/3)')
    parser.add_argument('--valid_augm', type=int, choices=[1, 2, 3], default=1,
                        help='Wybór trybu augmentacji dla walidacji (valid_augm1/2/3)')

    # --- ENKODER ---
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b4',
                        help='Nazwa enkodera z segmentation_models_pytorch, np. efficientnet-b4, resnet34, tu-convnext_tiny')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='Wagi enkodera, np. imagenet, ssl, swsl lub none (brak wag)')

    # --- SCHEDULER / EARLY STOPPING PARAMS ---
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine', 'none'], default='plateau')
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--early_patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--t_max', type=int, default=30)
    parser.add_argument('--amp', type=int, default=1, help='Włącz / wyłącz mixed precision (1/0)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maks. norm gradientu; None aby wyłączyć')

    # --- Loss ---
    parser.add_argument('--loss', type=str, default='wce_dice', choices=['ce', 'wce_dice'],
                        help='Funkcja straty: ce (weighted CE) lub wce_dice (weighted CE + Dice)')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Waga składnika CE w loss wce_dice')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Waga składnika Dice w loss wce_dice')
    parser.add_argument('--dice_smooth', type=float, default=1.0, help='Smooth dla DiceLoss')
    parser.add_argument('--dice_include_background', type=int, default=0, choices=[0, 1],
                        help='Czy uwzględniać tło (klasa 0) w Dice (0/1)')

    # --- Class-aware crop / oversampling (pod rzadkie klasy, np. c1) ---
    parser.add_argument('--class_aware_crop', type=int, default=1, choices=[0, 1],
                        help='Włącz class-aware cropping na treningu (0/1)')
    parser.add_argument('--oversample_class', type=int, default=1, help='Wartość klasy w masce, którą preferujemy w crop (np. 1 dla c1)')
    parser.add_argument('--oversample_p', type=float, default=0.5, help='Prawdopodobieństwo użycia class-aware crop dla próbki (np. 0.5)')
    parser.add_argument('--oversample_min_pixels', type=int, default=200, help='Minimalna liczba pikseli target_class w crop')
    parser.add_argument('--oversample_max_tries', type=int, default=50, help='Ile losowych cropów próbować zanim fallback')

    # --- Weights & Biases ---
    parser.add_argument('--use_wandb', type=bool, default=True, help='Włącz logowanie do Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='FUSION-train', help='Nazwa projektu w W&B')
    parser.add_argument('--wandb_entity', type=str, default='radoslaw-godlewski00-politechnika-warszawska',
                        help='Nazwa entity w W&B')

    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)


import torch
from tqdm.auto import tqdm
from typing import Optional

__all__ = [
    "to_index_targets",
    "update_confusion_matrix",
    "metrics_from_cm",
    "train_epoch_streaming",
    "valid_epoch_streaming",
]

def to_index_targets(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 4 and y.size(1) > 1:
        return y.argmax(1)
    return y.squeeze(1)

def update_confusion_matrix(cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int, chunk_size: int = 262144):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    # filtr: zachowujemy tylko poprawne klasy (0..num_classes-1)
    k = (targets >= 0) & (targets < num_classes)
    preds = preds[k]
    targets = targets[k]
    n = preds.numel()
    # przetwarzamy po kawałkach aby nie alokować ogromnych wektorów jednocześnie
    for i in range(0, n, chunk_size):
        p = preds[i:i+chunk_size]
        t = targets[i:i+chunk_size]
        inds = (t.to(torch.int64) * num_classes + p.to(torch.int64))
        cm += torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


# =============================================
#   Metryki z macierzy pomyłek
# =============================================
# Funkcja `metrics_from_cm` wylicza klasyczne metryki segmentacji z macierzy pomyłek:
# - dla każdej klasy oblicza TP, FP, FN
# - IoU, Dice, Precision, Recall, F1
# - accuracy globalna (liczona po wszystkich pikselach)
# Zwraca słownik z uśrednionymi (mean) wartościami między klasami oraz accuracy.

def metrics_from_cm(cm: torch.Tensor):
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    eps = 1e-7
    iou_c = tp / (tp + fp + fn + eps)
    dice_c = (2 * tp) / (2 * tp + fp + fn + eps)
    prec_c = tp / (tp + fp + eps)
    rec_c = tp / (tp + fn + eps)
    f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c + eps)
    acc_global = (tp.sum() / (cm.sum().float() + eps)).item()
    return {
        "iou": iou_c.mean().item(),
        "dice": dice_c.mean().item(),
        "acc": acc_global,
        "prec": prec_c.mean().item(),
        "rec": rec_c.mean().item(),
        "f1": f1_c.mean().item(),
        "iou_per_class": iou_c.tolist(),
        "f1_per_class": f1_c.tolist(),
    }


# =============================================
#   Pętla treningowa (streaming) - per-epoch
# `train_epoch_streaming` wykonuje jedną epokę treningową na danych strumieniowych:
# - ustawia model w tryb train()
# - iteruje po batchach z dataloadera (tqdm dla progress baru)
# - zapewnia mixed precision przez torch.cuda.amp (jeśli use_amp=True)
# - dla każdego batcha: forward, loss, backward (skalowany przez grad scaler), clip grad, optimizer.step
# - zbiera predykcje i aktualizuje macierz pomyłek oraz sumuje straty
# - zwraca słownik metryk oraz średnią straty na batch
# =============================================
def train_epoch_streaming(model, optimizer, criterion, dataloader, device, num_classes: int, use_amp: bool = True, grad_clip: Optional[float] = None):
    model.train()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    n_batches = 0
    n_loss_batches = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"]
        y_idx = to_index_targets(y)
        optimizer.zero_grad(set_to_none=True)

        # Forward w AMP OK, ale loss liczmy w fp32 (stabilniej, szczególnie CE i Dice)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(logits.float(), y.to(device, non_blocking=True).float())

        loss_val = _safe_loss_value(loss)
        if loss_val is None:
            _debug_invalid_loss("train", logits, y, y_idx)
            # pomijamy backward dla zepsutego batcha
            continue

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(1).detach().cpu()
        cm = update_confusion_matrix(cm, preds, y_idx.cpu(), num_classes)
        total_loss += loss_val
        n_batches += 1
        n_loss_batches += 1

    mets = metrics_from_cm(cm)
    mets["loss"] = total_loss / max(n_loss_batches, 1)
    return mets


# =============================================
#   Pętla walidacyjna (streaming) - per-epoch
# =============================================
# `valid_epoch_streaming` wykonuje jedną epokę walidacyjną:
# - ustawia model w tryb eval() i korzysta z torch.inference_mode() dla wydajności
# - nie wykonuje backward/optimizer.step
# - używa mixed precision w podobny sposób jak w treningu
# - aktualizuje macierz pomyłek i sumuje straty (jeśli criterion != None)
# - zwraca słownik metryk oraz średnią straty
def valid_epoch_streaming(model, criterion, dataloader, device, num_classes: int, use_amp: bool = True):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    n_batches = 0
    n_loss_batches = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Valid", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"]
            y_idx = to_index_targets(y)

            # logits mogą być w AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)

            if criterion is not None:
                # loss zawsze w fp32 (stabilnie)
                with torch.cuda.amp.autocast(enabled=False):
                    loss = criterion(logits.float(), y.to(device, non_blocking=True).float())
                loss_val = _safe_loss_value(loss)
                if loss_val is None:
                    _debug_invalid_loss("valid", logits, y, y_idx)
                else:
                    total_loss += loss_val
                    n_loss_batches += 1

            preds = logits.argmax(1).cpu()
            cm = update_confusion_matrix(cm, preds, y_idx.cpu(), num_classes)
            n_batches += 1

    mets = metrics_from_cm(cm)
    if criterion is not None and n_loss_batches > 0:
        mets["loss"] = total_loss / n_loss_batches
    return mets

def _safe_loss_value(loss: torch.Tensor) -> Optional[float]:
    """Zwraca bezpieczną wartość loss jako float lub None jeśli loss jest NaN/Inf."""
    if loss is None:
        return None
    if not torch.is_tensor(loss):
        try:
            loss = torch.as_tensor(loss)
        except Exception:
            return None
    if not torch.isfinite(loss).all():
        return None
    v = float(loss.detach().item())
    # dodatkowe zabezpieczenie przed ekstremami (u Ciebie valid loss potrafił iść w dziesiątki)
    if v > 1000.0:
        return None
    return v


def _debug_invalid_loss(prefix: str, logits: torch.Tensor, y: torch.Tensor, y_idx: torch.Tensor):
    try:
        lg_min = float(logits.detach().min().item())
        lg_max = float(logits.detach().max().item())
    except Exception:
        lg_min, lg_max = None, None
    try:
        yi_min = int(y_idx.detach().min().item())
        yi_max = int(y_idx.detach().max().item())
    except Exception:
        yi_min, yi_max = None, None
    print(
        f"[{prefix}] Invalid loss detected. "
        f"logits[min,max]=({lg_min},{lg_max}) target_idx[min,max]=({yi_min},{yi_max}) "
        f"target_shape={tuple(y.shape)} logits_shape={tuple(logits.shape)}"
    )