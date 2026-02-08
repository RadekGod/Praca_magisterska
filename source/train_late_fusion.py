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
            "model_name": f"Late_Fusion_Unet_{args.encoder_name}",
            "criterion": criterion.name if hasattr(criterion, "name") else type(criterion).__name__,
            "model_type": "late_fusion",
            "encoder_name": args.encoder_name,
            "encoder_weights_rgb": args.encoder_weights,
            "encoder_weights_sar": None,
            "class_weights": classes_wt.tolist(),
            "fusion_mode": args.fusion_mode,
        }

        run_name = (
            f"LATE_FUSION_Unet_Batch_16_{args.encoder_name}_"
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
        f"LATE_FUSION_Unet_Batch_16_{args.encoder_name}_"
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
    parser.add_argument("--batch_size", type=int, default=16)
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
