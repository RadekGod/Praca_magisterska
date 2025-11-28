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
        encoder_weights=None,  # zostanie dostosowane do 1 kanału przez SMP
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )
    log_number_of_parameters(model)

    classes_wt = np.ones([len(args.classes) + 1], dtype=np.float32)
    criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)

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
            verbose=True,
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
            "model_name": "Unet_efficientnet-b4_sar",
            "criterion": criterion.name if hasattr(criterion, "name") else type(criterion).__name__,
            "model_type": "sar",
        }
        wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "SAR-train"),
            entity=getattr(args, "wandb_entity", None),
            config=wandb_config,
            name=f"sar_seed:{args.seed}_lr:{args.learning_rate}_no-weights",
        )

        print("Number of epochs   :", args.n_epochs)
        print("Number of classes  :", len(args.classes) + 1)
        print("Batch size         :", args.batch_size)
        print("Device             :", device)
        print("AMP                :", args.amp)
        print("Grad clip          :", args.grad_clip)

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
    model_name = f"SAR_EfficientNetB4_s{args.seed}_{criterion.name}"
    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_results, "train_sar.txt")

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
            for k, v in logs_train.items():
                log_dict[f"train/{k}"] = float(v)
            for k, v in logs_valid.items():
                log_dict[f"valid/{k}"] = float(v)
            wandb_run.log(log_dict)

        score = logs_valid["iou"]
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        improvement = score - (max_score if max_score != -float("inf") else score)
        if max_score == -float("inf") or improvement > float(args.min_delta):
            max_score = score
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(args.save_model, f"{model_name}.pth"))
            print("Model saved in the folder : ", args.save_model)
            print("Model name is : ", model_name)
        else:
            bad_epochs += 1

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
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', type=str, default='../dataset/train')
    parser.add_argument('--save_model', type=str, default='model')
    parser.add_argument('--save_results', type=str, default="results")

    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine', 'none'], default='plateau')
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--early_patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.005)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--t_max', type=int, default=30)
    parser.add_argument('--amp', type=int, default=1, help='Włącz / wyłącz mixed precision (1/0)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maks. norm gradientu; None aby wyłączyć')
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
