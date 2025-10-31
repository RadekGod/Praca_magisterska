import os
import time
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import source
import segmentation_models_pytorch as smp
import argparse
import warnings
from source import streaming as S
from source.utils import append_epoch_log
from source.data_loader import build_data_loaders
warnings.filterwarnings("ignore")

class RGBDataset(source.dataset.Dataset):
    def __getitem__(self, idx):
        # Ładuj obraz RGB zamiast SAR
        img = self.load_multiband(self.fns[idx].replace("labels", "rgb_images"))
        msk = self.load_grayscale(self.fns[idx])
        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)
        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}


def train_model(args, model, optimizer, criterion, metric, device, scheduler=None):
    train_data_loader, val_data_loader = build_data_loaders(args, RGBDataset)
    os.makedirs(args.save_model, exist_ok=True)
    model_name = f"RGB_Pesudo_{model.name}_s{args.seed}_{criterion.name}"
    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_model, "train_rgb.txt")

    def _get_lr(opt):
        return ", ".join(f"{g['lr']:.6g}" for g in opt.param_groups)

    for epoch in range(args.n_epochs):
        print(f"\nEpoch: {epoch + 1} (lr: {_get_lr(optimizer)})")
        # --- trening strumieniowy ---
        logs_train = S.train_epoch_streaming(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=device,
            num_classes=num_classes,
            use_amp=False,
            grad_clip=None,
        )
        # --- walidacja strumieniowa ---
        logs_valid = S.valid_epoch_streaming(
            model=model,
            criterion=criterion,
            dataloader=val_data_loader,
            device=device,
            num_classes=num_classes,
            use_amp=False,
        )
        # --- LOGOWANIE DO KONSOLI ---
        print(f"Train Loss: {logs_train.get('loss'):.6f}, Valid Loss: {logs_valid.get('loss'):.6f}")
        print(f"Train IoU: {logs_train.get('iou'):.6f}, Valid IoU: {logs_valid.get('iou'):.6f}")
        print(f"Train Dice: {logs_train.get('dice'):.6f}, Valid Dice: {logs_valid.get('dice'):.6f}")
        print(f"Train Acc: {logs_train.get('acc'):.6f}, Valid Acc: {logs_valid.get('acc'):.6f}")
        print(f"Train Prec: {logs_train.get('prec'):.6f}, Valid Prec: {logs_valid.get('prec'):.6f}")
        print(f"Train Rec: {logs_train.get('rec'):.6f}, Valid Rec: {logs_valid.get('rec'):.6f}")
        #TODO Dodać jeszcze f score i iou2

        # --- ZAPIS DO PLIKU ---
        append_epoch_log(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        # --- LR scheduler ---
        score = logs_valid["iou"]
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()

        # --- EARLY STOPPING + checkpoint ---
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


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # UNet z EfficientNet-B4, in_channels=3 dla RGB
    model = smp.Unet(
        classes=len(args.classes)+1,
        in_channels=3,
        activation=None,
        encoder_weights="imagenet",
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)
    classes_wt = np.ones([len(args.classes)+1], dtype=np.float32)
    criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)
    metric = source.metrics.IoU2()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(
            [dict(params=model.module.parameters(), lr=args.learning_rate)]
        )
    # Upewnij się, że model jest na tym samym urządzeniu co dane
    model = model.to(device)

    # --- SCHEDULER ---
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

    print("Number of epochs   :", args.n_epochs)
    print("Number of classes  :", len(args.classes)+1)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)
    train_model(args, model, optimizer, criterion, metric, device, scheduler)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    parser = argparse.ArgumentParser(description='Model Training RGB')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--n_epochs', default=50)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--crop_size', default=256)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_root', default="../dataset/train")
    parser.add_argument('--save_model', default="model")
    parser.add_argument('--save_results', default="results")
    # --- SCHEDULER / EARLY STOPPING PARAMS ---
    parser.add_argument('--scheduler', choices=['plateau', 'cosine', 'none'], default='plateau')
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--early_patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.005)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--t_max', type=int, default=20)
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
