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
warnings.filterwarnings("ignore")

# =============================================
#   Dataset dla obrazów SAR (1 kanał)
# =============================================
class SARDataset(source.dataset.Dataset):
    def __getitem__(self, idx):
        # Ładowanie kanału SAR (grayscale) + maska
        img = self.load_grayscale(self.fns[idx].replace("labels", "sar_images"))  # [H,W]
        msk = self.load_grayscale(self.fns[idx])
        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)  # image -> [1,H,W]
        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

# =============================================
#   DataLoader (strumieniowe, oszczędne pamięciowo)
# =============================================

def data_loader(args):
    img_pths = [f for f in Path(args.data_root).rglob("*.tif") if "labels" in f.parts]
    random.shuffle(img_pths)
    split_idx = int(0.9 * len(img_pths))
    train_pths = [str(f) for f in img_pths[:split_idx]]
    val_pths = [str(f) for f in img_pths[split_idx:]]

    print("Total samples      :", len(img_pths))
    print("Training samples   :", len(train_pths))
    print("Validation samples :", len(val_pths))

    trainset = SARDataset(train_pths, classes=args.classes, size=args.crop_size, train=True)
    validset = SARDataset(val_pths, classes=args.classes, train=False)

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(int(args.num_workers), 2),  # ogranicz dla Windows
        pin_memory=False,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=1,  # oszczędność RAM
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, valid_loader

# =============================================
#   Główna pętla treningowa z early stopping + scheduler
# =============================================

def train_model(args, model, optimizer, criterion, device, scheduler=None):
    train_loader, valid_loader = data_loader(args)
    os.makedirs(args.save_model, exist_ok=True)
    model_name = f"SAR_EfficientNetB4_s{args.seed}_{criterion.name}"
    max_score = -float("inf")
    bad_epochs = 0
    num_classes = len(args.classes) + 1

    log_path = os.path.join(args.save_model, "train_sar.txt")

    def _get_lr(opt):
        return ", ".join(f"{g['lr']:.6g}" for g in opt.param_groups)

    for epoch in range(args.n_epochs):
        print(f"\nEpoch: {epoch + 1} (lr: {_get_lr(optimizer)})")
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
        print(f"Train Loss: {logs_train.get('loss'):.6f}, Valid Loss: {logs_valid.get('loss'):.6f}")
        print(f"Train IoU: {logs_train.get('iou'):.6f}, Valid IoU: {logs_valid.get('iou'):.6f}")
        print(f"Train Dice: {logs_train.get('dice'):.6f}, Valid Dice: {logs_valid.get('dice'):.6f}")
        print(f"Train Acc: {logs_train.get('acc'):.6f}, Valid Acc: {logs_valid.get('acc'):.6f}")
        print(f"Train Prec: {logs_train.get('prec'):.6f}, Valid Prec: {logs_valid.get('prec'):.6f}")
        print(f"Train Rec: {logs_train.get('rec'):.6f}, Valid Rec: {logs_valid.get('rec'):.6f}")

        # Zapis do pliku .txt (wspólna funkcja)
        append_epoch_log(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        score = logs_valid["iou"]
        if scheduler is not None:
            if args.scheduler == "plateau":
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
        classes=len(args.classes)+1,
        in_channels=1,
        activation=None,
        encoder_weights="imagenet",  # zostanie dostosowane do 1 kanału przez SMP
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", params)

    classes_wt = np.ones([len(args.classes)+1], dtype=np.float32)
    criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam([dict(params=model.module.parameters(), lr=float(args.learning_rate))])

    model = model.to(device)

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
    print("AMP                :", args.amp)
    print("Grad clip          :", args.grad_clip)

    train_model(args, model, optimizer, criterion, device, scheduler)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    parser = argparse.ArgumentParser(description='Training SAR (streaming, memory-efficient)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--classes', default=[1,2,3,4,5,6,7,8])
    parser.add_argument('--data_root', default='dataset/train')
    parser.add_argument('--save_model', default='model')
    parser.add_argument('--scheduler', choices=['plateau','cosine','none'], default='plateau')
    parser.add_argument('--lr_patience', type=int, default=4)
    parser.add_argument('--early_patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.003)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--t_max', type=int, default=30)
    parser.add_argument('--amp', type=int, default=1, help='Włącz / wyłącz mixed precision (1/0)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maks. norm gradientu; None aby wyłączyć')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
