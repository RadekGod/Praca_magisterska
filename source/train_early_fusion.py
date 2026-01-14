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
