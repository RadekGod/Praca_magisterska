import argparse
import os
import time
import warnings

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    # Tworzymy architekturę UNet z enkoderem EfficientNet-B4. in_channels=3 bo to skrypt dla RGB.
    # encoder_weights="imagenet" używa pretrenowanych wag dla enkodera.
    # =============================================
    model = smp.Unet(
        classes=len(args.classes)+1,
        in_channels=3,
        activation=None,
        encoder_weights="imagenet",
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )
    log_number_of_parameters(model)
    # Przygotowanie wag klas dla funkcji strat (CrossEntropy z wagami)
    classes_wt = np.ones([len(args.classes)+1], dtype=np.float32)
    criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)

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
    # Uruchamiamy pętlę treningową
    train_model(args, model, optimizer, criterion, device, scheduler)

# =============================================
#   Główna pętla treningowa z early stopping + scheduler
# Funkcja `train_model` odpowiada za cały przebieg treningu po epokach:
# - tworzy dataloadery przez `build_data_loaders`
# - dla każdej epoki: trening (streaming), walidacja (streaming), logowanie wyników
# - aktualizuje scheduler (ReduceLROnPlateau z metryką lub Cosine bez metryki)
# - realizuje logiczkę checkpointów (zapisywanie najlepszych wag) i early stopping
# =============================================
def train_model(args, model, optimizer, criterion, device, scheduler=None):
    train_data_loader, val_data_loader = build_data_loaders(args, RGBDataset)
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_results, exist_ok=True)
    model_name = f"RGB_Pesudo_{model.name}_s{args.seed}_{criterion.name}"
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
        #   Logowanie wyników epoki
        # Zapis do pliku i na konsolę: learning rate, metryki train/valid (IoU, loss itd.)
        # =============================================
        log_epoch_results(log_path, epoch + 1, _get_lr(optimizer), logs_train, logs_valid)

        # --- LR scheduler ---
        score = logs_valid["iou"]
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        # --- EARLY STOPPING + checkpoint ---
        # Jeśli nastąpi poprawa (większe IoU) zapisujemy checkpoint i resetujemy licznik "złych" epok.
        improvement = score - (max_score if max_score != -float("inf") else score)
        if max_score == -float("inf") or improvement > float(args.min_delta):
            max_score = score
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(args.save_model, f"{model_name}.pth"))
            print("Model saved in the folder : ", args.save_model)
            print("Model name is : ", model_name)
        else:
            bad_epochs += 1

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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    parser = argparse.ArgumentParser(description='Model Training RGB')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--n_epochs', default=3)
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
    parser.add_argument('--t_max', type=int, default=30)
    parser.add_argument('--amp', type=int, default=1, help='Włącz / wyłącz mixed precision (1/0)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maks. norm gradientu; None aby wyłączyć')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
