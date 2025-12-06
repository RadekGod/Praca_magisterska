from pathlib import Path
import random
from torch.utils.data import DataLoader
from . import transforms as T


# =============================================
#   Build Data Loaders
# Funkcja zwraca dwa DataLoadery: train oraz validation.
# =============================================
def build_data_loaders(args, DatasetClass):
    """Zwraca train_loader i valid_loader dla podanej klasy DatasetClass.
    DatasetClass powinien akceptować (fns, classes=..., size=..., train=..., sar_mean=..., sar_std=..., sar_normalize=..., train_augm=..., valid_augm=...)
    """

    # =============================================
    #   Scan i split dataset
    # =============================================
    # 1) Znajdź wszystkie pliki .tif w katalogu args.data_root, które znajdują się w katalogu "labels".
    # 2) Pomieszaj listę (random.shuffle) i podziel w proporcji 90%/10% na zbiór treningowy i walidacyjny.
    image_paths = [f for f in Path(args.data_root).rglob("*.tif") if "labels" in f.parts]
    random.shuffle(image_paths)
    split_index = int(0.9 * len(image_paths))
    train_paths = image_paths[:split_index]
    validate_paths = image_paths[split_index:]
    train_paths = [str(f) for f in train_paths]
    validate_paths = [str(f) for f in validate_paths]

    print("Total samples      :", len(image_paths))
    print("Training samples   :", len(train_paths))
    print("Validation samples :", len(validate_paths))

    # =============================================
    #   Obliczanie uśrednionych statystyk SAR
    # =============================================

    sar_mean, sar_std = (None, None)
    sar_normalize = getattr(args, 'sar_normalize', 'global')

    needs_sar = DatasetClass.__name__ in ("SARDataset", "FusionDataset")

    if needs_sar and sar_normalize == 'global':
        try:
            sar_mean, sar_std = T.compute_sar_stats(train_paths, load_fn=None)
        except Exception as e:
            print("Warning: failed to compute SAR stats in build_data_loaders:", e)

    # Tryby augmentacji przekazywane z args; mogą nie istnieć w starszych skryptach, więc używamy getattr.
    train_augm = getattr(args, 'train_augm')
    valid_augm = getattr(args, 'valid_augm')

    # =============================================
    #   Tworzenie datasetów i loaderów
    # =============================================
    trainset = DatasetClass(
        train_paths,
        classes=args.classes,
        size=getattr(args, 'crop_size', None),
        train=True,
        sar_mean=sar_mean,
        sar_std=sar_std,
        sar_normalize=sar_normalize,
        train_augm=train_augm,
        valid_augm=valid_augm,
    )
    validset = DatasetClass(
        validate_paths,
        classes=args.classes,
        train=False,
        sar_mean=sar_mean,
        sar_std=sar_std,
        sar_normalize=sar_normalize,
        train_augm=train_augm,
        valid_augm=valid_augm,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(int(args.num_workers), 2),
        pin_memory=False,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, valid_loader
