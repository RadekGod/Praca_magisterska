from pathlib import Path
import random
from torch.utils.data import DataLoader
from . import transforms as T


# =============================================
#   Build Data Loaders
# Funkcja zwraca dwa DataLoadery: train oraz validation.
# =============================================
def build_data_loaders(args, DatasetClass):
    """Buduje i zwraca `DataLoader` dla treningu oraz walidacji.

    Funkcja:
      1) skanuje `args.data_root` w poszukiwaniu plików etykiet `*.tif` znajdujących się w katalogu `labels`,
      2) losowo miesza próbki i dzieli je w proporcji 90%/10% (train/val),
      3) (opcjonalnie) liczy globalne statystyki normalizacji dla danych SAR na zbiorze treningowym,
      4) tworzy dwa obiekty `DatasetClass` (train i valid) i opakowuje je w `torch.utils.data.DataLoader`.

    Normalizacja SAR (o co chodzi w `sar_mean/sar_std`):
      - Dla danych SAR często nie ma „standardowych” wartości normalizacji (jak np. ImageNet dla RGB), a zakres
        intensywności może się mocno różnić między scenami. Dlatego (gdy `sar_normalize='global'`) liczymy na train
        globalne `mean/std` i przekazujemy je do datasetu.
      - Dataset/transformy mogą wtedy wykonać standaryzację (z-score) w stylu: `x_norm = (x - mean) / (std + eps)`.
        To stabilizuje skalę wejścia i zwykle ułatwia optymalizację.
      - Statystyki są liczone wyłącznie na zbiorze treningowym (żeby uniknąć data leakage) i używane też dla walidacji.
      - Dla RGB często stosuje się stałe schematy (np. skala [0,1] lub mean/std z ImageNet), więc tu nie liczymy
        globalnych statystyk RGB.

    Ważne założenia:
      - Funkcja zakłada, że „próbką” jest plik etykiety (maski) `*.tif` w katalogu `labels`.
        Konkretna implementacja `DatasetClass` zwykle na podstawie ścieżki do etykiety odnajduje odpowiadające
        obrazy (np. RGB/SAR) w sąsiednich katalogach.
      - `DatasetClass` powinien akceptować co najmniej argumenty przekazywane poniżej.

    Args:
        args: Obiekt konfiguracji (najczęściej `argparse.Namespace`) z polami:

            - data_root (str | Path): katalog główny datasetu.
            - classes (Sequence[int] | Sequence[str]): lista klas używana przez dataset (konwencja zależy od implementacji).
            - batch_size (int): batch size dla treningu.
            - num_workers (int): liczba workerów dla `DataLoader` (tu ograniczana do max 2).

            Pola opcjonalne (odczytywane przez `getattr`):

            - crop_size (int | None): rozmiar wycinka/cropa przekazywany do datasetu jako `size`.
            - sar_normalize (str): tryb normalizacji SAR; domyślnie `'global'`.
              Jeśli `'global'` i dataset jest SAR/Fusion, liczone są statystyki (mean/std) na train.
            - train_augm: identyfikator/konfiguracja augmentacji treningowej (format zależy od `DatasetClass`).
            - valid_augm: identyfikator/konfiguracja augmentacji walidacyjnej.

            Opcjonalne pola związane z class-aware crop / oversamplingiem (zastosowane tylko w train):

            - class_aware_crop (int|bool): czy stosować losowanie cropów preferujące daną klasę.
            - oversample_class (int): id klasy, którą chcemy nadpróbkować.
            - oversample_p (float): prawdopodobieństwo uruchomienia logiki oversamplingu dla próbki/cropa.
            - oversample_min_pixels (int): minimalna liczba pikseli danej klasy, by uznać crop za „trafiony”.
            - oversample_max_tries (int): maksymalna liczba prób losowania cropa, zanim zaakceptujemy wynik.

    Side effects:
        Wypisuje na stdout liczność zbioru całkowitego, treningowego i walidacyjnego.

    Notes:
        - Podział train/val jest losowy przy każdym uruchomieniu, o ile nie ustawisz wcześniej ziarna
          generatora losowego (np. `random.seed(...)`).
        - Statystyki SAR są liczone tylko, jeśli `DatasetClass.__name__` to `"SARDataset"` albo `"FusionDataset"`
          oraz `args.sar_normalize == 'global'`.
    """

    # =============================================
    #   Scan i split dataset
    # =============================================
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

    # --- class-aware crop / oversampling (tylko train) ---
    class_aware_crop = bool(getattr(args, 'class_aware_crop', 0))
    oversample_class = int(getattr(args, 'oversample_class', 1))
    oversample_p = float(getattr(args, 'oversample_p', 0.0))
    oversample_min_pixels = int(getattr(args, 'oversample_min_pixels', 20))
    oversample_max_tries = int(getattr(args, 'oversample_max_tries', 30))

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
        class_aware_crop=class_aware_crop,
        oversample_class=oversample_class,
        oversample_p=oversample_p,
        oversample_min_pixels=oversample_min_pixels,
        oversample_max_tries=oversample_max_tries,
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
        class_aware_crop=False,
        oversample_class=oversample_class,
        oversample_p=0.0,
        oversample_min_pixels=oversample_min_pixels,
        oversample_max_tries=oversample_max_tries,
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
