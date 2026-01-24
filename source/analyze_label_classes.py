"""Analiza rozkładu klas w maskach etykiet (.tif).

Skrypt przetwarza wszystkie pliki z maskami w zadanym katalogu i liczy
wystąpienia wartości pikseli (klas). Wynikiem jest tabela z liczbą pikseli
na klasę oraz udziałem procentowym względem wszystkich pikseli.

Konwencje:
- Maski powinny być 2D (H, W) z integer-ami reprezentującymi klasy (np. 0..C-1).
- Skrypt używa PIL do wczytywania obrazów; jeżeli obraz ma tryb inny niż
  "L" lub "I", zostanie przekonwertowany do 8-bitowego trybu szarości "L".
- Funkcja `update_class_counts` oczekuje maski jako numpy.ndarray; wartości
  są zliczane za pomocą np.unique.

Przykład użycia:
  python analyze_label_classes.py --labels-dir dataset/train/labels --ext .tif

Zwracane informacje:
- wypisywana jest liczba plików, liczba błędów wczytywania, a następnie
  tabela: klasa | piksele | udział [%].

"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń i zwraca Namespace.

    Opcje:
    - --labels-dir: katalog zawierający pliki etykiet (domyślnie ../dataset/train/labels)
    - --ext: rozszerzenie plików (np. .tif)

    Zwraca:
    - argparse.Namespace z atrybutami labels_dir i ext.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analiza rozkładu klas w maskach etykiet. "
            "Przechodzi po wszystkich plikach .tif w podanym katalogu, "
            "zlicza wartości pikseli (klasy) i wyświetla ich liczność oraz udział procentowy."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="../dataset/train/labels",
        help=(
            "Ścieżka do katalogu z maskami etykiet. "
            "Domyślnie ../dataset/train/labels względem katalogu source."
        ),
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".tif",
        help="Rozszerzenie plików etykiet (np. .tif). Domyślnie .tif.",
    )
    return parser.parse_args()


def find_label_files(labels_dir: Path, ext: str) -> List[Path]:
    """Zwraca posortowaną listę plików etykiet o zadanym rozszerzeniu.

    Szczegóły:
    - Nie przeszukujemy rekurencyjnie; oczekujemy jednego katalogu zawierającego
      pliki z etykietami (np. ~4000 plików). Funkcja filtruje po rozszereniu
      (case-insensitive) i zwraca posortowaną listę Path.

    Parametry:
    - labels_dir: Path do katalogu z plikami etykiet.
    - ext: rozszerzenie plików, z lub bez kropki (np. '.tif' lub 'tif').

    Zwraca:
    - list[Path]: posortowana lista plików pasujących do wzorca.

    Zachowanie w przypadku błędów:
    - Jeżeli katalog nie istnieje lub nie jest katalogiem, funkcja wypisze komunikat
      i zwróci pustą listę.
    """

    if not labels_dir.exists() or not labels_dir.is_dir():
        print(f"[BŁĄD] Katalog z etykietami nie istnieje lub nie jest katalogiem: {labels_dir}")
        return []

    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext

    files = [
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ext
    ]
    files.sort()
    return files


def load_mask_pil(path: Path) -> np.ndarray:
    """Wczytuje maskę jako obraz 2D (H, W) z zachowaniem wartości pikseli.

    Szczegóły:
    - Używamy PIL i konwertujemy do trybu "L" (8-bit szarości), jeśli to konieczne.
    - Jeżeli obraz ma więcej kanałów (np. RGB), bierzemy pierwszy kanał po konwersji.
    - Funkcja zwraca numpy.ndarray z typem odpowiadającym wartościom pikseli (np. uint8).

    Parametry:
    - path: Path do pliku z maską.

    Zwraca:
    - np.ndarray o wymiarach (H, W) z wartościami klas jako integer.

    Uwaga:
    - Funkcja zakłada, że klasy mieszczą się w zakresie typów obsługiwanych przez PIL
      (zwykle 0..255 dla trybu 'L'). Jeśli potrzebujesz innego zakresu, rozważ inny loader.
    """

    with Image.open(path) as img:
        if img.mode not in ("L", "I"):
            img = img.convert("L")
        mask = np.array(img)
    # Upewniamy się, że mamy 2D (bez kanału koloru)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def update_class_counts(counts: Dict[int, int], mask: np.ndarray) -> None:
    """Aktualizuje słownik `counts` o zliczenia wartości pikseli w `mask`.

    Parametry:
    - counts: słownik mapujący id_klasy -> liczba_pikseli; jest modyfikowany in-place.
    - mask: numpy.ndarray 2D z wartościami klas (integer).

    Szczegóły implementacji:
    - Używa np.unique(mask, return_counts=True) do szybkiego zliczania unikatów.
    - Pozwala skalować do dużych zestawów plików bez iteracji po pikselach w Pythonie.
    """

    values, pix_counts = np.unique(mask, return_counts=True)
    for v, c in zip(values, pix_counts):
        v_int = int(v)
        counts[v_int] = counts.get(v_int, 0) + int(c)


def main() -> None:
    """Główna funkcja skryptu: parsuje argumenty, zlicza klasy i drukuje podsumowanie.

    Efekt działania:
    - Wczytuje listę plików z katalogu etykiet, iteruje po nich, zliczając wartości pikseli.
    - Na końcu drukuje tabelę z liczbą pikseli na klasę i udziałem percentowym.

    Obsługa błędów:
    - Problemy z wczytaniem pojedynczego pliku są logowane jako ostrzeżenia i nie przerywają
      działania skryptu (plik jest pomijany).
    """
    args = parse_args()

    labels_dir = Path(args.labels_dir).resolve()
    print(f"Katalog z etykietami: {labels_dir}")
    files = find_label_files(labels_dir, args.ext)

    if not files:
        print("[INFO] Nie znaleziono żadnych plików etykiet o zadanym rozszerzeniu.")
        return

    print(f"Liczba znalezionych plików: {len(files)}")

    class_counts: Dict[int, int] = {}
    num_errors = 0

    for idx, path in enumerate(files, start=1):
        try:
            mask = load_mask_pil(path)
        except Exception as e:  # noqa: BLE001
            num_errors += 1
            print(f"[OSTRZEŻENIE] Problem z wczytaniem pliku {path.name}: {e}")
            continue

        update_class_counts(class_counts, mask)

        if idx % 500 == 0:
            print(f"Przetworzono {idx} / {len(files)} plików...")

    if not class_counts:
        print("[INFO] Nie udało się zliczyć żadnych pikseli (być może wszystkie pliki były błędne).")
        return

    total_pixels = sum(class_counts.values())
    if total_pixels == 0:
        print("[INFO] Łączna liczba pikseli wynosi 0.")
        return

    print("\n=== PODSUMOWANIE KLAS ===")
    print(f"Łączna liczba plików: {len(files)}")
    print(f"Liczba plików z błędami: {num_errors}")
    print(f"Łączna liczba pikseli (wszystkie klasy): {total_pixels}")
    print()

    print(f"{'Klasa':>10} | {'Piksele':>15} | {'Udział [%]':>10}")
    print("-" * 45)

    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_pixels) * 100.0
        print(f"{class_id:>10d} | {count:>15d} | {percentage:>9.4f}")

    print("-" * 45)
    print(f"{'SUMA':>10} | {total_pixels:>15d} | {100.0:>9.4f}")


if __name__ == "__main__":
    main()
