import numpy as np
import os

"""Moduł pomocniczy `utils`.

Zawiera funkcje pomocnicze używane przy trenowaniu i ewaluacji modeli:

- log_epoch_results(log_path, epoch_idx, lr_str, train_logs, valid_logs)
  Loguje metryki epoki (do pliku oraz na stdout). Obsługuje także metryki per-class jeśli są dostępne.

- _get_lr(optimizer)
  Zwraca sformatowany łańcuch z wartościami learning rate grup parametrów optymalizera.

- log_number_of_parameters(model)
  Wypisuje liczbę trenowalnych parametrów modelu.

Przykład użycia:
    from source.utils import log_epoch_results

Ten plik nie modyfikuje zachowania modeli — dostarcza tylko narzędzia pomocnicze do logowania
i diagnostyki podczas treningu.
"""

def log_epoch_results(log_path: str, epoch_idx: int, lr_str: str, train_logs: dict, valid_logs: dict):
    """Zapisuje i wypisuje metryki dla danej epoki treningowej.

    Funkcja dopisuje do pliku tekstowego (tworzy katalog jeśli potrzeba) podsumowanie metryk
    dla zbioru treningowego i walidacyjnego oraz wypisuje te same informacje na standardowe wyjście.

    Argumenty:
        log_path (str): Ścieżka do pliku logu (plik zostanie dopisany).
        epoch_idx (int): Numer epoki (liczba całkowita).
        lr_str (str): Reprezentacja learning rate (np. zwrócona przez `_get_lr`).
        train_logs (dict): Słownik metryk treningowych. Oczekiwane klucze: 'loss','iou','dice','acc','prec','rec','f1',
                           opcjonalnie 'iou_per_class' i 'f1_per_class' jako listy/ndarray.
        valid_logs (dict): Słownik metryk walidacyjnych (te same klucze co wyżej).

    Zachowanie i uwagi:
        - Brakujące metryki są traktowane jako NaN.
        - Metryki per-class (jeśli dostępne) są zapisywane w kolejnych wierszach.
        - Funkcja nie zwraca wartości; generuje wyjście do pliku i na stdout.
    """
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
    """Zwraca sformatowany string z wartości learning rate dla każdej grupy parametrów.

    Argumenty:
        optimizer: obiekt optymalizera (np. torch.optim.Optimizer).

    Zwraca:
        str: wartości lr oddzielone przecinkami.
    """
    return ", ".join(f"{g['lr']:.6g}" for g in optimizer.param_groups)


def log_number_of_parameters(model):
    """Wypisuje liczbę trenowalnych parametrów modelu.

    Argumenty:
        model: obiekt modelu udostępniający `parameters()` (np. moduł PyTorch).
    """
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)
