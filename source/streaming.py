"""Moduł `streaming` — narzędzia do trenowania/ewaluacji w trybie strumieniowym.

Zawiera pomocnicze funkcje i pętle epokowe zoptymalizowane pod względem pamięci:
- Konwersje targetów (one-hot -> indeks klasy)
- Aktualizacja macierzy pomyłek (z przetwarzaniem po kawałkach, by nie alokować dużych wektorów)
- Wyliczanie metryk segmentacyjnych z macierzy pomyłek (IoU, Dice, Precision, Recall, F1, accuracy)
- Pętle `train_epoch_streaming` i `valid_epoch_streaming` wspierające mixed precision (AMP)
  oraz bezpieczne liczenie strat (z pomijaniem batchy z NaN/Inf)

Konwencje i założenia:
- Wejściowe tensory `y` mogą być one-hot (B, C, H, W) lub indeksowe (B, 1, H, W) —
  funkcja `to_index_targets` konwertuje do indeksów klas.
- `update_confusion_matrix` zakłada, że `preds` i `targets` to tensory z indeksami klas (int),
  i że klasy należą do zakresu [0, num_classes-1]. Niepoprawne klasy są filtrowane.
- Pętle epokowe zwracają słownik metryk (iou, dice, prec, rec, f1, acc, iou_per_class, f1_per_class)
  oraz (opcjonalnie) średnią stratę pod kluczem 'loss'.

Ten moduł skupia się na stabilności i pamięciooszczędności podczas trenowania modeli segmentacyjnych.
"""

import torch
from tqdm.auto import tqdm
from typing import Optional

__all__ = [
    "to_index_targets",
    "update_confusion_matrix",
    "metrics_from_cm",
    "train_epoch_streaming",
    "valid_epoch_streaming",
]


def to_index_targets(y: torch.Tensor) -> torch.Tensor:
    """Konwertuje tensor targetów do indeksów klas (shape BxHxW lub HxW zależnie od kontekstu).

    Obsługuje przypadki:
      - one-hot: tensor 4D (B, C, H, W) -> argmax po kanale -> (B, H, W)
      - single-channel: tensor 3D/4D z kanałem 1 -> squeeze do (B, H, W) lub (H, W)

    Zwraca tensor indeksów typu całkowitego (bez kanału klasyfikacyjnego).
    """
    if y.ndim == 4 and y.size(1) > 1:
        return y.argmax(1)
    return y.squeeze(1)


def update_confusion_matrix(cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int, chunk_size: int = 262144):
    """Aktualizuje macierz pomyłek (confusion matrix) na podstawie predykcji i targetów.

    Funkcja:
      - spłaszcza predykcje i targety do wektora
      - filtruje niepoprawne targety (poza zakresem [0, num_classes-1])
      - przetwarza dane po kawałkach (chunk_size) aby uniknąć jednorazowych dużych alokacji
      - aktualizuje przekazany tensor `cm` (modyfikacja in-place)

    Argumenty:
      cm: tensor kształtu (num_classes, num_classes) typu całkowitego
      preds: tensor przewidywań (indeksy klas)
      targets: tensor prawdziwych etykiet (indeksy klas)
      num_classes: liczba klas
      chunk_size: maksymalna liczba elementów przetwarzana naraz (domyślnie ~256k)

    Zwraca:
      Zaktualizowany tensor `cm` (ten sam obiekt, na którym operowano).
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    # filtr: zachowujemy tylko poprawne klasy (0..num_classes-1)
    k = (targets >= 0) & (targets < num_classes)
    preds = preds[k]
    targets = targets[k]
    n = preds.numel()
    # przetwarzamy po kawałkach aby nie alokować ogromnych wektorów jednocześnie
    for i in range(0, n, chunk_size):
        p = preds[i:i+chunk_size]
        t = targets[i:i+chunk_size]
        inds = (t.to(torch.int64) * num_classes + p.to(torch.int64))
        cm += torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


# =============================================
#   Metryki z macierzy pomyłek
# =============================================
# Funkcja `metrics_from_cm` wylicza klasyczne metryki segmentacji z macierzy pomyłek:
# - dla każdej klasy oblicza TP, FP, FN
# - IoU, Dice, Precision, Recall, F1
# - accuracy globalna (liczona po wszystkich pikselach)
# Zwraca słownik z uśrednionymi (mean) wartościami między klasami oraz accuracy.

def metrics_from_cm(cm: torch.Tensor):
    """Oblicza metryki segmentacyjne z macierzy pomyłek.

    Argumenty:
      cm: macierz pomyłek (num_classes x num_classes)

    Zwraca słownik z kluczami:
      - 'iou','dice','acc','prec','rec','f1' : uśrednione (mean) wartości po klasach (float)
      - 'iou_per_class','f1_per_class': listy wartości per-klasa

    Uwaga:
      Funkcja dodaje małe eps dla stabilności numerycznej.
    """
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    eps = 1e-7
    iou_c = tp / (tp + fp + fn + eps)
    dice_c = (2 * tp) / (2 * tp + fp + fn + eps)
    prec_c = tp / (tp + fp + eps)
    rec_c = tp / (tp + fn + eps)
    f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c + eps)
    acc_global = (tp.sum() / (cm.sum().float() + eps)).item()
    return {
        "iou": iou_c.mean().item(),
        "dice": dice_c.mean().item(),
        "acc": acc_global,
        "prec": prec_c.mean().item(),
        "rec": rec_c.mean().item(),
        "f1": f1_c.mean().item(),
        "iou_per_class": iou_c.tolist(),
        "f1_per_class": f1_c.tolist(),
    }



# =============================================
#   Pętla treningowa (streaming) - per-epoch
# `train_epoch_streaming` wykonuje jedną epokę treningową na danych strumieniowych:
# - ustawia model w tryb train()
# - iteruje po batchach z dataloadera (tqdm dla progress baru)
# - zapewnia mixed precision przez torch.cuda.amp (jeśli use_amp=True)
# - dla każdego batcha: forward, loss, backward (skalowany przez grad scaler), clip grad, optimizer.step
# - zbiera predykcje i aktualizuje macierz pomyłek oraz sumuje straty
# - zwraca słownik metryk oraz średnią straty na batch
# =============================================
def train_epoch_streaming(model, optimizer, criterion, dataloader, device, num_classes: int, use_amp: bool = True, grad_clip: Optional[float] = None):
    """Wykonuje jedną epokę treningową na strumieniowych danych (memory-efficient).

    Argumenty:
      model: moduł PyTorch
      optimizer: optymalizator
      criterion: funkcja straty (może zwracać tensor skalar)
      dataloader: iterator zwracający batch'e słowników z kluczami 'x' i 'y'
      device: urządzenie (np. 'cuda')
      num_classes: liczba klas do zbudowania cm
      use_amp: czy używać torch.cuda.amp (domyślnie True)
      grad_clip: opcjonalna wartość do clipowania normy gradientu

    Zwraca:
      słownik metryk (tak jak w `metrics_from_cm`) z dodatkowym kluczem 'loss' zawierającym średnią stratę
      na batch (jeśli wystąpiły poprawne batch'e z obliczoną stratą).

    Zachowanie szczególne:
      - batch'e z NaN/Inf w loss są pomijane (funkcja _safe_loss_value zwraca None)
      - predykcje i targety są przenoszone na CPU przed aktualizacją macierzy pomyłek
    """
    model.train()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    n_batches = 0
    n_loss_batches = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"]
        y_idx = to_index_targets(y)
        optimizer.zero_grad(set_to_none=True)

        # Forward w AMP OK, ale loss liczmy w fp32 (stabilniej, szczególnie CE i Dice)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(logits.float(), y.to(device, non_blocking=True).float())

        loss_val = _safe_loss_value(loss)
        if loss_val is None:
            _debug_invalid_loss("train", logits, y, y_idx)
            # pomijamy backward dla zepsutego batcha
            continue

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(1).detach().cpu()
        cm = update_confusion_matrix(cm, preds, y_idx.cpu(), num_classes)
        total_loss += loss_val
        n_batches += 1
        n_loss_batches += 1

    mets = metrics_from_cm(cm)
    mets["loss"] = total_loss / max(n_loss_batches, 1)
    return mets


# =============================================
#   Pętla walidacyjna (streaming) - per-epoch
# =============================================
# `valid_epoch_streaming` wykonuje jedną epokę walidacyjną:
# - ustawia model w tryb eval() i korzysta z torch.inference_mode() dla wydajności
# - nie wykonuje backward/optimizer.step
# - używa mixed precision w podobny sposób jak w treningu
# - aktualizuje macierz pomyłek i sumuje straty (jeśli criterion != None)
# - zwraca słownik metryk oraz średnią straty
def valid_epoch_streaming(model, criterion, dataloader, device, num_classes: int, use_amp: bool = True):
    """Wykonuje jedną epokę walidacyjną (memory-efficient).

    Argumenty:
      model: moduł PyTorch ustawiony do ewaluacji
      criterion: funkcja straty lub None
      dataloader: iterator batch'y z 'x' i 'y'
      device: urządzenie (np. 'cuda')
      num_classes: liczba klas do zbudowania cm
      use_amp: czy używać AMP

    Zwraca:
      słownik metryk (jak w `metrics_from_cm`) oraz (opcjonalnie) 'loss' jeśli criterion nie jest None
    """
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    n_batches = 0
    n_loss_batches = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Valid", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"]
            y_idx = to_index_targets(y)

            # logits mogą być w AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)

            if criterion is not None:
                # loss zawsze w fp32 (stabilnie)
                with torch.cuda.amp.autocast(enabled=False):
                    loss = criterion(logits.float(), y.to(device, non_blocking=True).float())
                loss_val = _safe_loss_value(loss)
                if loss_val is None:
                    _debug_invalid_loss("valid", logits, y, y_idx)
                else:
                    total_loss += loss_val
                    n_loss_batches += 1

            preds = logits.argmax(1).cpu()
            cm = update_confusion_matrix(cm, preds, y_idx.cpu(), num_classes)
            n_batches += 1

    mets = metrics_from_cm(cm)
    if criterion is not None and n_loss_batches > 0:
        mets["loss"] = total_loss / n_loss_batches
    return mets


def _safe_loss_value(loss: torch.Tensor) -> Optional[float]:
    """Zwraca bezpieczną wartość loss jako float lub None jeśli loss jest NaN/Inf."""
    if loss is None:
        return None
    if not torch.is_tensor(loss):
        try:
            loss = torch.as_tensor(loss)
        except Exception:
            return None
    if not torch.isfinite(loss).all():
        return None
    v = float(loss.detach().item())
    # dodatkowe zabezpieczenie przed ekstremami (u Ciebie valid loss potrafił iść w dziesiątki)
    if v > 1000.0:
        return None
    return v


def _debug_invalid_loss(prefix: str, logits: torch.Tensor, y: torch.Tensor, y_idx: torch.Tensor):
    """Drukuje pomocnicze informacje diagnostyczne, gdy wykryto niepoprawną wartość straty.

    Wyświetla zakres wartości logits, zakres indeksów targetów oraz kształty tensorów.
    Przydaje się do zdiagnozowania przyczyny NaN/Inf w loss.
    """
    try:
        lg_min = float(logits.detach().min().item())
        lg_max = float(logits.detach().max().item())
    except Exception:
        lg_min, lg_max = None, None
    try:
        yi_min = int(y_idx.detach().min().item())
        yi_max = int(y_idx.detach().max().item())
    except Exception:
        yi_min, yi_max = None, None
    print(
        f"[{prefix}] Invalid loss detected. "
        f"logits[min,max]=({lg_min},{lg_max}) target_idx[min,max]=({yi_min},{yi_max}) "
        f"target_shape={tuple(y.shape)} logits_shape={tuple(logits.shape)}"
    )