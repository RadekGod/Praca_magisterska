"""Lossy do segmentacji (moduł.

Zawiera implementacje strat używanych w projekcie:

- DiceLoss: miękki (soft) Dice loss dla segmentacji wieloklasowej.
  Oczekuje, że 'logits' to surowe wyjścia sieci (bez softmax) o kształcie [B, C, H, W],
  a 'target' to one-hot [B, C, H, W].

- WeightedCEPlusDice: kombinacja ważonej CrossEntropy (liczonej na indeksach klas)
  i Dice (liczonego na softmax + one-hot). Przydatne, gdy chcemy łączyć cechy obu strat.

- CEWithLogitsLoss: wrapper na CrossEntropy, który przyjmuje target w formie one-hot
  i robi argmax wewnątrz.

Konwencje i uwagi:
- "logits" to surowe wartości wyjściowe sieci (nieprobabilistyczne); DiceLoss sam
  wykona softmax przed obliczeniami.
- "target" powinien być one-hot [B, C, H, W] jeśli loss tego wymaga (Dice, WeightedCEPlusDice,
  CEWithLogitsLoss). Jeśli używasz transformacji, upewnij się że target jest skonwertowany
  do one-hot przed wywołaniem tych strat.
- CrossEntropy w PyTorch oczekuje indeksów klas; klasy, które przyjmują one-hot,
  konwertują go wewnętrznie przez argmax(dim=1).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice loss dla segmentacji wieloklasowej.

    Opis:
    - Liczy "miękki" Dice score na podstawie prawdopodobieństw (softmax(logits)) przeciw
      one-hot target.

    Argumenty konstrukora:
    - smooth (float): czynnik stabilizujący w liczniku i mianowniku (domyślnie 1.0).
    - include_background (bool): czy wliczać kanał tła (indeks 0). Jeżeli False i
      jest więcej niż 1 klasa, tło zostanie wycięte (przydatne, gdy tło dominuje).

    Kształty:
    - logits: torch.Tensor o kształcie [B, C, H, W]
    - target: torch.Tensor one-hot o kształcie [B, C, H, W]

    Zwraca:
    - skalarna strata (tensor) typu float.

    Błędy:
    - ValueError jeśli `target.ndim != 4` (oczekiwany one-hot [B,C,H,W]).
    """

    def __init__(self, smooth: float = 1.0, include_background: bool = False):
        super().__init__()
        self.smooth = float(smooth)
        self.include_background = bool(include_background)
        self.name = "DiceLoss"

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        if target.ndim != 4:
            raise ValueError(f"DiceLoss expects target one-hot [B,C,H,W], got {tuple(target.shape)}")

        if not self.include_background and probs.size(1) > 1:
            probs = probs[:, 1:]
            target = target[:, 1:]

        dims = (0, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        denom = (probs + target).sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class WeightedCEPlusDice(nn.Module):
    """Kombinacja: ważona CrossEntropy + Dice.

    Opis:
    - CrossEntropy (ważona): liczone na indeksach klas (target.argmax(dim=1)).
    - Dice: liczone na softmax(logits) i na one-hot target.

    Argumenty konstrukora:
    - class_weights: lista/ndarray/tensor wag dla CrossEntropy (długość = liczba klas).
    - ce_weight (float): waga składowej CrossEntropy w łącznej stracie.
    - dice_weight (float): waga składowej Dice w łącznej stracie.
    - dice_smooth (float): parametr smooth przekazywany do DiceLoss.
    - dice_include_background (bool): czy wliczać kanał tła w DiceLoss.
    - device (str): device na którym umieszczone zostaną wagi (np. "cuda" lub "cpu").

    Kształty i zachowanie:
    - logits: [B, C, H, W]
    - target: one-hot [B, C, H, W]

    Zwraca:
    - skalarna strata będąca sumą ważonych składowych CE i Dice.
    """

    def __init__(self, class_weights, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 dice_smooth: float = 1.0, dice_include_background: bool = False, device: str = "cuda"):
        super().__init__()
        w = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.dice = DiceLoss(smooth=dice_smooth, include_background=dice_include_background)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.name = f"WCEPlusDice_ce{self.ce_weight}_dice{self.dice_weight}"

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_idx = target.argmax(dim=1)
        ce = self.ce(logits, y_idx)
        dice = self.dice(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice


class CEWithLogitsLoss(nn.Module):
    """CrossEntropy loss, przyjmujący target w formie one-hot.

    Opis:
    - Przyjmuje w konstruktorze wagi (numpy array) i device.
    - W forward robi argmax(target, dim=1) i wywołuje nn.CrossEntropyLoss.

    Argumenty konstrukora:
    - weights: numpy array wag dla klas.
    - device: nazwa urządzenia, na które przeniesione są wagi (domyślnie "cuda").

    Kształty:
    - input / logits: [B, C, H, W]
    - target: one-hot [B, C, H, W]

    Zwraca:
    - skalarna strata CrossEntropy.
    """

    def __init__(self, weights, device="cuda"):
        super().__init__()
        self.weight = torch.from_numpy(weights).float().to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.name = "CELoss"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        return loss
