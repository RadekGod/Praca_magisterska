import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice loss dla segmentacji wieloklasowej.

    Oczekuje:
    - logits: [B, C, H, W]
    - target: one-hot [B, C, H, W] (tak jak u Ciebie po ToTensor)

    include_background=False zwykle pomaga, gdy background dominuje.
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
    """Kombinacja: weighted CrossEntropy + Dice.

    - WCE: liczone po indeksach klas (target.argmax(1))
    - Dice: liczone na softmax i one-hot
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
    def __init__(self, weights, device="cuda"):
        super().__init__()
        self.weight = torch.from_numpy(weights).float().to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.name = "CELoss"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        return loss
