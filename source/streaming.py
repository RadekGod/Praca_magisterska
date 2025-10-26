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
    if y.ndim == 4 and y.size(1) > 1:
        return y.argmax(1)
    return y.squeeze(1)

def update_confusion_matrix(cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int, chunk_size: int = 262144):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    k = (targets >= 0) & (targets < num_classes)
    preds = preds[k]
    targets = targets[k]
    n = preds.numel()
    for i in range(0, n, chunk_size):
        p = preds[i:i+chunk_size]
        t = targets[i:i+chunk_size]
        inds = (t.to(torch.int64) * num_classes + p.to(torch.int64))
        cm += torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm

def metrics_from_cm(cm: torch.Tensor):
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    eps = 1e-7
    iou_c = tp / (tp + fp + fn + eps)
    dice_c = (2 * tp) / (2 * tp + fp + fn + eps)
    prec_c = tp / (tp + fp + eps)
    rec_c = tp / (tp + fn + eps)
    acc_global = (tp.sum() / (cm.sum().float() + eps)).item()
    return {
        "iou": iou_c.mean().item(),
        "dice": dice_c.mean().item(),
        "acc": acc_global,
        "prec": prec_c.mean().item(),
        "rec": rec_c.mean().item(),
    }

def train_epoch_streaming(model, optimizer, criterion, dataloader, device, num_classes: int, use_amp: bool = True, grad_clip: Optional[float] = None):
    model.train()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    n_batches = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"]
        y_idx = to_index_targets(y)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y.to(device, non_blocking=True))
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        preds = logits.argmax(1).detach().cpu()
        cm = update_confusion_matrix(cm, preds, y_idx.cpu(), num_classes)
        total_loss += float(loss.detach())
        n_batches += 1

    mets = metrics_from_cm(cm)
    mets["loss"] = total_loss / max(n_batches, 1)
    return mets

def valid_epoch_streaming(model, criterion, dataloader, device, num_classes: int, use_amp: bool = True):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    n_batches = 0
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
        for batch in tqdm(dataloader, desc="Valid", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"]
            y_idx = to_index_targets(y)
            logits = model(x)
            if criterion is not None:
                total_loss += float(criterion(logits, y.to(device, non_blocking=True)))
            preds = logits.argmax(1).cpu()
            cm = update_confusion_matrix(cm, preds, y_idx.cpu(), num_classes)
            n_batches += 1
    mets = metrics_from_cm(cm)
    if criterion is not None and n_batches > 0:
        mets["loss"] = total_loss / n_batches
    return mets

