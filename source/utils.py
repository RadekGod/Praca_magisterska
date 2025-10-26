import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


def progress(train_logs, valid_logs, loss_nm, metric_nm, nepochs, outdir, fn_out):
    loss_t = [dic[loss_nm] for dic in train_logs]
    loss_v = [dic[loss_nm] for dic in valid_logs]
    score_t = [dic[metric_nm] for dic in train_logs]
    score_v = [dic[metric_nm] for dic in valid_logs]

    epochs = range(0, len(score_t))
    plt.figure(figsize=(12, 5))

    # Train and validation metric
    # ---------------------------
    plt.subplot(1, 2, 1)

    idx = np.nonzero(score_t == max(score_t))[0][0]
    label = f"Train, {metric_nm}={max(score_t):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_t, "b", label=label)

    idx = np.nonzero(score_v == max(score_v))[0][0]
    label = f"Valid, {metric_nm}={max(score_v):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_v, "r", label=label)

    plt.title("Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel(metric_nm)
    plt.ylim(0, 1)
    plt.legend()

    # Train and validation loss
    # -------------------------
    plt.subplot(1, 2, 2)
    ymax = max(max(loss_t), max(loss_v))
    ymin = min(min(loss_t), min(loss_v))
    ymax = 1 if ymax <= 1 else ymax + 0.5
    ymin = 0 if ymin <= 0.5 else ymin - 0.5

    idx = np.nonzero(loss_t == min(loss_t))[0][0]
    label = f"Train {loss_nm}={min(loss_t):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_t, "b", label=label)

    idx = np.nonzero(loss_v == min(loss_v))[0][0]
    label = f"Valid {loss_nm}={min(loss_v):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_v, "r", label=label)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel("Loss")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(f"{outdir}/{fn_out}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    return


def append_epoch_log(log_path: str, epoch_idx: int, lr_str: str, train_logs: dict, valid_logs: dict):
    """Appenduje metryki epoki do pliku tekstowego.

    Parametry:
    - log_path: ścieżka do pliku .txt
    - epoch_idx: numer epoki (1-based)
    - lr_str: string z LR dla wszystkich grup paramów
    - train_logs: dict z kluczami: loss, iou, dice, acc, prec, rec
    - valid_logs: dict z kluczami: loss, iou, dice, acc, prec, rec
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
        f.write("Train Rec: {:.6f}, Valid Rec: {:.6f}\n\n".format(_g(train_logs, "rec"), _g(valid_logs, "rec")))
