import numpy as np
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


def log_epoch_results(log_path: str, epoch_idx: int, lr_str: str, train_logs: dict, valid_logs: dict):
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
        f.write("Train F1: {:.6f}, Valid F1: {:.6f}\n\n".format(_g(train_logs, "f1"), _g(valid_logs, "f1")))

        print(f"\nEpoch: {epoch_idx} (lr: {lr_str})")
        print(f"Train Loss: {train_logs.get('loss'):.6f}, Valid Loss: {valid_logs.get('loss'):.6f}")
        print(f"Train IoU: {train_logs.get('iou'):.6f}, Valid IoU: {valid_logs.get('iou'):.6f}")
        print(f"Train Dice: {train_logs.get('dice'):.6f}, Valid Dice: {valid_logs.get('dice'):.6f}")
        print(f"Train Acc: {train_logs.get('acc'):.6f}, Valid Acc: {valid_logs.get('acc'):.6f}")
        print(f"Train Prec: {train_logs.get('prec'):.6f}, Valid Prec: {valid_logs.get('prec'):.6f}")
        print(f"Train Rec: {train_logs.get('rec'):.6f}, Valid Rec: {valid_logs.get('rec'):.6f}")
        print(f"Train F1: {train_logs.get('f1'):.6f}, Valid F1: {valid_logs.get('f1'):.6f}")

def _get_lr(optimizer):
    return ", ".join(f"{g['lr']:.6g}" for g in optimizer.param_groups)

def log_number_of_parameters(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)
