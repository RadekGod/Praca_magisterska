import argparse
import os
from collections import OrderedDict

import torch
import segmentation_models_pytorch as smp

from source import transforms as T
from source.check_fusion_utils import (
    load_rgb_sar_pair_from_rgb_path,
    preprocess_rgb_sar_to_4ch_tensor,
    logits_to_mask_rgb,
    print_pred_distribution,
    save_mask_geotiff,
)


# Mapa kolorów: indeks klasy -> (R, G, B)
CLASS_COLORS = {
    0: (0, 0, 0),        # tło
    1: (255, 0, 0),      # klasa 1 - czerwony
    2: (0, 255, 0),      # klasa 2 - zielony
    3: (0, 0, 255),      # klasa 3 - niebieski
    4: (255, 255, 0),    # klasa 4 - żółty
    5: (255, 0, 255),    # klasa 5 - magenta
    6: (0, 255, 255),    # klasa 6 - cyjan
    7: (255, 128, 0),    # klasa 7 - pomarańczowy
    8: (128, 0, 255),    # klasa 8 - fioletowy
}


def build_model(num_classes: int, device: str = "cuda"):
    """Buduje model early-fusion (RGB+SAR = 4 kanały) zgodny z train_early_fusion.py."""
    model = smp.Unet(
        classes=num_classes,
        in_channels=4,
        activation=None,
        encoder_weights=None,  # wagi i tak wczytamy z .pth
        encoder_name=encoder_name,
        decoder_attention_type="scse",
    )
    model.to(device)
    model.eval()
    return model


def _print_first_conv_diagnostics(model: torch.nn.Module):
    """Wypisuje diagnostykę pierwszej konwolucji (czy faktycznie 4 kanały, statystyki wag)."""
    target = model.module if hasattr(model, "module") else model
    for name, p in target.named_parameters():
        if p.dim() == 4 and any(k in name.lower() for k in ["conv_stem", "conv1", "stem", "conv_first"]):
            w = p.detach().cpu()
            print(f"First conv param: {name} shape: {tuple(w.shape)}")
            print(f"  mean: {float(w.mean().item()):.6f} std: {float(w.std().item()):.6f}")
            ww = w.numpy()
            if ww.ndim == 4:
                ch_means = ww.mean(axis=(0, 2, 3))
                print("  Per-channel means:", ", ".join(f"{v:.6f}" for v in ch_means))
            return
    print("[WARN] Nie znaleziono parametru pierwszej konwolucji do diagnostyki.")


def run_inference(
    model,
    image_path: str,
    output_path: str,
    device: str,
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
):
    """Inferencja dla pojedynczego obrazu fusion (para RGB+SAR) i zapis kolorowej maski."""
    rgb, sar, georef_path = load_rgb_sar_pair_from_rgb_path(image_path)

    x = preprocess_rgb_sar_to_4ch_tensor(
        rgb,
        sar,
        sar_normalize=sar_normalize,
        sar_mean=sar_mean,
        sar_std=sar_std,
    ).to(device)

    with torch.no_grad():
        logits = model(x)

    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype('uint8')
    print_pred_distribution(pred, num_classes=len(CLASS_COLORS))

    rgb_mask = logits_to_mask_rgb(logits, class_colors=CLASS_COLORS)
    save_mask_geotiff(output_path, rgb_mask, georef_path)


def main():
    parser = argparse.ArgumentParser(description="check_model_fusion: inferencja + zapis kolorowej maski (RGB+SAR)")
    parser.add_argument(
        "--image_path",
        type=str,
        default="results/obrazy/fusion/oryginalne",
        help=(
            "Ścieżka do FOLDERU z parami plików: *_RGB.tif oraz *_SAR.tif (np. TrainArea_3902_RGB.tif i TrainArea_3902_SAR.tif). "
            "Tryb pojedynczego pliku nie jest obsługiwany."
        ),
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Wymuś CPU zamiast GPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/fusion/" + main_model_name + model_name_variant + ".pth",
        help="Ścieżka do modelu .pth",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/obrazy/fusion/model/" + encoder_name + "/" + model_name_variant + "/" + batch_size + "/" + model_fusion_variant,
        help="Ścieżka wyjściowa (folder).",
    )

    parser.add_argument(
        "--sar_normalize",
        type=str,
        default="global",
        choices=["global", "per_sample", "none"],
        help="Normalizacja SAR: global (mean/std z datasetu), per_sample, none",
    )
    parser.add_argument("--sar_mean", type=float, default=None, help="Mean SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument("--sar_std", type=float, default=None, help="Std SAR na skali [0,1] (opcjonalnie).")
    parser.add_argument(
        "--sar_stats_root",
        type=str,
        default="../dataset/train",
        help="Folder treningowy (np. ../dataset/train), z którego policzymy mean/std SAR jeśli nie podasz ręcznie.",
    )

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Ładuję wagi modelu z: {args.model_path}")

    # SAR mean/std (jeśli global)
    sar_mean = args.sar_mean
    sar_std = args.sar_std
    if args.sar_normalize == "global" and (sar_mean is None or sar_std is None):
        try:
            from pathlib import Path

            label_paths = [str(f) for f in Path(args.sar_stats_root).rglob("*.tif") if "labels" in f.parts]
            m, s = T.compute_sar_stats(label_paths, load_fn=None)
            sar_mean = m if sar_mean is None else sar_mean
            sar_std = s if sar_std is None else sar_std
            print(f"SAR stats (computed): mean={sar_mean} std={sar_std} (scale [0,1])")
        except Exception as e:
            print("[WARN] Nie udało się policzyć SAR mean/std, inferencja może być bez normalizacji SAR.")
            print("       Błąd:", e)

    # build + load
    num_classes = len(CLASS_COLORS)
    model = build_model(num_classes=num_classes, device=device)

    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print("[WARN] Missing keys when loading state_dict:")
        for k in missing[:50]:
            print("  -", k)
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    if unexpected:
        print("[WARN] Unexpected keys when loading state_dict:")
        for k in unexpected[:50]:
            print("  -", k)
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more")

    _print_first_conv_diagnostics(model)
    if sar_mean is not None and sar_std is not None:
        print(f"SAR mean/std used: mean={float(sar_mean):.6f} std={float(sar_std):.6f} (scale [0,1])")

    # Wejściem musi być folder
    if not os.path.isdir(args.image_path):
        raise ValueError(
            "Wymagany jest folder z parami *_RGB.tif + *_SAR.tif. "
            f"Podano: {args.image_path}"
        )

    in_dir = args.image_path
    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    rgb_files = [fn for fn in sorted(os.listdir(in_dir)) if fn.lower().endswith("_rgb.tif")]
    if not rgb_files:
        raise FileNotFoundError(f"Nie znaleziono plików *_RGB.tif w folderze: {in_dir}")

    missing_sar = []
    for fn in rgb_files:
        sar_fn = fn[:-8] + "_SAR.tif"
        if not os.path.exists(os.path.join(in_dir, sar_fn)):
            missing_sar.append(sar_fn)
    if missing_sar:
        raise FileNotFoundError(
            "Brakuje pasujących plików *_SAR.tif dla części *_RGB.tif. Brakujące (pierwsze 20): "
            + ", ".join(missing_sar[:20])
        )

    for fn in rgb_files:
        in_path = os.path.join(in_dir, fn)
        out_path = os.path.join(out_dir, fn.replace("_RGB.tif", ".tif"))
        print(f"Przetwarzam: {in_path} -> {out_path}")
        run_inference(
            model=model,
            image_path=in_path,
            output_path=out_path,
            device=device,
            sar_normalize=args.sar_normalize,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )


if __name__ == "__main__":
    model_name_variant = "T1V1"
    model_fusion_variant = "EARLY"
    encoder_name = "efficientnet-b4"
    batch_size = "Batch_16"
    main_model_name = "EARLY_FUSION_Unet_" + batch_size + "_" + encoder_name + "_WCEPlusDice_ce1.0_dice1.0_lr_0.001_augm"
    main()
