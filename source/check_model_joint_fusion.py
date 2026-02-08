import argparse
import os
from collections import OrderedDict

import torch

# --- Umożliwia uruchomienie skryptu jako pliku: python path\to\check_model_joint_fusion.py
# (wtedy katalog projektu może nie być na PYTHONPATH).
try:
    from source import transforms as T
    from source.train_joint_fusion import JointFusionUnet
    from source.check_fusion_utils import (
        load_rgb_sar_pair_from_rgb_path,
        preprocess_rgb_sar_to_4ch_tensor,
        logits_to_mask_rgb,
        print_pred_distribution,
        save_mask_geotiff,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from source import transforms as T
    from source.train_joint_fusion import JointFusionUnet
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


def build_model(
    num_classes: int,
    device: str = "cuda",
    feature_fusion: str = "concat",
):
    """Buduje model JointFusion zgodny z train_joint_fusion.py."""
    model = JointFusionUnet(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights_rgb=None,  # wagi i tak wczytamy z checkpointu .pth
        encoder_weights_sar=None,
        decoder_attention_type="scse",
        feature_fusion=feature_fusion,
        encoder_depth=5,
        safe_nan_to_num=True,
    )
    model.to(device)
    model.eval()
    return model


def run_inference(
    model,
    rgb_path: str,
    output_path: str,
    device: str,
    sar_normalize: str = "global",
    sar_mean: float | None = None,
    sar_std: float | None = None,
):
    """Inferencja dla pojedynczej pary RGB+SAR (wejście: ścieżka do *_RGB.tif)."""
    rgb, sar, georef_path = load_rgb_sar_pair_from_rgb_path(rgb_path)

    x = preprocess_rgb_sar_to_4ch_tensor(
        rgb,
        sar,
        sar_normalize=sar_normalize,
        sar_mean=sar_mean,
        sar_std=sar_std,
    ).to(device)

    with torch.no_grad():
        logits = model(x)  # (1,C,H,W)

    pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype("uint8")
    print_pred_distribution(pred, num_classes=len(CLASS_COLORS))

    rgb_mask = logits_to_mask_rgb(logits, class_colors=CLASS_COLORS)
    save_mask_geotiff(output_path, rgb_mask, georef_path)


def main():
    parser = argparse.ArgumentParser(
        description="check_model_joint_fusion: Joint/Intermediate Fusion (feature-level) inferencja + zapis kolorowej maski"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="results/obrazy/fusion/oryginalne",
        help=(
            "Ścieżka do FOLDERU z parami plików: *_RGB.tif oraz *_SAR.tif. "
            "Tryb pojedynczego pliku nie jest obsługiwany."
        ),
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Wymuś CPU zamiast GPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/fusion/" + main_model_name + model_name_variant + ".pth",
        help="Ścieżka do checkpointu joint-fusion (.pth) zapisanego przez train_joint_fusion.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/obrazy/fusion/model/" + encoder_name + "/" + model_name_variant + "/" + batch_size + "/" + model_fusion_variant,
        help="Folder wyjściowy na maski .tif",
    )
    parser.add_argument(
        "--feature_fusion",
        type=str,
        default="concat",
        choices=["concat", "sum", "mean"],
        help="Jak łączyć feature maps RGB i SAR na każdym poziomie (musi pasować do treningu)",
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
    model = build_model(num_classes=num_classes, device=device, feature_fusion=args.feature_fusion)

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

    # Wejściem musi być folder
    if not os.path.isdir(args.image_path):
        raise ValueError(f"Wymagany jest folder z parami *_RGB.tif + *_SAR.tif. Podano: {args.image_path}")

    in_dir = args.image_path
    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    # Przetwarzamy tylko *_RGB.tif
    rgb_files = [fn for fn in sorted(os.listdir(in_dir)) if fn.lower().endswith("_rgb.tif")]
    if not rgb_files:
        raise FileNotFoundError(f"Nie znaleziono plików *_RGB.tif w folderze: {in_dir}")

    # walidacja par
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
            rgb_path=in_path,
            output_path=out_path,
            device=device,
            sar_normalize=args.sar_normalize,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )


if __name__ == "__main__":
    model_name_variant = "T1V1"
    model_fusion_variant = "JOINT_CONCAT"
    encoder_name = "efficientnet-b4"
    batch_size = "Batch_16"
    main_model_name = "JOINT_FUSION_Unet_" + batch_size + "_" + encoder_name + "_WCEPlusDice_ce1.0_dice1.0_lr_0.001_feat_concat_augm"
    main()
