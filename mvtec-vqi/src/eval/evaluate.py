import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from src.data.mvtec import MVTecADDataset
from src.infer.predictor import AnomalyPredictor
from src.utils import viz
from src.utils.common import get_device, load_config, prepare_output_dir, resolve_path, set_seed
from src.utils.metrics import auroc_image, auroc_pixel, dice_at_threshold, threshold_map_otsu, threshold_map_percentile
from src.utils.transforms import build_mask_transform, build_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", default="padim_resnet50")
    parser.add_argument("--category")
    parser.add_argument("--device")
    parser.add_argument("--batch_size", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    category = args.category or config.get("category", "bottle")
    set_seed(config.get("seed", 42))
    device = get_device(args.device)
    image_size = config.get("image_size", 256)
    transform = build_transform(image_size, normalize=args.backend != "cae")
    mask_transform = build_mask_transform(image_size)
    dataset = MVTecADDataset(
        root=resolve_path(config.get("data_dir", "data/mvtec_ad")),
        category=category,
        split="test",
        transform=transform,
        mask_transform=mask_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size or 1,
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    predictor = AnomalyPredictor(
        backend=args.backend,
        category=category,
        config=config,
        device=device,
    )
    image_scores = []
    image_labels = []
    maps = []
    masks = []
    paths = []
    visuals = []
    for batch in dataloader:
        images, labels, mask_tensors, file_paths = batch
        for image_tensor, label, mask_tensor, path in zip(images, labels, mask_tensors, file_paths):
            result = predictor.predict_tensor(image_tensor.unsqueeze(0))
            image_scores.append(result["score"])
            image_labels.append(int(label))
            maps.append(result["anomaly_map"])
            masks.append(mask_tensor.squeeze(0).numpy())
            paths.append(path)
            overlay = result["overlay"]
            visuals.append((overlay, result["heatmap"], path))
    image_auroc = auroc_image(image_scores, image_labels)
    pixel_auroc = auroc_pixel(maps, masks)
    dice_otsu_values = []
    dice_percent_values = []
    percentile = config["evaluation"].get("percentile_threshold", 0.9)
    for amap, mask in zip(maps, masks):
        otsu = threshold_map_otsu(amap)
        dice_otsu_values.append(dice_at_threshold(amap, mask, otsu))
        threshold = threshold_map_percentile(amap, percentile)
        dice_percent_values.append(dice_at_threshold(amap, mask, threshold))
    dice_otsu = float(np.mean(dice_otsu_values)) if dice_otsu_values else float("nan")
    dice_percent = float(np.mean(dice_percent_values)) if dice_percent_values else float("nan")
    report_dir = prepare_output_dir(resolve_path(config.get("reports_dir", "reports")), args.backend, category)
    report_path = report_dir / "metrics.json"
    report = {
        "backend": args.backend,
        "category": category,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "dice_otsu": dice_otsu,
        "dice_percent": dice_percent,
        "threshold_percentile": percentile,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    top_indices = np.argsort(np.array(image_scores))[::-1][: min(5, len(visuals))]
    for idx in top_indices:
        overlay, heatmap, path = visuals[idx]
        base_name = Path(path).stem
        overlay_path = report_dir / f"{base_name}_overlay.png"
        heatmap_path = report_dir / f"{base_name}_heatmap.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    print(f"Raport zapisany w {report_path}")


if __name__ == "__main__":
    main()
