import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.mvtec import MVTecADDataset
from src.models.padim import PaDiMModel
from src.utils.common import get_device, load_config, prepare_output_dir, resolve_path, set_seed
from src.utils.transforms import build_mask_transform, build_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category")
    parser.add_argument("--device")
    parser.add_argument("--out")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    category = args.category or config.get("category", "bottle")
    set_seed(config.get("seed", 42))
    device = get_device(args.device)
    image_size = config.get("image_size", 256)
    transform = build_transform(image_size)
    dataset = MVTecADDataset(
        root=resolve_path(config.get("data_dir", "data/mvtec_ad")),
        category=category,
        split="train",
        transform=transform,
        mask_transform=build_mask_transform(image_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    backbone = config["padim"].get("backbone", "resnet50")
    model = PaDiMModel(
        image_size=image_size,
        selected_channels=config["padim"]["selected_channels"],
        gaussian_kernel=config["padim"]["gaussian_kernel"],
        blur_sigma=config["padim"]["blur_sigma"],
        device=device,
        seed=config.get("seed", 42),
        backbone=backbone,
    )
    model.fit(dataloader)
    out_root = args.out or config.get("artifacts_dir", "artifacts")
    out_dir = prepare_output_dir(resolve_path(out_root), f"padim_{backbone}", category)
    out_path = out_dir / "model.pt"
    model.save(out_path)
    print(f"Zapisano statystyki PaDiM ({backbone}) w {out_path}")


if __name__ == "__main__":
    main()
