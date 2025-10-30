import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.data.mvtec import MVTecADDataset
from src.models.cae import CAEModel
from src.utils.common import get_device, load_config, prepare_output_dir, resolve_path, set_seed
from src.utils.transforms import build_mask_transform, build_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out")
    return parser.parse_args()


def load_splits(splits_dir):
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    if not train_file.exists() or not val_file.exists():
        return None, None
    train_paths = [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_paths = [line.strip() for line in val_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return train_paths, val_paths


def create_subsets(dataset, train_paths, val_paths):
    path_to_index = {str(item[0]): idx for idx, item in enumerate(dataset.entries)}
    train_indices = [path_to_index[path] for path in train_paths if path in path_to_index]
    val_indices = [path_to_index[path] for path in val_paths if path in path_to_index]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


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
    splits_path = resolve_path(Path("data") / "splits" / category)
    train_paths, val_paths = load_splits(splits_path)
    if train_paths is not None and val_paths is not None:
        train_subset, val_subset = create_subsets(dataset, train_paths, val_paths)
    else:
        val_size = max(1, int(0.1 * len(dataset)))
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_subset,
        batch_size=config["cae"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["cae"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    model = CAEModel(
        image_size=image_size,
        config=config["cae"],
        device=device,
    )
    epochs = args.epochs if args.epochs is not None else config["cae"]["epochs"]
    model.fit(train_loader, val_loader=val_loader, epochs=epochs)
    out_root = args.out or config.get("artifacts_dir", "artifacts")
    out_dir = prepare_output_dir(resolve_path(out_root), "cae", category)
    out_path = out_dir / "model.pt"
    model.save(out_path)
    print(f"Zapisano model CAE w {out_path}")


if __name__ == "__main__":
    main()
