import argparse
import random
from pathlib import Path

from src.utils.common import ensure_dir, load_config, resolve_path, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_dir")
    parser.add_argument("--splits_dir")
    return parser.parse_args()


def collect_images(root):
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    images = []
    for pattern in patterns:
        images.extend(sorted(root.glob(pattern)))
    return images


def main():
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or config.get("data_dir", "data/mvtec_ad")
    dataset_root = resolve_path(data_dir) / args.category / "train" / "good"
    if not dataset_root.exists():
        raise RuntimeError(f"Nie znaleziono katalogu {dataset_root}")
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)
    images = collect_images(dataset_root)
    if not images:
        raise RuntimeError("Brak obrazów do podziału")
    random.shuffle(images)
    val_count = max(1, int(len(images) * args.val_ratio))
    val_images = images[:val_count]
    train_images = images[val_count:]
    splits_base = args.splits_dir or Path("data") / "splits"
    splits_dir = ensure_dir(resolve_path(splits_base) / args.category)
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    train_file.write_text("\n".join(str(p) for p in train_images), encoding="utf-8")
    val_file.write_text("\n".join(str(p) for p in val_images), encoding="utf-8")
    print(f"Zapisano {len(train_images)} ścieżek treningowych i {len(val_images)} walidacyjnych w {splits_dir}")


if __name__ == "__main__":
    main()
