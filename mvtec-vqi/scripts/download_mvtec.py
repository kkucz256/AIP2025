import argparse
import os
import subprocess
from pathlib import Path

from src.utils.common import ensure_dir, load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or config.get("data_dir", "data/mvtec_ad")
    target_dir = resolve_path(data_dir)
    ensure_dir(target_dir)
    dataset_dir = target_dir / "mvtec_anomaly_detection"
    if dataset_dir.exists():
        print(f"Dane już istnieją w {dataset_dir}")
        return
    env_home = Path(os.getenv("HOME", ""))
    kaggle_token = env_home / ".kaggle" / "kaggle.json"
    if not kaggle_token.exists():
        raise RuntimeError("Brak pliku ~/.kaggle/kaggle.json dla API Kaggle")
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "ipythonx/mvtec-ad",
        "-p",
        str(target_dir),
        "--unzip",
    ]
    print("Pobieranie MVTec AD...")
    subprocess.run(command, check=True)
    archives = list(target_dir.glob("*.zip"))
    for archive in archives:
        archive.unlink()
    print(f"Gotowe. Dane w {dataset_dir}")


if __name__ == "__main__":
    main()
