from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MVTecADDataset(Dataset):
    def __init__(self, root, category, split, transform=None, mask_transform=None):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.entries = self._scan()

    def _scan(self):
        entries = []
        base = self.root / self.category / self.split
        if not base.exists():
            raise RuntimeError(f"Nie znaleziono katalogu {base}")
        if self.split == "train":
            good_dir = base / "good"
            for path in sorted(good_dir.rglob("*")):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    entries.append((path, 0, None))
        else:
            ground_truth_root = self.root / self.category / "ground_truth"
            for defect_dir in sorted(base.iterdir()):
                if not defect_dir.is_dir():
                    continue
                label = 0 if defect_dir.name == "good" else 1
                for path in sorted(defect_dir.rglob("*")):
                    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    mask_path = None
                    if label == 1:
                        rel = path.relative_to(base / defect_dir.name)
                        mask_path = ground_truth_root / defect_dir.name / rel
                        mask_path = mask_path.with_name(mask_path.stem + "_mask" + mask_path.suffix)
                        if not mask_path.exists():
                            mask_path = None
                    entries.append((path, label, mask_path))
        if not entries:
            raise RuntimeError("Brak danych w zadanym zbiorze")
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, label, mask_path = self.entries[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        if self.split == "train":
            mask_tensor = torch.zeros(1, image_tensor.shape[1], image_tensor.shape[2])
        else:
            if mask_path and mask_path.exists():
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", image.size, 0)
            if self.mask_transform:
                mask_tensor = self.mask_transform(mask)
            else:
                mask_tensor = torch.from_numpy(np.array(mask, dtype="float32") / 255.0).unsqueeze(0)
        return image_tensor, label, mask_tensor, str(path)
