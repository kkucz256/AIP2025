
import cv2
import numpy as np
import torch
from PIL import Image

from src.models.cae import CAEModel
from src.models.padim import PaDiMModel
from src.utils import viz
from src.utils.common import get_device, resolve_path
from src.utils.transforms import build_transform


class AnomalyPredictor:
    def __init__(self, backend, category, config, device=None):
        self.backend = backend
        self.category = category
        self.config = config
        self.device = device or get_device()
        self.transform = build_transform(config.get("image_size", 256))
        self.model = None
        self._load_model()

    def _artifact_path(self):
        artifacts_root = resolve_path(self.config.get("artifacts_dir", "artifacts"))
        backend_dir = artifacts_root / self.backend / self.category
        path = backend_dir / "model.pt"
        if not path.exists():
            raise RuntimeError(f"Brak pliku modelu {path}")
        return path

    def _load_model(self):
        if self.backend == "padim_resnet50":
            model = PaDiMModel(
                image_size=self.config.get("image_size", 256),
                selected_channels=self.config["padim"]["selected_channels"],
                gaussian_kernel=self.config["padim"]["gaussian_kernel"],
                blur_sigma=self.config["padim"]["blur_sigma"],
                device=self.device,
                seed=self.config.get("seed", 42),
            )
            model.load(self._artifact_path())
            self.model = model
        elif self.backend == "cae":
            model = CAEModel(
                image_size=self.config.get("image_size", 256),
                config=self.config["cae"],
                device=self.device,
            )
            model.load(self._artifact_path())
            self.model = model
        else:
            raise ValueError(f"Nieobsługiwany backend {self.backend}")

    def preprocess_array(self, array):
        image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def predict_tensor(self, tensor):
        with torch.no_grad():
            if self.backend == "padim_resnet50":
                maps, scores = self.model.predict(tensor.to(self.device))
            else:
                maps, scores = self.model.predict(tensor.to(self.device))
        amap = maps[0]
        score = scores[0]
        array = tensor[0]
        image = viz.tensor_to_image(array)
        resized_map = cv2.resize(amap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = viz.map_to_heatmap(resized_map)
        overlay = viz.overlay_heatmap(image, heatmap, 0.5)
        return {
            "score": float(score),
            "anomaly_map": resized_map,
            "heatmap": heatmap,
            "overlay": overlay,
        }

    def predict_array(self, array):
        tensor = self.preprocess_array(array)
        result = self.predict_tensor(tensor)
        return result
