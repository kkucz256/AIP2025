
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
        normalize = backend != "cae"
        self.transform = build_transform(config.get("image_size", 256), normalize=normalize)
        infer_cfg = config.get("infer", {})
        backend_cfg = infer_cfg.get("backend_defaults", {}).get(backend, {})
        self.default_score_percentile = backend_cfg.get("score_percentile", infer_cfg.get("score_percentile"))
        self.model = None
        self._amp_enabled = self._should_enable_amp()
        self._load_model()

    def _should_enable_amp(self):
        infer_cfg = self.config.get("infer", {})
        value = infer_cfg.get("use_amp", "auto")
        if isinstance(value, str) and value.lower() == "auto":
            return self.device.type == "cuda"
        enabled = bool(value)
        return enabled and self.device.type in {"cuda", "cpu"}

    def _artifact_path(self):
        artifacts_root = resolve_path(self.config.get("artifacts_dir", "artifacts"))
        backend_dir = artifacts_root / self.backend / self.category
        path = backend_dir / "model.pt"
        if not path.exists():
            raise RuntimeError(f"Brak pliku modelu {path}")
        return path

    def _load_model(self):
        if self.backend.startswith("padim"):
            # Extract backbone from backend name (e.g., padim_resnet50 -> resnet50)
            parts = self.backend.split("_", 1)
            backbone = parts[1] if len(parts) > 1 else "resnet50"
            
            model = PaDiMModel(
                image_size=self.config.get("image_size", 256),
                selected_channels=self.config["padim"]["selected_channels"],
                gaussian_kernel=self.config["padim"]["gaussian_kernel"],
                blur_sigma=self.config["padim"]["blur_sigma"],
                device=self.device,
                seed=self.config.get("seed", 42),
                backbone=backbone,
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
            model.config["use_amp"] = self._amp_enabled
            self.model = model
        else:
            raise ValueError(f"Nieobsługiwany backend {self.backend}")

    def preprocess_array(self, array):
        image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def predict_tensor(self, tensor, score_percentile=None, blur_override=None):
        effective_percentile = self.default_score_percentile if score_percentile is None else score_percentile
        if effective_percentile is not None:
            effective_percentile = float(effective_percentile)
        if self.backend.startswith("padim"):
            maps, scores, raw_maps = self.model.predict(
                tensor.to(self.device),
                score_percentile=effective_percentile,
                return_raw=True,
                blur_override=blur_override,
            )
        else:
            maps, scores, raw_maps = self.model.predict(
                tensor.to(self.device),
                score_percentile=effective_percentile,
                return_raw=True,
            )
        amap = maps[0]
        amap_raw = raw_maps[0]
        array = tensor[0]
        image = viz.tensor_to_image(array, denormalize=self.backend != "cae")
        resized_map = cv2.resize(amap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        resized_raw = cv2.resize(amap_raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        score = float(scores[0])
        heatmap = viz.map_to_heatmap(resized_map)
        overlay = viz.overlay_heatmap(image, heatmap, 0.5)
        return {
            "score": float(score),
            "anomaly_map": resized_map,
            "anomaly_map_raw": resized_raw,
            "heatmap": heatmap,
            "overlay": overlay,
        }

    def predict_array(self, array, score_percentile=None, blur_override=None):
        tensor = self.preprocess_array(array)
        result = self.predict_tensor(tensor, score_percentile=score_percentile, blur_override=blur_override)
        return result
