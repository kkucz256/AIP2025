import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class PaDiMModel:
    def __init__(self, image_size, selected_channels, gaussian_kernel, blur_sigma, device, seed=42):
        self.device = device
        self.image_size = image_size
        self.selected_channels = selected_channels
        self.gaussian_kernel = max(1, gaussian_kernel)
        if self.gaussian_kernel % 2 == 0:
            self.gaussian_kernel += 1
        self.blur_sigma = blur_sigma
        self.seed = seed
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        base.eval()
        for param in base.parameters():
            param.requires_grad_(False)
        self.extractor = create_feature_extractor(
            base,
            return_nodes={
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
            },
        ).to(self.device)
        self.mean = None
        self.cov_inv = None
        self.channel_indices = None
        self.feature_shape = None

    def _merge_features(self, features):
        x1 = features["layer1"]
        x2 = F.interpolate(features["layer2"], size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x3 = F.interpolate(features["layer3"], size=x1.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x1, x2, x3], dim=1)

    def fit(self, dataloader):
        self.extractor.eval()
        feature_maps = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device)
                outputs = self.extractor(images)
                merged = self._merge_features(outputs).cpu()
                feature_maps.append(merged)
        features = torch.cat(feature_maps, dim=0)
        total_channels = features.shape[1]
        if self.selected_channels > total_channels:
            self.selected_channels = total_channels
        generator = torch.Generator().manual_seed(self.seed)
        self.channel_indices = torch.randperm(total_channels, generator=generator)[: self.selected_channels]
        selected = features[:, self.channel_indices]
        n, c, h, w = selected.shape
        self.feature_shape = (h, w)
        selected = selected.permute(0, 2, 3, 1).reshape(n, -1, c)
        means = selected.mean(dim=0)
        residuals = selected - means.unsqueeze(0)
        residuals = residuals.to(torch.float64)
        covariance = torch.einsum("nsc,nsd->scd", residuals, residuals)
        denominator = max(n - 1, 1)
        covariance = covariance / denominator
        eye = torch.eye(c, dtype=torch.float64, device=covariance.device).unsqueeze(0)
        covariance = covariance + 0.01 * eye
        covariance_inv = torch.linalg.inv(covariance)
        self.mean = means.to(torch.float32).to(self.device)
        self.cov_inv = covariance_inv.to(torch.float32).to(self.device)
        self.channel_indices = self.channel_indices.to(torch.long).to(self.device)

    def predict(self, tensor):
        self.extractor.eval()
        with torch.no_grad():
            tensor = tensor.to(self.device)
            outputs = self.extractor(tensor)
            merged = self._merge_features(outputs)
            merged = merged.index_select(1, self.channel_indices)
            b, c, h, w = merged.shape
            merged = merged.permute(0, 2, 3, 1).reshape(b, -1, c)
            mean = self.mean.reshape(1, -1, c)
            cov_inv = self.cov_inv
            diff = merged - mean
            proj = torch.einsum("bsc,scd->bsd", diff, cov_inv)
            dist = (proj * diff).sum(dim=2)
            dist = torch.sqrt(torch.clamp(dist, min=1e-6))
            dist = dist.reshape(b, self.feature_shape[0], self.feature_shape[1])
            dist = dist.unsqueeze(1)
            dist = F.interpolate(dist, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        maps = []
        scores = []
        for sample in dist:
            amap = sample.squeeze(0).detach().cpu().numpy().astype(np.float32)
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
            if self.gaussian_kernel > 1:
                amap = cv2.GaussianBlur(amap, (self.gaussian_kernel, self.gaussian_kernel), self.blur_sigma)
            amap = np.clip(amap, 0.0, 1.0)
            maps.append(amap)
            scores.append(float(amap.max()))
        return maps, scores

    def save(self, path):
        state = {
            "mean": self.mean.detach().cpu(),
            "cov_inv": self.cov_inv.detach().cpu(),
            "channel_indices": self.channel_indices.detach().cpu(),
            "feature_shape": self.feature_shape,
            "image_size": self.image_size,
            "selected_channels": self.selected_channels,
            "gaussian_kernel": self.gaussian_kernel,
            "blur_sigma": self.blur_sigma,
            "seed": self.seed,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.mean = state["mean"].to(self.device)
        self.cov_inv = state["cov_inv"].to(self.device)
        self.channel_indices = state["channel_indices"].to(torch.long).to(self.device)
        self.feature_shape = tuple(state["feature_shape"])
        self.image_size = state["image_size"]
        self.selected_channels = state["selected_channels"]
        self.gaussian_kernel = state["gaussian_kernel"]
        self.blur_sigma = state["blur_sigma"]
        self.seed = state.get("seed", self.seed)
        self.extractor.eval()
