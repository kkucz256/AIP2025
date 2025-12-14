import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class ConvAutoencoder(nn.Module):
    def __init__(self, base_channels, latent_channels):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, latent_channels, 3, stride=2, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class CAEModel:
    def __init__(self, image_size, config, device):
        self.image_size = image_size
        self.config = dict(config)
        self.device = device
        self.score_percentile = float(self.config.get("score_percentile", 0.9))
        self.model = ConvAutoencoder(self.config["base_channels"], self.config["latent_channels"]).to(device)

    def _ssim_map(self, x, y):
        k1 = 0.01
        k2 = 0.03
        c1 = k1 * k1
        c2 = k2 * k2
        mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
        mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)
        sigma_x = F.avg_pool2d(x * x, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim = numerator / (denominator + 1e-8)
        ssim = torch.clamp(ssim, 0.0, 1.0)
        return ssim.mean(dim=1, keepdim=True)

    def _normalize_map(self, amap):
        min_val = amap.min()
        max_val = amap.max()
        return (amap - min_val) / (max_val - min_val + 1e-8)

    def fit(self, train_loader, val_loader=None, epochs=30):
        amp_enabled = self.device.type == "cuda" and self.config.get("use_amp", True)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        optimizer = Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        best_loss = float("inf")
        best_state = None
        for epoch in range(epochs):
            self.model.train()
            running = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device)
                optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=scaler.is_enabled()):
                    recon = self.model(images)
                    mse = F.mse_loss(recon, images)
                    ssim_map = self._ssim_map(images, recon)
                    ssim_loss = 1.0 - ssim_map.mean()
                    loss = self.config["loss_mse_weight"] * mse + self.config["loss_ssim_weight"] * ssim_loss
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                running += loss.item() * images.size(0)
            train_loss = running / len(train_loader.dataset)
            val_loss = train_loss
            if val_loader:
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch[0] if isinstance(batch, (list, tuple)) else batch
                        images = images.to(self.device)
                        with torch.autocast(device_type=self.device.type, enabled=scaler.is_enabled()):
                            recon = self.model(images)
                            mse = F.mse_loss(recon, images)
                            ssim_map = self._ssim_map(images, recon)
                            ssim_loss = 1.0 - ssim_map.mean()
                            value = self.config["loss_mse_weight"] * mse + self.config["loss_ssim_weight"] * ssim_loss
                        total += value.item() * images.size(0)
                val_loss = total / len(val_loader.dataset)
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            monitor = val_loss if val_loader else train_loss
            if monitor < best_loss:
                best_loss = monitor
                best_state = self.model.state_dict()
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, tensor, score_percentile=None, return_raw=False):
        #test
        self.model.eval()
        percentile = self.score_percentile if score_percentile is None else float(score_percentile)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda" and self.config.get("use_amp", True)
        ):
            tensor = tensor.to(self.device)
            recon = self.model(tensor)
            mse_map = torch.mean((tensor - recon) ** 2, dim=1, keepdim=True)
            ssim_map = self._ssim_map(tensor, recon)
            anomaly_map = self.config["loss_mse_weight"] * mse_map + self.config["loss_ssim_weight"] * (1.0 - ssim_map)
        maps = []
        raw_maps = []
        scores = []
        for amap in anomaly_map:
            amap_np = amap.squeeze(0).detach().cpu().numpy().astype(np.float32)
            margin = 16
            h, w = amap_np.shape
            if margin > 0 and h > 2*margin and w > 2*margin:
                amap_np[:margin, :] = 0
                amap_np[h-margin:, :] = 0
                amap_np[:, :margin] = 0
                amap_np[:, w-margin:] = 0

            raw_maps.append(amap_np)
            scores.append(float(np.quantile(amap_np, percentile)))
            norm_map = self._normalize_map(amap_np)
            norm_map = np.clip(norm_map, 0.0, 1.0)
            maps.append(norm_map)
        if return_raw:
            return maps, scores, raw_maps
        return maps, scores

    def save(self, path):
        self.config["score_percentile"] = self.score_percentile
        state = {
            "model": self.model.state_dict(),
            "config": self.config,
            "image_size": self.image_size,
            "score_percentile": self.score_percentile,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        if "config" in state:
            self.config.update(state["config"])
        self.image_size = state.get("image_size", self.image_size)
        self.score_percentile = float(state.get("score_percentile", self.config.get("score_percentile", 0.9)))
        self.model.to(self.device)
        self.model.eval()
