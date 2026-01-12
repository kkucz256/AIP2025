import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.mvtec import MVTecADDataset
from src.infer.predictor import AnomalyPredictor
from src.utils.common import get_device, resolve_path
from src.utils.transforms import build_transform, build_mask_transform
from src.utils.metrics import (
    auroc_image, 
    auroc_pixel, 
    threshold_map_otsu, 
    threshold_map_percentile, 
    dice_at_threshold,
    calculate_best_f1,
    compute_pro_score
)

class BenchmarkRunner:
    def __init__(self, config, backends, categories=None, device=None):
        self.config = config
        self.backends = backends
        self.device = device or get_device()
        self.data_dir = resolve_path(config.get("data_dir", "data/mvtec_ad"))
        
        # Auto-detect categories if not provided
        if categories is None:
             self.categories = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir() and d.name != "splits"])
        else:
            self.categories = categories

    def run(self):
        results = []
        for backend in self.backends:
            for category in self.categories:
                print(f"Benchmarking {backend} on {category}...")
                try:
                    res = self.benchmark_single(backend, category)
                    results.append(res)
                except Exception as e:
                    print(f"Failed to benchmark {backend} on {category}: {e}")
                    results.append({
                        "backend": backend,
                        "category": category,
                        "error": str(e)
                    })
        return results

    def benchmark_single(self, backend, category):
        # Setup similar to evaluate.py
        image_size = self.config.get("image_size", 256)
        transform = build_transform(image_size, normalize=backend != "cae")
        mask_transform = build_mask_transform(image_size)
        
        try:
            dataset = MVTecADDataset(
                root=self.data_dir,
                category=category,
                split="test",
                transform=transform,
                mask_transform=mask_transform
            )
        except Exception as e:
             raise RuntimeError(f"Could not load dataset: {e}")

        dataloader = DataLoader(
            dataset,
            batch_size=1, # Measure single image inference for timing accuracy
            shuffle=False,
            num_workers=self.config.get("num_workers", 2),
            pin_memory=self.device.type == "cuda"
        )

        predictor = AnomalyPredictor(
            backend=backend,
            category=category,
            config=self.config,
            device=self.device
        )

        image_scores = []
        image_labels = []
        maps = []
        masks = []
        inference_times = []

        # Warmup
        if len(dataset) > 0:
            dummy = dataset[0][0].unsqueeze(0)
            for _ in range(5):
                predictor.predict_tensor(dummy)

        for i, batch in enumerate(dataloader):
            images, labels, mask_tensors, _ = batch
            image_tensor = images[0].unsqueeze(0) # Batch size 1
            
            start_time = time.perf_counter()
            result = predictor.predict_tensor(image_tensor)
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000) # ms

            image_scores.append(result["score"])
            image_labels.append(int(labels[0]))
            maps.append(result["anomaly_map"])
            masks.append(mask_tensors[0].squeeze(0).numpy())

        # Calculate Metrics
        img_auroc = auroc_image(image_scores, image_labels)
        pix_auroc = auroc_pixel(maps, masks)
        
        # Advanced Metrics
        f1_max, precision_at_max, recall_at_max, best_threshold = calculate_best_f1(maps, masks)
        pro_score = compute_pro_score(maps, masks, threshold=best_threshold)
        
        # Dice scores
        dice_otsu_values = []
        dice_percent_values = []
        percentile = self.config["evaluation"].get("percentile_threshold", 0.9) 

        for amap, mask in zip(maps, masks):
            otsu = threshold_map_otsu(amap)
            dice_otsu_values.append(dice_at_threshold(amap, mask, otsu))
            
            thresh = threshold_map_percentile(amap, percentile)
            dice_percent_values.append(dice_at_threshold(amap, mask, thresh))
            
        avg_dice_otsu = float(np.mean(dice_otsu_values)) if dice_otsu_values else float('nan')
        avg_dice_percent = float(np.mean(dice_percent_values)) if dice_percent_values else float('nan')
        avg_inference_time = float(np.mean(inference_times)) if inference_times else float('nan')

        return {
            "backend": backend,
            "category": category,
            "image_auroc": img_auroc,
            "pixel_auroc": pix_auroc,
            "f1_max": f1_max,
            "precision": precision_at_max,
            "recall": recall_at_max,
            "pro_score": pro_score,
            "dice_otsu": avg_dice_otsu,
            "dice_percent": avg_dice_percent,
            "inference_time_ms": avg_inference_time,
            "num_samples": len(dataset)
        }
