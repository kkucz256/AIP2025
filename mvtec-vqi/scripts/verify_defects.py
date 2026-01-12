import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(".")

from src.infer.predictor import AnomalyPredictor
from src.utils.common import load_config, get_device

def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    category = "bottle"
    
    # We focus on the best model
    backend = "padim_wide_resnet50_2"
    
    print(f"Loading {backend}...")
    predictor = AnomalyPredictor(backend=backend, category=category, config=config, device=device)
    
    # Hard coded paths to some likely hard cases based on standard MVTec structure
    # broken_large, broken_small, contamination
    test_root = Path("data/mvtec_ad/bottle/test")
    
    cases = [
        ("contamination", test_root / "contamination"),
        ("broken_small", test_root / "broken_small"),
        ("good", test_root / "good") # To check false positives
    ]

    print("\n=== Testing Hard Cases ===")
    
    # Check current config defaults
    default_threshold = config["infer"]["backend_defaults"][backend]["threshold"]
    default_percentile = config["infer"]["backend_defaults"][backend]["score_percentile"]
    
    print(f"Config Threshold: {default_threshold}")
    print(f"Config Percentile: {default_percentile}")

    results = []

    for name, path in cases:
        if not path.exists():
            print(f"Skipping {name}, path not found: {path}")
            continue
            
        print(f"\n--- {name} ---")
        image_files = list(path.glob("*.png"))[:5] # Test first 5 of each
        
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            
            # Predict with config settings
            res = predictor.predict_array(arr, score_percentile=default_percentile)
            score = res["score"]
            status = "DEFECT" if score >= default_threshold else "OK"
            
            print(f"{img_path.name}: Score={score:.4f} (p={default_percentile}) -> {status}")
            results.append({"type": name, "score": score, "path": img_path})

    # Summary analysis
    goods = [r["score"] for r in results if r["type"] == "good"]
    defects = [r["score"] for r in results if r["type"] != "good"]
    
    max_good = max(goods) if goods else 0
    min_defect = min(defects) if defects else 0
    
    print("\n=== Analysis ===")
    print(f"Max 'Good' Score: {max_good:.4f}")
    print(f"Min 'Defect' Score: {min_defect:.4f}")
    
    if max_good < min_defect:
        suggested_threshold = (max_good + min_defect) / 2
        print(f"Separation possible! Suggested Threshold: {suggested_threshold:.4f}")
    else:
        print("Overlapping scores. Threshold separation difficult with current parameters.")

if __name__ == "__main__":
    main()
