import itertools
import json
import subprocess
import sys
import yaml
from pathlib import Path

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f)

import os

def run_command(cmd):
    print(f"Running: {cmd}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    subprocess.check_call(cmd, shell=True, env=env)

def main():
    base_config_path = "configs/default.yaml"
    temp_config_path = "configs/optimize_temp.yaml"
    base_config = load_config(base_config_path)
    
    category = "bottle"
    base_config["category"] = category
    
    # Define search space
    padim_backbones = ["resnet50", "wide_resnet50_2", "efficientnet_b4"]
    # For CAE, we vary latent size and rotation augmentation
    cae_configs = [
        {"latent_channels": 128, "rotation": 5},
        {"latent_channels": 256, "rotation": 5},
        {"latent_channels": 256, "rotation": 15},
        {"latent_channels": 512, "rotation": 5},
    ]

    results = []

    print("=== Optimizing PaDiM ===")
    best_padim_score = -1
    best_padim_cfg = None

    for backbone in padim_backbones:
        print(f"\n--- Testing PaDiM with {backbone} ---")
        cfg = base_config.copy()
        cfg["padim"]["backbone"] = backbone
        
        save_config(cfg, temp_config_path)
        
        try:
            # Train
            run_command(f"{sys.executable} src/train/fit_padim.py --config {temp_config_path}")
            
            # Evaluate
            backend_name = f"padim_{backbone}"
            run_command(f"{sys.executable} src/eval/evaluate.py --config {temp_config_path} --backend {backend_name}")
            
            # Read metrics
            report_path = Path("reports") / backend_name / category / "metrics.json"
            if report_path.exists():
                with open(report_path, "r") as f:
                    metrics = json.load(f)
                score = metrics["image_auroc"] + metrics["pixel_auroc"]
                print(f"Score: {score} (Image AUROC: {metrics['image_auroc']}, Pixel AUROC: {metrics['pixel_auroc']})")
                results.append({"model": "padim", "backbone": backbone, "metrics": metrics, "score": score})
                
                if score > best_padim_score:
                    best_padim_score = score
                    best_padim_cfg = backbone
            else:
                print("Error: Report not found.")
        except Exception as e:
            print(f"Failed for {backbone}: {e}")

    print(f"Best PaDiM Backbone: {best_padim_cfg} (Score: {best_padim_score})")

    print("\n=== Optimizing CAE ===")
    best_cae_score = -1
    best_cae_cfg = None

    for c_conf in cae_configs:
        lc = c_conf["latent_channels"]
        rot = c_conf["rotation"]
        print(f"\n--- Testing CAE with Latent={lc}, Rotation={rot} ---")
        cfg = base_config.copy()
        cfg["cae"]["latent_channels"] = lc
        cfg["rotation"] = rot
        # Reduced epochs for optimization speed, user can train longer later
        cfg["cae"]["epochs"] = 30 
        
        save_config(cfg, temp_config_path)
        
        try:
            # Train
            run_command(f"{sys.executable} src/train/train_cae.py --config {temp_config_path}")
            
            # Evaluate
            run_command(f"{sys.executable} src/eval/evaluate.py --config {temp_config_path} --backend cae")
            
            # Read metrics
            report_path = Path("reports") / "cae" / category / "metrics.json"
            if report_path.exists():
                with open(report_path, "r") as f:
                    metrics = json.load(f)
                score = metrics["image_auroc"] + metrics["pixel_auroc"]
                print(f"Score: {score} (Image AUROC: {metrics['image_auroc']}, Pixel AUROC: {metrics['pixel_auroc']})")
                results.append({"model": "cae", "config": c_conf, "metrics": metrics, "score": score})
                
                if score > best_cae_score:
                    best_cae_score = score
                    best_cae_cfg = c_conf
            else:
                print("Error: Report not found.")
        except Exception as e:
            print(f"Failed for CAE {c_conf}: {e}")

    print(f"Best CAE Config: {best_cae_cfg} (Score: {best_cae_score})")
    
    # Save best results summary
    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
