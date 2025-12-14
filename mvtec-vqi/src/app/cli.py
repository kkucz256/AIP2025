import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.infer.predictor import AnomalyPredictor
from src.utils.common import get_device, load_config, resolve_path, set_seed
from src.utils.terminal import clear_screen, select_option


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def prompt_float(message: str, default: float) -> float:
    try:
        raw = input(f"{message} [{default}]: ").strip()
        if not raw:
            return default
        return float(raw.replace(",", "."))
    except ValueError:
        print("Nieprawidlowa wartosc, pozostawiam poprzednia.")
        return default


class PredictorCache:
    def __init__(self, config):
        self.config = config
        self._cache: Dict[Tuple[str, str, str], AnomalyPredictor] = {}

    def get(self, backend: str, category: str, device, weights_path: Path = None, config_override=None) -> AnomalyPredictor:
        key = (backend, category, str(device))
        if key not in self._cache:
            cfg = config_override or self.config
            predictor = AnomalyPredictor(backend=backend, category=category, config=cfg, device=device)
            
            if weights_path and weights_path.exists():
                print(f"DEBUG: Ladowanie wag z {weights_path}")
                if hasattr(predictor, "model") and hasattr(predictor.model, "load"):
                     predictor.model.load(str(weights_path))
                elif hasattr(predictor, "load"):
                     predictor.load(str(weights_path))
                else:
                    print("ERROR: Predictor nie ma metody load!")
            else:
                print(f"WARNING: Nie znaleziono wag w {weights_path}! Uzywam losowego modelu (bez sensu).")
            
            self._cache[key] = predictor
        return self._cache[key]

    def clear(self):
        self._cache.clear()


class TerminalApp:
    def __init__(self, config_path: Path, device_request: Optional[str] = None):
        self.config = load_config(config_path)
        set_seed(self.config.get("seed", 42))
        self.data_root = resolve_path(self.config.get("data_dir", "data/mvtec_ad"))
        self.device = get_device(device_request)
        self.device_request = device_request
        self.cache = PredictorCache(self.config)
        self.state: Dict[str, Optional[object]] = {
            "backend": "padim_resnet50",
            "category": self.config.get("category", "bottle"),
            "image_path": None,
        }
        self.backend_params: Dict[str, Dict[str, float]] = {
            "padim_resnet50": self._backend_defaults("padim_resnet50"),
            "cae": self._backend_defaults("cae"),
        }
        self.common_params = {
            "save_overlay": bool(self.config.get("infer", {}).get("save_overlay", True)),
            "overlay_dir": resolve_path(self.config.get("infer", {}).get("overlay_dir", "reports/overlays_cli")),
        }

    def _backend_defaults(self, backend: str) -> Dict[str, float]:
        infer_cfg = self.config.get("infer", {})
        backend_cfg = infer_cfg.get("backend_defaults", {}).get(backend, {})
        threshold = backend_cfg.get("threshold", infer_cfg.get("threshold", 0.55))
        percentile = backend_cfg.get("score_percentile", infer_cfg.get("score_percentile", 0.9))
        params = {
            "threshold": float(threshold),
            "score_percentile": float(percentile),
        }
        if backend == "padim_resnet50":
            params["gaussian_kernel"] = self.config["padim"]["gaussian_kernel"]
            params["blur_sigma"] = self.config["padim"]["blur_sigma"]
        return params

    def run(self):
        while True:
            choice = self.show_main_menu()
            if choice == "model":
                self.choose_model()
            elif choice == "image":
                self.choose_image()
            elif choice == "params":
                self.configure_params()
            elif choice == "analyze":
                self.run_inference()
            elif choice == "exit" or choice is None:
                clear_screen()
                print("Zakonczono.")
                return

    def show_main_menu(self):
        image_label = "brak"
        if self.state["image_path"]:
            try:
                image_label = str(Path(self.state["image_path"]).relative_to(self.data_root))
            except Exception:
                image_label = str(self.state["image_path"])
        params = self.backend_params[self.state["backend"]]
        summary = f"prog={params['threshold']:.2f}, perc={params['score_percentile']:.2f}"
        options = [
            (f"1. Wybor modelu ({self.state['backend']})", "model"),
            (f"2. Wybor zdjecia ({image_label})", "image"),
            (f"3. Parametry analizy ({summary})", "params"),
            ("4. Uruchom analize", "analyze"),
            ("Wyjscie", "exit"),
        ]
        footer = f"Aktualne urzadzenie: {self.device} | Dane: {self.data_root}"
        return select_option("=== MVTec VQI - tryb terminalowy ===", options, footer=footer)

    def choose_model(self):
        options = [
            ("PaDiM (ResNet50) - szybki, statystyczny", "padim_resnet50"),
            ("CAE (autoenkoder) - dokladniejszy, wolniejszy", "cae"),
            ("Instrukcje modeli", "__help__"),
            ("Cofnij", "__back__"),
        ]
        while True:
            selection = select_option(
                "Wybierz model (aktywny oznaczony jako (current))",
                options,
                current_value=self.state["backend"],
            )
            if selection in (None, "__back__"):
                return
            if selection == "__help__":
                self._show_model_help()
                continue
            self.state["backend"] = selection
            self.cache.clear()
            return

    def _show_model_help(self):
        clear_screen()
        print("=== Instrukcje modeli ===\n")
        print("PaDiM (ResNet50):")
        print("- Model statystyczny na cechach z sieci ResNet50 pretrained.")
        print("- Szybki inference, niskie wymagania sprzetowe.")
        print("- Wygladzanie mapy Gaussa (kernel, sigma) pomaga redukowac szum.")
        print("\nCAE (Convolutional Autoencoder):")
        print("- Uczy sie rekonstrukcji obrazow dobrych; roznica = mapa anomalii.")
        print("- Dokladniejszy na zlozonych defektach, wolniejszy od PaDiM.")
        print("- Wynik bazuje na percentylu mapy (domyslnie 0.9) jako skorelowany z defektem.")
        print("\nWybierz model zaleznie od potrzeb: PaDiM dla szybkosci, CAE dla jakosci.")
        input("\nEnter, aby wrocic do menu wyboru modelu...")

    def _list_categories(self):
        if not self.data_root.exists():
            return []
        return sorted([d for d in self.data_root.iterdir() if d.is_dir()])

    def _list_images(self, base: Path):
        images = []
        for path in sorted(base.glob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            images.append(path)
        return images

    def _list_defect_types(self, category_path: Path):
        test_root = category_path / "test"
        if not test_root.exists():
            return []
        return [d for d in sorted(test_root.iterdir()) if d.is_dir()]

    def choose_image(self):
        if not self.data_root.exists():
            input(f"Brak katalogu danych: {self.data_root}. Nacisnij Enter, aby wrocic.")
            return
        categories = self._list_categories()
        if not categories:
            input("Nie znaleziono folderow kategorii w data/mvtec_ad. Nacisnij Enter.")
            return
        cat_options = [(cat.name, cat) for cat in categories]
        cat_options.append(("Cofnij", "__back__"))
        category = select_option(
            "Wybierz kategorie produktu",
            cat_options,
            current_value=next((c for c in categories if c.name == self.state.get("category")), None),
        )
        if not category or category == "__back__":
            return
        defect_types = self._list_defect_types(category)
        if not defect_types:
            input("Brak podfolderow w test/. Enter aby wrocic.")
            return
        type_options = [(dt.name, dt) for dt in defect_types]
        type_options.append(("Cofnij", "__back__"))
        defect_type = select_option(
            "Wybierz typ (test/...)",
            type_options,
        )
        if not defect_type or defect_type == "__back__":
            return
        images = self._list_images(defect_type)
        if not images:
            input("Brak obrazow w wybranej kategorii. Enter aby wrocic.")
            return
        selection = self._select_image_with_pagination(images)
        if selection:
            self.state["image_path"] = selection
            try:
                self.state["category"] = Path(selection).relative_to(self.data_root).parts[0]
            except Exception:
                self.state["category"] = category.name

    def _select_image_with_pagination(self, images):
        page_size = 15
        items = [(path.name, path) for path in images]
        total_pages = max(1, (len(items) + page_size - 1) // page_size)
        page = 0
        current = self.state.get("image_path")
        while True:
            start = page * page_size
            end = min(start + page_size, len(items))
            subset = items[start:end]
            options = list(subset)
            if total_pages > 1:
                options.append((f"Nastepna strona ({page + 1}/{total_pages})", "__next__"))
                options.append((f"Poprzednia strona ({page + 1}/{total_pages})", "__prev__"))
            options.append(("Cofnij", "__back__"))
            selection = select_option(
                "Wybierz obraz do analizy (15 pozycji na strone)",
                options,
                current_value=current,
                footer=f"Obrazy: {len(items)} | Strona {page + 1}/{total_pages}",
            )
            if selection in (None, "__back__"):
                return None
            if selection == "__next__":
                page = (page + 1) % total_pages
                continue
            if selection == "__prev__":
                page = (page - 1 + total_pages) % total_pages
                continue
            return selection

    def configure_params(self):
        while True:
            params = self.backend_params[self.state["backend"]]
            overlay_state = "ON" if self.common_params["save_overlay"] else "OFF"
            options = [
                (f"Prog klasyfikacji (aktualnie {params['threshold']:.2f})", "threshold"),
                (f"Percentyl mapy anomalii (aktualnie {params['score_percentile']:.2f})", "percentile"),
            ]
            if self.state["backend"] == "padim_resnet50":
                options.append(
                    (f"Wygladzanie PaDiM kernel={params['gaussian_kernel']} sigma={params['blur_sigma']}", "smoothing")
                )
            options.extend(
                [
                    (f"Zapis zrzutow {overlay_state}", "overlay"),
                    (f"Urzadzenie: {self.device}", "device"),
                    ("Instrukcje parametrow", "__help__"),
                    ("Cofnij", "__back__"),
                ]
            )
            selection = select_option("Parametry analizy", options)
            if selection in (None, "__back__"):
                return
            if selection == "__help__":
                self._show_params_help()
                continue
            if selection == "threshold":
                params["threshold"] = prompt_float("Podaj nowy prog anomalii (0-1)", params["threshold"])
            elif selection == "percentile":
                value = prompt_float("Percentyl do obliczenia wyniku (0-1)", params["score_percentile"])
                params["score_percentile"] = min(max(value, 0.0), 1.0)
            elif selection == "smoothing" and self.state["backend"] == "padim_resnet50":
                params["gaussian_kernel"] = int(
                    prompt_float("Rozmiar jadra Gaussa dla PaDiM (nieparzyste)", params["gaussian_kernel"])
                )
                params["blur_sigma"] = prompt_float("Sigma Gaussa dla PaDiM", params["blur_sigma"])
            elif selection == "overlay":
                self.common_params["save_overlay"] = not self.common_params["save_overlay"]
            elif selection == "device":
                self.device = self._choose_device()
                self.cache.clear()

    def _show_params_help(self):
        clear_screen()
        print("=== Instrukcje parametrow ===\n")
        print("Prog klasyfikacji:")
        print("- Wynik powyzej progu = DEFEKT, ponizej = OK.")
        print("Percentyl mapy anomalii:")
        print("- Wynik to percentyl z wartosci mapy (domyslnie 0.9); wyzszy = bardziej konserwatywny.")
        print("Wygladzanie PaDiM (kernel, sigma):")
        print("- Gauss redukuje szum na mapie; wiekszy kernel/sigma = gladsza mapa, mniejsza czulosc na drobne defekty.")
        print("Zapis zrzutow:")
        print("- Jesli ON, overlay i heatmapa sa zapisywane w katalogu overlay (nadpisywane).")
        print("Urzadzenie:")
        print("- Wybor CPU/CUDA/MPS lub auto (preferuje GPU).")
        input("\nEnter, aby wrocic do parametrow...")

    def _choose_device(self):
        options = [
            ("Auto (CUDA>MPS>CPU)", "auto"),
            ("CPU", "cpu"),
            ("CUDA", "cuda"),
            ("MPS (Apple)", "mps"),
            ("Cofnij", "__back__"),
        ]
        selection = select_option("Wybierz urzadzenie obliczeniowe", options)
        if selection in (None, "__back__"):
            return self.device
        self.device_request = selection if selection != "auto" else None
        return get_device(self.device_request)

    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _save_visuals(self, overlay, heatmap, source_path: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.state["backend"]
        source_name = source_path.stem
        
        if not self.common_params["save_overlay"]:
            return None, None
        out_dir = Path(self.common_params["overlay_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        filename_base = f"{timestamp}_{model_name}_{source_name}"
        overlay_path = out_dir / f"{filename_base}_overlay.png"
        heatmap_path = out_dir / f"{filename_base}_heatmap.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        
        return overlay_path, heatmap_path

    def run_inference(self):
        if not self.state["image_path"]:
            input("Najpierw wybierz zdjecie. Enter aby wrocic.")
            return
        backend = self.state["backend"]
        category = self.state["category"]
        params = self.backend_params[backend]
        project_root = Path.cwd()
        weights_path = project_root / "artifacts" / backend / category / "model.pt"
        try:
            predictor = self.cache.get(backend, category, self.device, weights_path=weights_path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            input(f"Nie udalo sie zaladowac modelu ({exc}). Enter aby wrocic.")
            return
        try:
            image_array = self._load_image(Path(self.state["image_path"]))
        except FileNotFoundError:
            input("Wybrane zdjecie nie istnieje. Enter aby wrocic.")
            return
        blur_override = None
        if backend == "padim_resnet50":
            blur_override = {
                "gaussian_kernel": params["gaussian_kernel"],
                "blur_sigma": params["blur_sigma"],
            }
        result = predictor.predict_array(
            image_array,
            score_percentile=params["score_percentile"],
            blur_override=blur_override,
        )
        score = result["score"]
        is_anomaly = score >= params["threshold"]
        overlay_path, heatmap_path = self._save_visuals(result["overlay"], result["heatmap"], Path(self.state["image_path"]))
        clear_screen()
        status = "DEFEKT" if is_anomaly else "OK"
        print("=== Wynik analizy ===")
        print(f"Model: {backend} | Kategoria: {category} | Urzadzenie: {self.device}")
        print(f"Wynik: {score:.4f}  | Prog: {params['threshold']:.4f}  | Klasyfikacja: {status}")
        if overlay_path:
            print(f"Overlay zapisany w: {overlay_path}")
            print(f"Mapa cieplna zapisana w: {heatmap_path}")
        else:
            print("Zapis wizualizacji: wylaczony")
        input("\nNacisnij Enter, aby wrocic do menu...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    app = TerminalApp(config_path=Path(args.config), device_request=args.device)
    try:
        app.run()
    except KeyboardInterrupt:
        clear_screen()
        print("Przerwano przez uzytkownika.")
        sys.exit(0)


if __name__ == "__main__":
    main()
