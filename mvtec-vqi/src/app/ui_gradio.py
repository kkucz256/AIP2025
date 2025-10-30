import argparse
import os
import threading
import time

import gradio as gr
import numpy as np
import requests

from src.utils import viz
from src.utils.common import load_config, resolve_path, set_seed


def launch_api(host, port, config_path):
    os.environ["MVTEC_CONFIG"] = str(resolve_path(config_path))
    import uvicorn
    from src.app.api import app

    uvicorn.run(app, host=host, port=port, log_level="info")


def encode_image(image):
    return viz.encode_png_base64(image)


def decode_response(data):
    return {
        "heatmap": viz.decode_png_base64(data["heatmap"]),
        "overlay": viz.decode_png_base64(data["overlay"]),
        "map": viz.decode_array_base64(data["map"]),
        "score": float(data["score"]),
    }


def call_api(image, backend, category, api_url):
    payload = {
        "backend": backend,
        "category": category,
        "image_base64": encode_image(image),
    }
    response = requests.post(api_url, json=payload, timeout=30)
    response.raise_for_status()
    return decode_response(response.json())


def prepare_overlay(image, heatmap, amap, threshold):
    threshold = float(max(0.0, min(1.0, threshold)))
    
    # Resize heatmap and amap to match the input image dimensions
    target_height, target_width = image.shape[0], image.shape[1]
    resized_heatmap = viz.resize_image(heatmap, target_width, target_height)
    resized_map = viz.resize_image(amap, target_width, target_height)
    
    mask = (resized_map >= threshold).astype(np.uint8)
    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
    focused_heatmap = resized_heatmap.copy()
    focused_heatmap[mask_rgb == 0] = image[mask_rgb == 0]
    overlay = viz.overlay_heatmap(image, focused_heatmap, 0.6)
    return overlay


def process_stream(frame, backend, category, threshold, state, api_url, min_interval):
    if frame is None:
        return state["overlay"], state["heatmap"], state["score"], state["fps"], state
    frame = frame.astype(np.uint8)
    now = time.time()
    if now - state["last_time"] < min_interval:
        return state["overlay"], state["heatmap"], state["score"], state["fps"], state
    try:
        result = call_api(frame, backend, category, api_url)
        overlay = prepare_overlay(frame, result["heatmap"], result["map"], threshold)
        fps = 1.0 / max(now - state["last_time"], 1e-6)
        new_state = {
            "last_time": now,
            "overlay": overlay,
            "heatmap": result["heatmap"],
            "score": result["score"],
            "fps": fps,
        }
        return overlay, result["heatmap"], result["score"], fps, new_state
    except Exception:
        return state["overlay"], state["heatmap"], state["score"], state["fps"], state


def process_image(image, backend, category, threshold, api_url):
    if image is None:
        return None, None, 0.0
    image = image.astype(np.uint8)
    result = call_api(image, backend, category, api_url)
    overlay = prepare_overlay(image, result["heatmap"], result["map"], threshold)
    return overlay, result["heatmap"], result["score"]


def build_ui(config, api_host, api_port):
    api_url = f"http://127.0.0.1:{api_port}/infer"
    min_interval = 1.0 / max(1, config["app"].get("max_fps", 8))
    backends = ["padim_resnet50", "cae"]
    categories = sorted([config.get("category", "bottle")])
    with gr.Blocks(title="Wizualna Inspekcja Jakości") as demo:
        backend_input = gr.Dropdown(choices=backends, value=backends[0], label="Backend")
        category_input = gr.Textbox(value=config.get("category", "bottle"), label="Kategoria")
        threshold_input = gr.Slider(minimum=0.1, maximum=0.99, value=0.5, step=0.01, label="Próg anomalii")
        with gr.Tab("Kamera"):
            cam = gr.Image(sources=["webcam"], streaming=True)
            overlay_out = gr.Image(label="Nakładka")
            heatmap_out = gr.Image(label="Heatmapa")
            score_out = gr.Number(label="Score")
            fps_out = gr.Number(label="FPS")
            state = gr.State({
                "last_time": 0.0,
                "overlay": None,
                "heatmap": None,
                "score": 0.0,
                "fps": 0.0,
            })
            cam.stream(
                fn=lambda frame, backend, category, threshold, state: process_stream(
                    frame,
                    backend,
                    category,
                    threshold,
                    state,
                    api_url,
                    min_interval,
                ),
                inputs=[cam, backend_input, category_input, threshold_input, state],
                outputs=[overlay_out, heatmap_out, score_out, fps_out, state],
            )
        with gr.Tab("Plik"):
            image_input = gr.Image(type="numpy", label="Obraz")
            overlay_file = gr.Image(label="Nakładka")
            heatmap_file = gr.Image(label="Heatmapa")
            score_file = gr.Number(label="Score")
            analyze_button = gr.Button("Analizuj")
            analyze_button.click(
                fn=lambda image, backend, category, threshold: process_image(
                    image,
                    backend,
                    category,
                    threshold,
                    api_url,
                ),
                inputs=[image_input, backend_input, category_input, threshold_input],
                outputs=[overlay_file, heatmap_file, score_file],
            )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    api_host = config["app"].get("api_host", "0.0.0.0")
    api_port = config["app"].get("api_port", 8000)
    ui_host = config["app"].get("ui_host", "0.0.0.0")
    ui_port = config["app"].get("ui_port", 7860)
    api_thread = threading.Thread(target=launch_api, args=(api_host, api_port, args.config), daemon=True)
    api_thread.start()
    time.sleep(1.0)
    demo = build_ui(config, api_host, api_port)
    demo.launch(server_name=ui_host, server_port=ui_port, share=False)


if __name__ == "__main__":
    main()
