import base64
import io

import cv2
import numpy as np
import torch


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return array


def map_to_heatmap(anomaly_map):
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image, heatmap, alpha):
    image = np.clip(image, 0, 255).astype(np.uint8)
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)


def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def encode_png_base64(image):
    success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def decode_png_base64(data):
    binary = base64.b64decode(data)
    array = np.frombuffer(binary, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_array_base64(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_array_base64(data):
    binary = base64.b64decode(data)
    buffer = io.BytesIO(binary)
    buffer.seek(0)
    return np.load(buffer, allow_pickle=False)
