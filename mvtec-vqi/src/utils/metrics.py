import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu


def auroc_image(scores, labels):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auroc_pixel(maps, masks):
    flat_maps = []
    flat_masks = []
    for amap, mask in zip(maps, masks):
        flat_maps.append(amap.reshape(-1))
        flat_masks.append(mask.reshape(-1))
    flat_maps = np.concatenate(flat_maps)
    flat_masks = np.concatenate(flat_masks)
    if len(np.unique(flat_masks)) < 2:
        return float("nan")
    return float(roc_auc_score(flat_masks, flat_maps))


def dice_score(binary_pred, binary_target):
    intersection = 2.0 * np.sum(binary_pred * binary_target)
    denom = np.sum(binary_pred) + np.sum(binary_target)
    if denom == 0:
        return 1.0
    return float(intersection / denom)


def threshold_map_otsu(anomaly_map):
    values = anomaly_map.reshape(-1)
    if np.allclose(values.max(), values.min()):
        return float(values.max())
    return float(threshold_otsu(values))


def threshold_map_percentile(anomaly_map, percentile):
    values = anomaly_map.reshape(-1)
    percentile_value = max(0.0, min(100.0, percentile * 100.0))
    return float(np.percentile(values, percentile_value))


def dice_at_threshold(anomaly_map, mask, threshold):
    pred = (anomaly_map >= threshold).astype(np.uint8)
    target = (mask > 0.5).astype(np.uint8)
    return dice_score(pred, target)
