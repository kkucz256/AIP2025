import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from skimage.filters import threshold_otsu
from skimage import measure


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


def calculate_best_f1(maps, masks):
    """
    Calculates the best F1 score, precision, and recall by searching for the optimal threshold.
    Returns: (f1_max, precision, recall, optimal_threshold)
    """
    flat_maps = []
    flat_masks = []
    for amap, mask in zip(maps, masks):
        flat_maps.append(amap.reshape(-1))
        flat_masks.append(mask.reshape(-1))
    
    y_scores = np.concatenate(flat_maps)
    y_true = np.concatenate(flat_masks).astype(int)
    
    if len(np.unique(y_true)) < 2:
         return float("nan"), float("nan"), float("nan"), 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # F1 = 2 * (P * R) / (P + R)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.divide(numerator, denominator)
        f1_scores[denominator == 0] = 0.0

    idx = np.argmax(f1_scores)
    return (
        float(f1_scores[idx]),
        float(precisions[idx]),
        float(recalls[idx]),
        float(thresholds[idx]) if idx < len(thresholds) else float(thresholds[-1])
    )


def compute_pro_score(maps, masks, threshold=None):
    """
    Computes Per-Region Overlap (PRO) score.
    PRO is defined as the average coverage of each ground truth connected component by the prediction.
    If threshold is None, it uses Otsu's method per map (simplified) or requires a fixed threshold.
    For integration with PRO curve, one would typically scan thresholds.
    Here we calculate PRO at a specific threshold (e.g., from Best F1 or Otsu).
    """
    pro_values = []
    
    for amap, mask in zip(maps, masks):
        # Determine threshold if not provided (fallback to Otsu)
        th = threshold
        if th is None:
             th = threshold_map_otsu(amap)
             
        binary_pred = (amap >= th).astype(int)
        binary_mask = (mask > 0.5).astype(int)
        
        # Label connected components in Ground Truth
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        
        for region in regions:
            # For each defect region in GT
            # Calculate what fraction of this region is covered by prediction
            region_mask = (labeled_mask == region.label)
            pixel_count = np.sum(region_mask)
            
            if pixel_count == 0:
                continue
                
            intersection = np.sum(binary_pred & region_mask)
            pro = intersection / pixel_count
            pro_values.append(pro)
            
    if not pro_values:
        return 1.0 if len(masks) > 0 else 0.0 # If no defects exist, PRO is ideally 1 (perfect) or undefined.
        
    return float(np.mean(pro_values))


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
