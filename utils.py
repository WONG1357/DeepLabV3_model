import torch
import numpy as np
from scipy import ndimage

def calculate_dice(pred, target, num_classes=2, smooth=1e-6):
    """
    Calculate Dice score for a batch of predictions and targets.
    """
    pred = pred.argmax(dim=1)  # Convert logits to class predictions
    dice_scores = []
    for batch_idx in range(pred.size(0)):
        pred_flat = pred[batch_idx].flatten()
        target_flat = target[batch_idx].flatten()
        intersection = ((pred_flat == 1) & (target_flat == 1)).sum().item()
        pred_sum = (pred_flat == 1).sum().item()
        target_sum = (target_flat == 1).sum().item()
        # Handle edge case where both prediction and target are empty
        if pred_sum + target_sum == 0:
            dice = 1.0  # Perfect score if both are empty
        else:
            dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_scores.append(dice)
    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0

def post_process_mask(mask):
    """
    Applies post-processing to a binary segmentation mask by selecting the largest
    connected component and filling holes.
    """
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labels.ravel())
    if len(component_sizes) > 1:
        largest_component_label = component_sizes[1:].argmax() + 1
        processed_mask = (labels == largest_component_label)
        processed_mask = ndimage.binary_fill_holes(processed_mask)
        return processed_mask.astype(np.uint8)
    return mask
