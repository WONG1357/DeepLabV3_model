import matplotlib.pyplot as plt
import numpy as np

def visualize_and_save(processed_img, gt_mask, pred_raw, pred_post, save_path, title):
    """
    Plot and save comparison images: input, ground truth, raw prediction, and post-processed prediction.
    """
    if processed_img.shape[0] == 3:  # RGB
        image_np = processed_img.cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image_np = (image_np * 255).astype(np.uint8)
    else:  # Grayscale
        image_np = processed_img.cpu().squeeze(0).numpy() * 255

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    axes[2].imshow(pred_raw, cmap='gray')
    axes[2].set_title("Raw Prediction")
    axes[2].axis('off')
    axes[3].imshow(pred_post, cmap='gray')
    axes[3].set_title("Post-Processed Prediction")
    axes[3].axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)