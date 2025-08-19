import torch
import numpy as np
from tqdm import tqdm
import os

def evaluate_and_save(model, dataloader, device, save_dir, dataset_type, post_process_mask, visualize_and_save):
    """
    Evaluate the model on the dataset and save predictions.
    """
    model.eval()
    total_dice_before = 0
    total_dice_after = 0
    num_samples = 0
    smooth = 1e-6

    os.makedirs(save_dir, exist_ok=True)
    print(f"{dataset_type} set predictions will be saved to: {save_dir}")

    with torch.no_grad():
        for i, (images, gt_masks) in enumerate(tqdm(dataloader, desc=f"{dataset_type} Set Prediction")):
            images = images.to(device)
            gt_masks_np = gt_masks.cpu().numpy()

            outputs = model(images)['out']
            preds_raw = outputs.argmax(dim=1).cpu().numpy()
            preds_post = np.array([post_process_mask(pred) for pred in preds_raw])

            for j in range(images.shape[0]):
                image_idx = i * dataloader.batch_size + j

                gt = np.squeeze(gt_masks_np[j]).flatten()
                pred_before = np.squeeze(preds_raw[j]).flatten()
                pred_after = np.squeeze(preds_post[j]).flatten()

                intersection_before = (pred_before * gt).sum()
                total_dice_before += (2. * intersection_before + smooth) / (pred_before.sum() + gt.sum() + smooth)

                intersection_after = (pred_after * gt).sum()
                total_dice_after += (2. * intersection_after + smooth) / (pred_after.sum() + gt.sum() + smooth)

                num_samples += 1

                save_path = os.path.join(save_dir, f"{dataset_type.lower()}_prediction_{image_idx+1}.png")
                visualize_and_save(
                    processed_img=images[j],
                    gt_mask=gt_masks_np[j],
                    pred_raw=preds_raw[j],
                    pred_post=preds_post[j],
                    save_path=save_path,
                    title=f"{dataset_type} Set - Prediction {image_idx+1}"
                )

    avg_dice_before = total_dice_before / num_samples
    avg_dice_after = total_dice_after / num_samples
    print(f"\n--- {dataset_type} Set Evaluation Complete ---")
    print(f"Total {dataset_type} Images Processed: {num_samples}")
    print(f"Average Dice (Before Post-Processing): {avg_dice_before:.4f}")
    print(f"Average Dice (After Post-Processing): {avg_dice_after:.4f}")