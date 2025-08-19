import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

class UltrasoundNpyDataset(Dataset):
    def __init__(self, x_data, y_data, is_train=True):
        """
        Args:
            x_data (np.array): NumPy array of images (N, H, W, 1).
            y_data (np.array): NumPy array of masks (N, H, W, 1).
            is_train (bool): Kept for API consistency, no augmentations applied.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.is_train = is_train

        # Image-only transform (normalization for RGB)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Get raw numpy data
        image_np = self.x_data[idx]
        mask_np = self.y_data[idx]

        # Convert to (H, W, C) for PIL
        if image_np.shape[0] in [1, 3]:
            image_np = image_np.transpose(1, 2, 0)

        # Convert grayscale to RGB
        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)

        # Squeeze mask if needed (N, H, W, 1) -> (H, W)
        if len(mask_np.shape) == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np.squeeze(-1)

        # Convert to PIL Images
        image = Image.fromarray((image_np * 255).astype(np.uint8))
        mask = Image.fromarray(mask_np.astype(np.uint8))

        # Resize to fixed size
        image = F.resize(image, (256, 256))
        mask = F.resize(mask, (256, 256), interpolation=Image.NEAREST)

        # Convert to tensors
        image = self.image_transform(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask