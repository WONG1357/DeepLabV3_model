import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import os
from google.colab import drive
from dataset import UltrasoundNpyDataset
from utils import calculate_dice, post_process_mask
from visualization import visualize_and_save
from train import train_model
from evaluate import evaluate_and_save

# Mount Google Drive
drive.mount('/content/drive')

# Step 1: Load Data
base_path = '/content/drive/MyDrive/intern RF transverse latest file/'
train_image_file = os.path.join(base_path, 'X_train.npy')
train_mask_file = os.path.join(base_path, 'y_train.npy')
val_image_file = os.path.join(base_path, 'X_val.npy')
val_mask_file = os.path.join(base_path, 'y_val.npy')
test_image_file = os.path.join(base_path, 'X_test.npy')
test_mask_file = os.path.join(base_path, 'y_test.npy')

try:
    x_train = np.load(train_image_file)
    y_train = np.load(train_mask_file)
    x_val = np.load(val_image_file)
    y_val = np.load(val_mask_file)
    x_test = np.load(test_image_file)
    y_test = np.load(test_mask_file)
except Exception as e:
    print(f"Error loading .npy files: {e}")
    raise

print("Train shapes:", x_train.shape, y_train.shape, np.unique(y_train))
print("Val shapes:", x_val.shape, y_val.shape, np.unique(y_val))
print("Test shapes:", x_test.shape, y_test.shape, np.unique(y_test))

# Create datasets
train_dataset = UltrasoundNpyDataset(x_train, y_train, is_train=True)
val_dataset = UltrasoundNpyDataset(x_val, y_val, is_train=False)
test_dataset = UltrasoundNpyDataset(x_test, y_test, is_train=False)

# Create DataLoaders
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

# Verify batch shapes
try:
    images, masks = next(iter(train_dataloader))
    print("Batch shapes - Images:", images.shape, "Masks:", masks.shape)
except Exception as e:
    print(f"Error in DataLoader: {e}")
    raise

# Step 2: Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(weights=None, num_classes=2)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# Step 3: Train Model
num_epochs = 50
best_model_path = '/content/drive/MyDrive/internship models/deeplabv3+ resnet 50/rectus femoris/deeplabv3plus_resnet50_best.pth'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, best_model_path)

# Step 4: Load Best Model for Evaluation
print("Loading the best model for evaluation...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
model.eval()

# Step 5: Evaluate and Save Predictions
base_save_dir = '/content/drive/MyDrive/internship models/deeplabv3+ resnet 50/rectus femoris/segmentation_results_with_preprocessing'
train_save_dir = os.path.join(base_save_dir, 'train_set_predictions')
test_save_dir = os.path.join(base_save_dir, 'test_set_predictions')

evaluate_and_save(model, train_dataloader, device, train_save_dir, "Train", post_process_mask, visualize_and_save)
evaluate_and_save(model, test_dataloader, device, test_save_dir, "Test", post_process_mask, visualize_and_save)

print("All predictions saved successfully to Google Drive.")