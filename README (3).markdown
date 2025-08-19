# DeepLabV3+ for Rectus Femoris Muscle Segmentation

This repository contains a PyTorch implementation of DeepLabV3+ for segmenting the rectus femoris muscle in ultrasound images. The code is modular, leveraging Google Drive for data storage and trained on `.npy` files containing ultrasound images and corresponding masks.

## Repository Structure

- `dataset.py`: Defines the `UltrasoundNpyDataset` class for loading and preprocessing ultrasound data.
- `utils.py`: Contains utility functions for calculating Dice scores and post-processing segmentation masks.
- `visualization.py`: Handles visualization and saving of input images, ground truth, and predictions.
- `train.py`: Implements the training loop for the DeepLabV3+ model.
- `evaluate.py`: Manages evaluation and saving of predictions for train and test sets.
- `main.py`: Orchestrates the entire pipeline, from data loading to training and evaluation.
- `README.md`: This file.

## Prerequisites

- Python 3.8+
- Google Colab with GPU support (recommended) or a local environment with CUDA-enabled GPU
- Google Drive account for data storage
- Required Python packages (see `requirements.txt`)

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install Dependencies**:
   Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place your `.npy` files (`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`) in a Google Drive folder, e.g., `/intern RF transverse latest file/`.
   - Ensure the ultrasound images (`X_*.npy`) and masks (`y_*.npy`) are in the format `(N, H, W, 1)` for grayscale images and masks.

4. **Mount Google Drive**:
   - If using Google Colab, the `main.py` script automatically mounts Google Drive.
   - If running locally, ensure Google Drive is accessible or modify `main.py` to use local paths.

## Usage

1. **Update File Paths**:
   - Open `main.py` and update the following paths to match your Google Drive structure:
     ```python
     base_path = '/content/drive/MyDrive/intern RF transverse latest file/'
     best_model_path = '/content/drive/MyDrive/internship models/deeplabv3+ resnet 50/rectus femoris/deeplabv3plus_resnet50_best.pth'
     base_save_dir = '/content/drive/MyDrive/internship models/deeplabv3+ resnet 50/rectus femoris/segmentation_results_with_preprocessing'
     ```
   - Ensure the directories for saving models and predictions exist or will be created (`os.makedirs` handles this).

2. **Run the Pipeline**:
   - Execute the main script to train the model, evaluate it, and save predictions:
     ```bash
     python main.py
     ```
   - The script will:
     - Load and preprocess data.
     - Train the DeepLabV3+ model for 50 epochs.
     - Save the best model based on validation Dice score.
     - Evaluate the model on train and test sets, saving predictions as PNG files.

3. **Outputs**:
   - **Model**: The best model is saved at `best_model_path`.
   - **Predictions**: Visualization images (input, ground truth, raw prediction, post-processed prediction) are saved in `base_save_dir/train_set_predictions/` and `base_save_dir/test_set_predictions/`.
   - **Metrics**: Training and validation losses, Dice scores, and test set evaluation metrics are printed to the console.

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
matplotlib==3.7.1
Pillow==9.5.0
scipy==1.10.1
tqdm==4.65.0
```

## Notes

- The model uses DeepLabV3+ with a ResNet-50 backbone, trained with SGD optimizer (learning rate: 0.01, momentum: 0.9, weight decay: 0.0005).
- Data is resized to 256x256 and normalized to match ImageNet statistics.
- Post-processing selects the largest connected component and fills holes in segmentation masks.
- Ensure sufficient Google Drive storage for saving models and predictions.
- For local execution, modify Google Drive paths and remove the `drive.mount` call if using local storage.

## License

This project is licensed under the MIT License.