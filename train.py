import torch
from torch.utils.data import DataLoader

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, best_model_path):
    """
    Train the DeepLabV3+ model and save the best model based on validation Dice score.
    """
    best_dice = 0.0
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_dataloader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)['out']
                val_loss += criterion(outputs, masks).item()
                dice_score = calculate_dice(outputs, masks)
                val_dice_scores.append(dice_score)

        val_loss = val_loss / max(len(val_dataloader), 1)
        val_dice = sum(val_dice_scores) / len(val_dice_scores) if val_dice_scores else 0.0

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Dice: {best_dice:.4f} at epoch {epoch+1}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        model.train()

    return model
