import sys
sys.path.append('/Users/akihiro/cat-age-cnn')

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from data.dataloader import train_dataloader, val_dataloader
import os
import matplotlib.pyplot as plt

device = torch.device("mps")

dataset_sizes = {
    "train": len(train_dataloader.dataset),
    "val": len(val_dataloader.dataset),
}

num_classes = 23

# === ResNet18 に変更 ===
model_ft = models.resnet18(weights="IMAGENET1K_V1")
for param in model_ft.parameters():
    param.requires_grad = True

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, 23)
)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00001)
scheduler = StepLR(optimizer_ft, step_size=10, gamma=0.98)

train_loss_history = []
val_loss_history = []
train_mae_history = []
val_mae_history = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=150):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_mae = 0

            dataloader = train_dataloader if phase == "train" else val_dataloader
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_mae += torch.sum(torch.abs(preds.float() - labels.data.float()))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mae = running_mae / dataset_sizes[phase]

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_mae_history.append(epoch_mae.item())
            else:
                val_loss_history.append(epoch_loss)
                val_mae_history.append(epoch_mae.item())

            print(f"{phase} Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}")

        scheduler.step()
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=150)

def plot_training():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_mae_history, label="Train MAE")
    plt.plot(val_mae_history, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    filename = f"outputs/logs/training_curve_{timestamp}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Training curve saved: {filename}")

plot_training()

def save_model_with_timestamp(model, directory="outputs/checkpoints"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(directory, f"resnet18_cat_age_{timestamp}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"モデルが正常に保存されました: {checkpoint_path}")

save_model_with_timestamp(model_ft)
print("トレーニング完了！")
