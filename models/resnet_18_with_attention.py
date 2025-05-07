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
from models.se_resnet import load_pretrained_weights, se_resnet18

device = torch.device("mps")

dataset_sizes = {
    "train": len(train_dataloader.dataset),
    "val": len(val_dataloader.dataset),
}

num_classes = 23

# === ResNet18 に変更 ===
model_ft = se_resnet18(num_classes=23).to(device)
for param in model_ft.parameters():
    param.requires_grad = True
model_ft = load_pretrained_weights(model_ft)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
scheduler = StepLR(optimizer_ft, step_size=1, gamma=0.99)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0

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
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        scheduler.step()
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=25)

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
    plt.plot(train_acc_history, label="Train Acc")
    plt.plot(val_acc_history, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    filename = f"outputs/logs/training_curve_{timestamp}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Training curve saved: {filename}")

plot_training()

def save_model_with_timestamp(model, directory="outputs/checkpoints"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(directory, f"resnet18_cat_age_se_{timestamp}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"モデルが正常に保存されました: {checkpoint_path}")

save_model_with_timestamp(model_ft)
print("トレーニング完了！")
