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

model_ft = models.resnet50(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)
model_ft = model_ft.to(device)

criterion = nn.MSELoss()
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)

scheduler = StepLR(optimizer_ft, step_size=10, gamma=0.1)

train_loss_history = []
val_loss_history = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            dataloader = train_dataloader if phase == "train" else val_dataloader

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == "train":
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

            print(f"{phase} Loss: {epoch_loss:.4f}")
            
        scheduler.step()
            

    return model


model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=25)



def plot_training(epoch):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    filename = f"outputs/logs/training_curve_epoch_{epoch}_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

    print(f"Training curve saved: {filename}")


plot_training(epoch=25)

def save_model_with_timestamp(model, directory="outputs/checkpoints"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(directory, f"resnet_cat_age_{timestamp}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"モデルが正常に保存されました: {checkpoint_path}")

save_model_with_timestamp(model_ft)

print("トレーニング完了！")
