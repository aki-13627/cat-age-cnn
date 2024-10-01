import sys
sys.path.append('/Users/akihiro/cat-age-cnn')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data.dataloader import train_dataloader, val_dataloader
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_sizes = {
    "train": len(train_dataloader.dataset),
    "val": len(val_dataloader.dataset),
}

model_ft = models.resnet50(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

criterion = nn.MSELoss()
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)

train_loss_history = []
val_loss_history = []


def train_model(model, criterion, optimizer, num_epochs=25):
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

    return model


model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=25)


def plot_training():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("outputs/logs/training_curve.png")
    plt.show()


plot_training()

print("トレーニング完了！")
