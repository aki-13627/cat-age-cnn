import sys

sys.path.append('/Users/akihiro/cat-age-cnn')

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.se_resnet import se_resnet50, load_pretrained_weights, SEBlock
from data.dataloader import train_dataloader, val_dataloader
import os
import matplotlib.pyplot as plt


device = torch.device("mps")

dataset_sizes = {
    "train": len(train_dataloader.dataset),
    "val": len(val_dataloader.dataset),
}


num_classes = 23
num_outputs = num_classes - 1


model_ft = se_resnet50(num_classes=1000)

model_ft = load_pretrained_weights(model_ft)


for param in model_ft.parameters():
    param.requires_grad = True

num_ftrs = 2048


 
model_ft.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, num_outputs) 
)

model_ft = model_ft.to(device)


criterion = nn.BCEWithLogitsLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.000005)
scheduler = StepLR(optimizer_ft, step_size=10, gamma=0.98)


train_loss_history = []
val_loss_history = []
train_mae_history = []
val_mae_history = []


def make_rank_label(batch_labels, num_outputs):
    batch_size = batch_labels.size(0)
    rank_labels = torch.zeros(batch_size, num_outputs, device=device)
    
    for i in range(batch_size):
        age = batch_labels[i].item()
        if age > 0:
            fill_cnt = min(age, num_outputs)
            rank_labels[i, :fill_cnt] = 1.0
            
    return rank_labels


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_mae = 0

            dataloader = train_dataloader if phase == "train" else val_dataloader
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                
                rank_labels = make_rank_label(labels, num_outputs)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    
                    
                    loss = criterion(outputs, rank_labels)
                    
                    probs = torch.sigmoid(outputs)
                    preds = torch.sum(probs > 0.5, dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                running_mae += torch.sum(torch.abs(preds.float() - labels.float()))

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


model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=100)


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

    filename = f"outputs/logs/training_curve_se_resnet50_ordinal_{timestamp}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Training curve saved: {filename}")

plot_training()


print("\n=== SEBlock Weights Check ===")
count = 0
for name, module in model_ft.named_modules():
    if isinstance(module, SEBlock):
        w = module.fc2.weight.data.cpu().numpy()
        if count < 3:
            print(f"SEBlock at '{name}': fc2.weight mean={w.mean():.4f}, std={w.std():.4f}")
        count += 1
print(f"Total SEBlocks checked: {count}")


def save_model_with_timestamp(model, directory="outputs/checkpoints"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(directory, f"se_resnet50_ordinal_regression_{timestamp}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"モデルが正常に保存されました: {checkpoint_path}")

save_model_with_timestamp(model_ft)
print("トレーニング完了！")