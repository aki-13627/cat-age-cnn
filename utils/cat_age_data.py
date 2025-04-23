import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import os


class CatAgeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None, split="train"):
        df = pd.read_csv(csv_file)

        # 指定されたsplit（train, val, test）のみを使用
        self.labels_df = df[df["split"] == split].reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx]["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        age = int(self.labels_df.iloc[idx]["age"])
        age_tensor = torch.tensor(age, dtype=torch.long)  # 分類用に long 型に変換

        return image, age_tensor
