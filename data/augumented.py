from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import torch

class AugmentedCatAgeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, split="train", num_augments=4):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        self.num_augments = num_augments

        # split に応じてフィルタリング
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        age = row['age']

        augmented_images = [
            self.transform(image) for _ in range(self.num_augments)
        ]

        # (num_augments, C, H, W)
        images_tensor = torch.stack(augmented_images)
        return images_tensor, age