from torchvision import transforms
from torch.utils.data import DataLoader
from utils.cat_age_data import CatAgeDataset

csv_file = '/Users/akihiro/cat-age-cnn/data/filename-age-split.csv'
img_dir = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'

from torchvision import transforms

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}


# データセットを分割読み込み
cat_age_train_dataset = CatAgeDataset(csv_file=csv_file, img_dir=img_dir, transform=data_transforms['train'], split="train")
cat_age_val_dataset = CatAgeDataset(csv_file=csv_file, img_dir=img_dir, transform=data_transforms['val'], split="val")

train_dataloader = DataLoader(cat_age_train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader = DataLoader(cat_age_val_dataset, batch_size=32, shuffle=False, num_workers=0)

if __name__ == "__main__":
    for images, ages in train_dataloader:
        print(f"Train Batch of images: {images.size()}")
        print(f"Train Batch of ages: {ages}")
        
    for images, ages in val_dataloader:
        print(f"Val Batch of images: {images.size()}")
        print(f"Val Batch of ages: {ages}")
