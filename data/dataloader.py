from torchvision import transforms
from torch.utils.data import DataLoader
from utils.cat_age_data import CatAgeDataset

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


csv_file = '/Users/akihiro/cat-age-cnn/data/filename-age.csv'
img_dir = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'

cat_age_train_dataset = CatAgeDataset(csv_file=csv_file, img_dir=img_dir, transform=data_transforms['train'])
cat_age_val_dataset = CatAgeDataset(csv_file=csv_file, img_dir=img_dir, transform=data_transforms['val'])

train_dataloader = DataLoader(cat_age_train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader = DataLoader(cat_age_val_dataset, batch_size=32, shuffle=False, num_workers=0)

if __name__ == "__main__":
    for images, ages in train_dataloader:
        print(f"Train Batch of images: {images.size()}")
        print(f"Train Batch of ages: {ages}")
        
    for images, ages in val_dataloader:
        print(f"Val Batch of images: {images.size()}")
        print(f"Val Batch of ages: {ages}")
