import torch
from torchvision import models, transforms
from PIL import Image
import os

model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 1)

model_ft.load_state_dict(torch.load('outputs/checkpoints/resnet_cat_age.pth'))
model_ft.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(img_path, model):
    image = Image.open(img_path)
    image = data_transforms(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted_age = output.item()
        predicted_age = round(predicted_age)
        return predicted_age

if __name__ == "__main__":
    image_folder = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'
    
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            predicted_age = predict_image(img_path, model_ft)
            print(f"{img_name} の推定年齢: {predicted_age}")
