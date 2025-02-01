import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 1)

model_ft.load_state_dict(torch.load('outputs/checkpoints/resnet_cat_age_20250201-185504.pth'))
model_ft.eval()

device = torch.device('mps')
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
        predicted_age = round(output.item())
        return predicted_age

if __name__ == "__main__":
    image_folder = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'
    csv_file = '/Users/akihiro/cat-age-cnn/data/filename-age.csv'

    df = pd.read_csv(csv_file)
    age_dict = dict(zip(df["filename"], df["age"]))

    results = []
    errors = []
    correct_prediction = 0
    correct_prediction_2 = 0
    threshold = 1

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            actual_age = age_dict.get(img_name)
            predicted_age = predict_image(img_path, model_ft)

            if actual_age is not None:
                error = abs(predicted_age - actual_age)
                errors.append(error)
                results.append([img_name, actual_age, predicted_age, error])
                
                if error <= 1:
                    correct_prediction += 1
                if error <= 2:
                    correct_prediction_2 += 1
                print(f"{img_name} - 実際の年齢: {actual_age}, 推定年齢: {predicted_age}, 誤差: {error}")
            else:
                print(f"{img_name} の実際の年齢がCSVに見つかりません")
                
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    accuracy = (correct_prediction / len(errors)) * 100
    accuracy_2 = (correct_prediction_2 / len(errors)) * 100
    
    print("\n=== モデルの評価結果 ===")
    print(f"平均誤差 (MAE): {mean_error:.2f}")
    print(f"誤差の標準偏差: {std_error:.2f}")
    print(f"誤差1までを許容した場合の正確性: {accuracy}%")
    print(f"誤差2までを許容した場合の正確性: {accuracy_2}%")
    
    results_df = pd.DataFrame(results, columns=["filename", "actual_age", "predicted_age", "error"])
    results_df.to_csv("outputs/prediction_results.csv", index=False)
    print("結果を 'outputs/prediction_results.csv' に保存しました！")