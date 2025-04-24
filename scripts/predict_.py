import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import torch.nn as nn

# === モデルの定義と読み込み ===
device = torch.device('mps')
num_classes = 23

model_ft = models.resnet18(weights=None)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 23)  # ここは分類数に応じて
)
model_ft.load_state_dict(torch.load('outputs/checkpoints/resnet18_cat_age_20250423-162621.pth'))
model_ft = model_ft.to(device)
model_ft.eval()

# === 前処理 ===
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 画像1枚を推論する関数 ===
def predict_image(img_path, model):
    image = Image.open(img_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
        return predicted_class.item()

# === メイン処理 ===
if __name__ == "__main__":
    image_folder = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'
    csv_file = '/Users/akihiro/cat-age-cnn/data/filename-age-split.csv'

    df = pd.read_csv(csv_file)
    test_df = df
    age_dict = dict(zip(test_df["filename"], test_df["age"]))
    target_filenames = set(test_df["filename"])

    results = []
    errors = []
    correct_prediction = 0
    correct_prediction_2 = 0

    for img_name in target_filenames:
        print("処理中:", img_name)
        if not (img_name.endswith(".jpg") or img_name.endswith(".png")):
            continue

        img_path = os.path.join(image_folder, img_name)
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

    # === 結果集計 ===
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    accuracy = (correct_prediction / len(errors)) * 100
    accuracy_2 = (correct_prediction_2 / len(errors)) * 100

    print("\n=== モデルの評価結果 ===")
    print(f"平均誤差 (MAE): {mean_error:.2f}")
    print(f"誤差の標準偏差: {std_error:.2f}")
    print(f"誤差1までを許容した場合の正確性: {accuracy:.1f}%")
    print(f"誤差2までを許容した場合の正確性: {accuracy_2:.1f}%")

    results_df = pd.DataFrame(results, columns=["filename", "actual_age", "predicted_age", "error"])
    results_df.to_csv("outputs/prediction_results.csv", index=False)
    print("結果を 'outputs/prediction_results.csv' に保存しました！")
