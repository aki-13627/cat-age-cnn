import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import time

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.se_resnet import se_resnet18, load_pretrained_weights

# === モデルの定義と読み込み ===
device = torch.device('mps')
num_classes = 23

model_ft = se_resnet18(num_classes=num_classes).to(device)
# 提供された新しいモデルパスを使用
model_path = "outputs/checkpoints/resnet18_cat_age_se_20250928-213450.pth"
model_ft.load_state_dict(torch.load(model_path))
model_ft.eval()

# === 前処理 ===
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === 画像1枚を推論する関数 ===
def predict_image(img_path, model, device):
    try:
        image = Image.open(img_path).convert("RGB")
        image = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)
            return predicted_class.item()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# === メイン処理 ===
if __name__ == "__main__":
    image_folder = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'
    csv_file = '/Users/akihiro/cat-age-cnn/data/filename-age-split.csv'

    df = pd.read_csv(csv_file)
    test_df = df[df["split"] == "test"].copy()
    
    # 実際にあるファイルのみに絞り込む
    all_files_in_dir = set(os.listdir(image_folder))
    test_df = test_df[test_df['filename'].isin(all_files_in_dir)]
    
    age_dict = dict(zip(test_df["filename"], test_df["age"]))
    target_filenames = sorted(list(set(test_df["filename"])))

    results = []
    errors = []
    correct_prediction_1 = 0
    correct_prediction_2 = 0

    print("=== 予測を開始します ===")
    for img_name in target_filenames:
        img_path = os.path.join(image_folder, img_name)
        actual_age = age_dict.get(img_name)
        
        predicted_age = predict_image(img_path, model_ft, device)

        if predicted_age is not None and actual_age is not None:
            error = abs(predicted_age - actual_age)
            errors.append(error)
            results.append([img_name, actual_age, predicted_age, error])
            
            # 許容誤差の計算
            if error <= 1:
                correct_prediction_1 += 1
            if error <= 2:
                correct_prediction_2 += 1
            
            print(f"ファイル: {img_name}, 実際: {actual_age}歳, 予測: {predicted_age}歳, 誤差: {error}")

    # === 結果集計 ===
    if errors:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        accuracy_1 = (correct_prediction_1 / len(errors)) * 100
        accuracy_2 = (correct_prediction_2 / len(errors)) * 100
    
        print("\n=== モデルの評価結果 ===")
        print(f"テストデータ数: {len(errors)}")
        print(f"平均絶対誤差 (MAE): {mean_error:.2f}")
        print(f"誤差の標準偏差: {std_error:.2f}")
        print(f"誤差1までを許容した場合の精度: {accuracy_1:.1f}%")
        print(f"誤差2までを許容した場合の精度: {accuracy_2:.1f}%")

        results_df = pd.DataFrame(results, columns=["filename", "actual_age", "predicted_age", "error"])
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_csv_path = f"outputs/prediction_results_se_{timestamp}.csv"
        results_df.to_csv(output_csv_path, index=False)
        print(f"結果を '{output_csv_path}' に保存しました！")
    else:
        print("テストデータがありませんでした。")