import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import torch.nn as nn

# === 設定 ===
MODEL_PATH = 'outputs/checkpoints/resnet50_ordinal_regression_20260105-202116.pth' 
IMAGE_FOLDER = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'
CSV_FILE = '/Users/akihiro/cat-age-cnn/data/filename-age-split.csv'

# ★追加: ログファイルの保存先
LOG_FILE_PATH = 'outputs/evaluation_log.txt'

device = torch.device('mps')

# クラス数定義 (Ordinal Regression用)
NUM_CLASSES = 23
NUM_OUTPUTS = NUM_CLASSES - 1 

# === ログ出力用関数の定義 ===
def log_print(message):
    """コンソールに表示しつつ、ファイルにも書き込む関数"""
    print(message)  # ターミナル表示
    
    # ディレクトリがなければ作成
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ファイルに追記モードで書き込み
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# === モデルの定義と読み込み ===
model_ft = models.resnet50(weights=None)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, NUM_OUTPUTS) 
)

model_ft.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
model_ft = model_ft.to(device)
model_ft.eval()

# === 前処理 ===
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 画像1枚を推論する関数 ===
def predict_image(img_path, model):
    image = Image.open(img_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)
        predicted_age = torch.sum(probs > 0.5).item()
        return predicted_age

# === メイン処理 ===
if __name__ == "__main__":
    # 実行前に古いログファイルを削除（新規作成したい場合）
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)

    df = pd.read_csv(CSV_FILE)
    test_df = df[df["split"] == "val"]
    age_dict = dict(zip(test_df["filename"], test_df["age"]))
    target_filenames = set(test_df["filename"])

    results = []
    errors = []
    correct_prediction = 0
    correct_prediction_2 = 0

    log_print(f"評価開始: 対象枚数 {len(target_filenames)} 枚")
    log_print(f"ログ保存先: {LOG_FILE_PATH}")

    for img_name in target_filenames:
        if not (img_name.lower().endswith(".jpg") or img_name.lower().endswith(".png")):
            continue

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        
        if not os.path.exists(img_path):
            log_print(f"警告: 画像が見つかりません -> {img_path}")
            continue

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

            # print を log_print に変更
            log_print(f"{img_name} - 実年齢: {actual_age:2d}, 推定: {predicted_age:2d}, 誤差: {error}")
        else:
            log_print(f"{img_name} の実際の年齢がCSVに見つかりません")

    if len(errors) > 0:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        accuracy = (correct_prediction / len(errors)) * 100
        accuracy_2 = (correct_prediction_2 / len(errors)) * 100

        # 結果表示も log_print に変更
        log_print("\n=== Ordinal Regression モデルの評価結果 ===")
        log_print(f"平均誤差 (MAE): {mean_error:.2f}")
        log_print(f"誤差の標準偏差: {std_error:.2f}")
        log_print(f"誤差1までを許容した場合の正確性: {accuracy:.1f}%")
        log_print(f"誤差2までを許容した場合の正確性: {accuracy_2:.1f}%")

        results_df = pd.DataFrame(results, columns=["filename", "actual_age", "predicted_age", "error"])
        save_path = "outputs/prediction_results_ordinal.csv"
        results_df.to_csv(save_path, index=False)
        log_print(f"詳細なCSV結果を '{save_path}' に保存しました！")
    else:
        log_print("評価対象のデータがありませんでした。パスを確認してください。")