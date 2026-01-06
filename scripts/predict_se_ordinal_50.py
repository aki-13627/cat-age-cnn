import sys
# パスはご自身の環境に合わせて調整してください
sys.path.append('/Users/akihiro/cat-age-cnn')

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from models.se_resnet import se_resnet50  # カスタムモデルのインポート

# ==========================================
# ★設定: 学習済みのモデルパスを指定してください
# ==========================================
MODEL_PATH = 'outputs/checkpoints/se_resnet50_ordinal_regression_20260105-204440.pth'

# その他のパス設定
IMAGE_FOLDER = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'
CSV_FILE = '/Users/akihiro/cat-age-cnn/data/filename-age-split.csv'
LOG_FILE_PATH = 'outputs/evaluation_log_se_ordinal.txt'

# デバイス設定
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Ordinal Regression設定
NUM_CLASSES = 23
NUM_OUTPUTS = NUM_CLASSES - 1  # 22個の閾値

# === ログ出力用関数 ===
def log_print(message):
    print(message)
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# === モデルの準備 ===
def load_model():
    log_print(f"モデルを読み込んでいます: {MODEL_PATH}")
    
    # 1. アーキテクチャの定義 (学習時と全く同じにする)
    model = se_resnet50(num_classes=1000) # 初期化
    
    # 2. 出力層の書き換え (Ordinal Regression仕様)
    # ResNet50のFC入力は2048固定
    num_ftrs = 2048
    
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), # 学習時と同じ設定
        nn.Linear(num_ftrs, NUM_OUTPUTS)
    )
    
    # 3. 重みのロード
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        log_print(f"エラー: モデルファイルが見つかりません -> {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        log_print(f"エラー: 重みのロードに失敗しました。\n詳細: {e}")
        sys.exit(1)

    model = model.to(device)
    model.eval()
    return model

# === 前処理 (Validation時と同じにする) ===
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 推論関数 (Ordinal Regression) ===
def predict_image(img_path, model):
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # --- Ordinal Regression ロジック ---
            # 1. Sigmoidで確率(0~1)に変換
            probs = torch.sigmoid(outputs)
            
            # 2. 閾値0.5を超えている要素の数をカウントして年齢とする
            # output: [0.99, 0.95, 0.80, 0.10, ...] -> 3つ超えている -> 3歳
            predicted_age = torch.sum(probs > 0.5).item()
            
            return predicted_age
            
    except Exception as e:
        log_print(f"画像処理エラー: {img_path}\n{e}")
        return None

# === メイン処理 ===
if __name__ == "__main__":
    # ログファイルのリセット
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)

    # モデルロード
    model = load_model()

    # データ読み込み
    df = pd.read_csv(CSV_FILE)
    # 検証データ(val)のみを対象にする場合
    test_df = df[df["split"] == "val"]
    
    age_dict = dict(zip(test_df["filename"], test_df["age"]))
    target_filenames = set(test_df["filename"])

    results = []
    errors = []
    correct_prediction = 0   # 誤差1以内 (従来指標)
    exact_match = 0          # 完全一致
    correct_prediction_2 = 0 # 誤差2以内

    log_print(f"評価開始: 対象枚数 {len(target_filenames)} 枚")
    log_print("-" * 30)

    for img_name in target_filenames:
        if not (img_name.lower().endswith(".jpg") or img_name.lower().endswith(".png")):
            continue

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        
        if not os.path.exists(img_path):
            log_print(f"警告: 画像が見つかりません -> {img_path}")
            continue

        actual_age = age_dict.get(img_name)
        predicted_age = predict_image(img_path, model)

        if predicted_age is not None and actual_age is not None:
            error = abs(predicted_age - actual_age)
            errors.append(error)
            results.append([img_name, actual_age, predicted_age, error])

            if error == 0:
                exact_match += 1
            if error <= 1:
                correct_prediction += 1
            if error <= 2:
                correct_prediction_2 += 1

            log_print(f"{img_name} - 実年齢: {actual_age:2d}, 推定: {predicted_age:2d}, 誤差: {error}")

    # === 結果集計 ===
    if len(errors) > 0:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        acc_exact = (exact_match / len(errors)) * 100
        acc_1 = (correct_prediction / len(errors)) * 100
        acc_2 = (correct_prediction_2 / len(errors)) * 100

        log_print("\n" + "="*30)
        log_print(" SE-ResNet50 (Ordinal) 評価結果")
        log_print("="*30)
        log_print(f"平均絶対誤差 (MAE) : {mean_error:.2f} 歳")
        log_print(f"誤差の標準偏差     : {std_error:.2f}")
        log_print(f"完全一致率 (Acc±0) : {acc_exact:.1f}%")
        log_print(f"Acc (許容誤差 ±1歳): {acc_1:.1f}%")
        log_print(f"Acc (許容誤差 ±2歳): {acc_2:.1f}%")

        # CSV保存
        results_df = pd.DataFrame(results, columns=["filename", "actual_age", "predicted_age", "error"])
        save_path = "outputs/prediction_results_se_ordinal.csv"
        results_df.to_csv(save_path, index=False)
        log_print(f"\n詳細なCSV結果を '{save_path}' に保存しました！")
    else:
        log_print("評価対象のデータがありませんでした。")