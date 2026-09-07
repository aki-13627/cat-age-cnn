import sys
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd # 表計算用にpandasを追加

# パスはご自身の環境に合わせて調整してください
sys.path.append('/Users/akihiro/cat-age-cnn')
from models.se_resnet import se_resnet50

# ==========================================
# ★設定
# ==========================================
# 評価対象の画像フォルダ
IMAGE_DIR = "data/any_to_predict"

# 学習済みモデルのパス
MODEL_PATH = 'outputs/checkpoints/se_resnet50_ordinal_regression_20260105-204440.pth'

# 出力画像の保存先
OUTPUT_FILENAME = "outputs/all_gradcam_any_results.png"
# ログファイルの保存先 (拡張子を変えて設定)
LOG_FILENAME = "outputs/all_gradcam_any_results.txt"

# デバイス設定
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Ordinal Regression設定
NUM_CLASSES = 23
NUM_OUTPUTS = NUM_CLASSES - 1  # 22

# グリッド表示の列数（横に何枚並べるか）
GRID_COLS = 5

# ==========================================
# Grad-CAM用のクラス定義
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        output = self.model(x)
        probs = torch.sigmoid(output)
        predicted_age = torch.sum(probs > 0.5).item()
        
        self.model.zero_grad()
        if predicted_age > 0:
            target_index = int(predicted_age - 1)
        else:
            target_index = 0
            
        one_hot = torch.zeros_like(output)
        one_hot[0][target_index] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0].detach().cpu().numpy()
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i].item()

        heatmap = np.mean(activation, axis=0)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, predicted_age

# ==========================================
# 関数定義
# ==========================================
def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    model = se_resnet50(num_classes=1000)
    num_ftrs = 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, NUM_OUTPUTS)
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model = model.to(device)
    model.eval()
    return model

def preprocess_image_for_gradcam(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    vis_image = transforms.Resize((224, 224))(raw_image)
    
    normalize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = normalize_transform(raw_image).unsqueeze(0).to(device)
    return vis_image, input_tensor

def show_cam_on_image(img, mask):
    img = np.array(img)
    img = img.astype(np.float32) / 255
    heatmap = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def extract_age_from_filename(filename):
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        age_part = base.split('_')[-1]
        return int(age_part)
    except ValueError:
        return None

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    # 1. モデルロードとGrad-CAM準備
    model = load_model()
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # 2. 画像ファイルリスト取得
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_files.sort()

    num_images = len(image_files)
    if num_images == 0:
        print(f"画像が見つかりませんでした: {IMAGE_DIR}")
        exit()

    print(f"対象枚数: {num_images} 枚")
    print("Grad-CAM画像を生成中...")

    # 3. 全画像のGrad-CAM生成ループ
    results = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        true_age = extract_age_from_filename(filename)
        if true_age is None: continue

        # 推論とGrad-CAM
        vis_image, input_tensor = preprocess_image_for_gradcam(img_path)
        mask, predicted_age = grad_cam(input_tensor)
        
        # オーバーレイ画像作成
        overlay_img = show_cam_on_image(vis_image, mask)
        
        # 結果リストへの格納
        results.append({
            'filename': filename,
            'true_age': true_age,
            'pred_age': predicted_age,
            'overlay_img': overlay_img, # 画像用
            'diff': abs(true_age - predicted_age) # 集計用
        })

    # ==========================================
    # 4. ログ出力と指標計算 (追加部分)
    # ==========================================
    if results:
        df = pd.DataFrame(results)
        
        # 指標計算
        mae = df['diff'].mean()
        std_dev = df['diff'].std()
        acc_0 = (df['diff'] == 0).mean() * 100
        acc_1 = (df['diff'] <= 1).mean() * 100
        acc_2 = (df['diff'] <= 2).mean() * 100

        # ログ用テキストの作成
        log_content = []
        log_content.append(f"Model: {MODEL_PATH}")
        log_content.append(f"Date: {pd.Timestamp.now()}")
        log_content.append("-" * 50)
        log_content.append("\n### 詳細データ\n")
        log_content.append("| ファイル名 | 実年齢 | 推定 | 誤差 |")
        log_content.append("|---|---|---|---|")
        
        for index, row in df.iterrows():
            log_content.append(f"| {row['filename']} | {row['true_age']} | {row['pred_age']} | {row['diff']} |")

        log_content.append("\n<br>\n")
        log_content.append("### 評価結果サマリー\n")
        log_content.append("| 項目 | 値 |")
        log_content.append("|---|---|")
        log_content.append(f"| 平均絶対誤差 (MAE) | {mae:.2f} 歳 |")
        log_content.append(f"| 誤差の標準偏差 | {std_dev:.2f} |")
        log_content.append(f"| 完全一致率 (Acc±0) | {acc_0:.1f}% |")
        log_content.append(f"| Acc (許容誤差 ±1歳) | {acc_1:.1f}% |")
        log_content.append(f"| Acc (許容誤差 ±2歳) | {acc_2:.1f}% |")
        log_content.append(f"| 検証枚数 | {len(df)} 枚 |")

        log_text = "\n".join(log_content)

        # コンソール表示
        print(log_text)

        # ファイル保存
        try:
            os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)
            with open(LOG_FILENAME, 'w', encoding='utf-8') as f:
                f.write(log_text)
            print(f"\nログを保存しました: {LOG_FILENAME}")
        except Exception as e:
            print(f"ログ保存エラー: {e}")

    # ==========================================
    # 5. グリッド画像の生成と保存
    # ==========================================
    print("\n画像を結合して保存します...")

    grid_cols = GRID_COLS
    grid_rows = math.ceil(num_images / grid_cols)

    plt.figure(figsize=(4 * grid_cols, 4 * grid_rows))

    for i, res in enumerate(results):
        plt.subplot(grid_rows, grid_cols, i + 1)
        plt.imshow(res['overlay_img'])
        
        short_filename = res['filename'].split('_')[0] + "..."
        title_text = f"True: {res['true_age']}, Pred: {res['pred_age']}\n{short_filename}"
        
        color = 'black'
        if res['diff'] >= 3:
            color = 'red'

        plt.title(title_text, fontsize=10, color=color)
        plt.axis('off')

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    plt.savefig(OUTPUT_FILENAME)
    print(f"結果画像を保存しました: {OUTPUT_FILENAME}")