import sys
# パスはご自身の環境に合わせて調整してください
sys.path.append('/Users/akihiro/cat-age-cnn')

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
import cv2
import math
from models.se_resnet import se_resnet50

# ==========================================
# ★設定
# ==========================================
MODEL_PATH = 'outputs/checkpoints/se_resnet50_ordinal_regression_20260106-174333.pth'
IMAGE_FOLDER = '/Users/akihiro/cat-age-cnn/data/processed-for-cnn'
CSV_FILE = '/Users/akihiro/cat-age-cnn/data/filename-age-split.csv'
LOG_FILE_PATH = 'outputs/evaluation_log_window_conf.txt'
GRID_OUTPUT_PATH = 'outputs/evaluation_grid_window_conf.png'

GRID_COLS = 5
MAX_IMAGES = 50 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 23
NUM_OUTPUTS = NUM_CLASSES - 1

# ==========================================
# Grad-CAM クラス定義 (±1歳確率対応版)
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
        probs = torch.sigmoid(output) # [1, 22] Output of P(y > k)
        
        # --- 年齢と確率分布の計算 ---
        # 予測年齢 (閾値0.5を超えた数)
        predicted_age = torch.sum(probs > 0.5).item()
        pred_idx = int(predicted_age)
        pred_idx = min(max(pred_idx, 0), NUM_CLASSES - 1) # 安全策

        # 確率分布(PMF)を計算: P(k) = P(>k-1) - P(>k)
        p_cumulative = probs.squeeze().detach().cpu()
        # [1.0, p0, p1, ..., p21, 0.0]
        p_extended = torch.cat([torch.tensor([1.0]), p_cumulative, torch.tensor([0.0])])
        
        # 個別の確率 (PMF)
        pmf = p_extended[:-1] - p_extended[1:] # shape [23]
        pmf = torch.clamp(pmf, min=0.0) # 念のためマイナスをカット

        # --- ±1歳の確率和 (Window Confidence) ---
        # 範囲: [pred-1, pred+1] (ただしインデックス外は除外)
        start_idx = max(0, pred_idx - 1)
        end_idx = min(NUM_CLASSES - 1, pred_idx + 1)
        
        # 該当範囲の確率を合計
        window_confidence = torch.sum(pmf[start_idx : end_idx + 1]).item()
        
        # 単体の確率も一応持っておく
        single_confidence = pmf[pred_idx].item()
        # ------------------------

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
            
        return heatmap, predicted_age, window_confidence

# ==========================================
# ユーティリティ関数
# ==========================================
def log_print(message):
    print(message)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def load_model():
    log_print(f"Loading model: {MODEL_PATH}")
    model = se_resnet50(num_classes=1000)
    num_ftrs = 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, NUM_OUTPUTS)
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    except Exception as e:
        log_print(f"Error: {e}")
        sys.exit(1)
    model = model.to(device)
    model.eval()
    return model

def create_overlay(img_pil, heatmap):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(heatmap, 0.4, img_cv, 0.6, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay)

def draw_text(img, actual, predicted, confidence):
    draw = ImageDraw.Draw(img)
    # 表示テキストを "Conf(±1): XX%" に変更
    text = f"Act:{actual} Pred:{predicted}\nConf(±1):{confidence:.0%}"
    
    bbox = draw.textbbox((10, 10), text)
    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill="black")
    draw.text((10, 10), text, fill="white")
    
    error = abs(actual - predicted)
    color = "green" if error == 0 else "yellow" if error <= 1 else "red"
    draw.rectangle([0, 0, img.width-1, img.height-1], outline=color, width=5)
    return img

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    if os.path.exists(LOG_FILE_PATH): os.remove(LOG_FILE_PATH)
    
    # 1. 準備
    model = load_model()
    grad_cam = GradCAM(model, model.layer4[-1])
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    df = pd.read_csv(CSV_FILE)
    test_df = df[df["split"] == "val"]
    target_files = list(set(test_df["filename"]))
    
    if MAX_IMAGES:
        target_files = target_files[:MAX_IMAGES]

    processed_images = []
    errors_list = []
    
    # ヘッダー出力
    log_print(f"評価開始: 対象 {len(target_files)} 枚")
    log_print("-" * 75)
    log_print(f"{'Filename':<30} | {'Act':<5} | {'Pred':<5} | {'Error':<5} | {'Conf(±1)'}")
    log_print("-" * 75)

    # 2. 推論ループ
    for i, img_name in enumerate(target_files):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        if not os.path.exists(img_path): continue
        
        actual_age = test_df[test_df["filename"] == img_name]["age"].values[0]
        
        raw_img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(raw_img).unsqueeze(0).to(device)
        
        # 変更点: confidence は ±1歳の合計確率
        heatmap, predicted_age, confidence = grad_cam(input_tensor)
        
        error = abs(actual_age - predicted_age)
        errors_list.append(error)
        
        log_print(f"{img_name:<30} | {actual_age:<5} | {predicted_age:<5} | {error:<5} | {confidence:.1%}")
        
        vis_img = transforms.CenterCrop(224)(transforms.Resize(256)(raw_img))
        overlay_img = create_overlay(vis_img, heatmap)
        
        final_img = draw_text(overlay_img, actual_age, predicted_age, confidence)
        processed_images.append(final_img)

    # 3. 統計情報の出力
    if errors_list:
        mean_error = np.mean(errors_list)
        std_error = np.std(errors_list)
        
        n_total = len(errors_list)
        n_exact = sum(1 for e in errors_list if e == 0)
        n_within_1 = sum(1 for e in errors_list if e <= 1)
        n_within_2 = sum(1 for e in errors_list if e <= 2)

        acc_exact = (n_exact / n_total) * 100
        acc_1 = (n_within_1 / n_total) * 100
        acc_2 = (n_within_2 / n_total) * 100

        log_print("\n" + "="*30)
        log_print(" 最終評価結果 (Summary)")
        log_print("="*30)
        log_print(f"平均絶対誤差 (MAE) : {mean_error:.2f} 歳")
        log_print(f"誤差の標準偏差     : {std_error:.2f}")
        log_print(f"完全一致率 (Acc±0) : {acc_exact:.1f}%")
        log_print(f"Acc (許容誤差 ±1歳): {acc_1:.1f}%")
        log_print(f"Acc (許容誤差 ±2歳): {acc_2:.1f}%")

    # 4. グリッド保存
    if processed_images:
        num_imgs = len(processed_images)
        rows = math.ceil(num_imgs / GRID_COLS)
        w, h = 224, 224
        grid_img = Image.new('RGB', (GRID_COLS * w, rows * h), color='white')
        
        for idx, img in enumerate(processed_images):
            r = idx // GRID_COLS
            c = idx % GRID_COLS
            grid_img.paste(img, (c * w, r * h))
            
        grid_img.save(GRID_OUTPUT_PATH)
        log_print("-" * 60)
        log_print(f"Grid image saved to: {GRID_OUTPUT_PATH}")
        log_print(f"Log saved to: {LOG_FILE_PATH}")