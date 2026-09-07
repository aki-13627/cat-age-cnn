import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

# ==========================================
# ★設定
# ==========================================
INPUT_IMAGE_PATH = "/Users/akihiro/cat-age-cnn/data/processed/00097_12_00_00_0.jpg"
OUTPUT_PLOT_PATH = "val_preprocessing_steps.png"

# Validation用の設定値
RESIZE_SIZE = 256
CROP_SIZE = 224

# 正規化パラメータ
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ★日本語フォントの設定 (Mac用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# ==========================================
# 関数定義
# ==========================================

def denormalize(tensor):
    """NormalizeされたTensorを表示用画像(numpy array, 0-1)に戻す"""
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    img_np = tensor.permute(1, 2, 0).numpy()
    return img_np

def draw_crop_box(pil_img, crop_size):
    """画像上にセンタークロップの範囲を示す赤枠を描画する"""
    img_np = np.array(pil_img)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    h, w, _ = img_np.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size
    
    cv2.rectangle(img_np, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    return Image.fromarray(img_np)

# ==========================================
# メイン処理
# ==========================================

def main():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"エラー: 画像が見つかりません: {INPUT_IMAGE_PATH}")
        return

    print(f"Processing image: {INPUT_IMAGE_PATH}")

    # --- 1. 元画像の読み込み ---
    original_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    w1, h1 = original_pil.size

    # Step A: リサイズ
    resize_transform = transforms.Resize(RESIZE_SIZE)
    resized_pil = resize_transform(original_pil)
    w2, h2 = resized_pil.size
    
    # (可視化用) 赤枠付き画像
    resized_with_box_pil = draw_crop_box(resized_pil, CROP_SIZE)

    # Step B: 切り抜き
    crop_transform = transforms.CenterCrop(CROP_SIZE)
    cropped_pil = crop_transform(resized_pil)
    w3, h3 = cropped_pil.size

    # Step C & D: テンソル変換と正規化
    tensor_normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    final_tensor = tensor_normalize_transform(cropped_pil)
    
    # (可視化用) 逆正規化
    final_image_denormalized = denormalize(final_tensor)
    h4, w4, _ = final_image_denormalized.shape # numpy shape is H, W, C

    # ==========================================
    # 可視化設定
    # ==========================================
    plt.figure(figsize=(20, 6)) # 縦幅を少し広げてテキストスペースを確保

    # 共通のプロット設定関数
    def plot_image(ax, img, title, width, height, is_numpy=False):
        ax.imshow(img)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15) # タイトル大きく
        
        # 軸の目盛りを消す
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 画像の下にサイズを表示 (xlabelを利用)
        size_text = f"幅:{width}px  高さ:{height}px"
        ax.set_xlabel(size_text, fontsize=16, labelpad=10) # ラベル大きく

    # 1. 元画像
    ax1 = plt.subplot(1, 4, 1)
    plot_image(ax1, original_pil, "1. 元画像", w1, h1)

    # 2. リサイズ + クロップ枠
    ax2 = plt.subplot(1, 4, 2)
    plot_image(ax2, resized_with_box_pil, f"2. リサイズ ({RESIZE_SIZE})", w2, h2)
    
    # 3. クロップ結果
    ax3 = plt.subplot(1, 4, 3)
    plot_image(ax3, cropped_pil, "3. 切り抜き後 (CenterCrop)", w3, h3)


    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches='tight')
    print(f"完了！画像を保存しました: {OUTPUT_PLOT_PATH}")
    # plt.show()

if __name__ == "__main__":
    main()