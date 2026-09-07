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
# 入力画像のパス (存在を確認してください)
INPUT_IMAGE_PATH = "/Users/akihiro/cat-age-cnn/data/processed/00097_12_00_00_0.jpg"

# 出力する可視化画像のパス
OUTPUT_PLOT_PATH = "preprocessing_steps_visualization.png"

# リサイズサイズ
IMAGE_SIZE = (224, 224)

# データ拡張の可視化パターン数
NUM_AUGMENTS = 3

# 再現性のためのシード固定（任意。外すと毎回違う拡張結果になります）
# torch.manual_seed(42)

# ==========================================
# 変換定義とヘルパー関数
# ==========================================

# 正規化のパラメータ
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 学習用のデータ拡張 (元のコードから抜粋)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# NormalizeされたTensorを表示用に元に戻す関数
def denormalize(tensor):
    """
    NormalizeされたTensor (C, H, W) を逆変換して、
    表示可能な (H, W, C) のnumpy配列 (0.0-1.0) に戻す。
    """
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m) # 逆計算: (値 * std) + mean
    tensor = torch.clamp(tensor, 0, 1) # 値を0-1の範囲にクリップ
    img_np = tensor.permute(1, 2, 0).numpy() # (C,H,W) -> (H,W,C)
    return img_np

# ==========================================
# メイン処理
# ==========================================

def main():
    # 1. 画像の確認
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"エラー: 指定された画像が見つかりません: {INPUT_IMAGE_PATH}")
        return

    print(f"Processing image: {INPUT_IMAGE_PATH}")

    # --- Step 1: Original Image ---
    # OpenCVで読み込み (BGR)
    original_bgr = cv2.imread(INPUT_IMAGE_PATH)
    # 表示用にRGBに変換
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # --- Step 2: Preprocessing (Resize with OpenCV) ---
    # 最初のコードブロックの処理を再現
    resized_bgr = cv2.resize(original_bgr, IMAGE_SIZE)
    # 表示用にRGBに変換
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # --- Step 3: Augmentation (PyTorch transforms) ---
    # PyTorchのtransform入力用に、リサイズ後の画像をPIL Imageに変換
    pil_img = Image.fromarray(resized_rgb)

    augmented_images = []
    for i in range(NUM_AUGMENTS):
        print(f"Applying augmentation {i+1}/{NUM_AUGMENTS}...")
        transformed_tensor = train_transform(pil_img)
        
        
        denormalized_img = denormalize(transformed_tensor)
        augmented_images.append(denormalized_img)

    print("Creating visualization plot...")

    total_plots = NUM_AUGMENTS
    plt.figure(figsize=(3 * total_plots, 4 )) 

    for i in range(NUM_AUGMENTS):
        plt.subplot(1, total_plots, i + 1)
        plt.imshow(augmented_images[i])
        plt.axis('off')

    plt.tight_layout()
    
    # 保存
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches='tight')
    print(f"完了！可視化画像を保存しました: {OUTPUT_PLOT_PATH}")
    
    
    

if __name__ == "__main__":
    main()