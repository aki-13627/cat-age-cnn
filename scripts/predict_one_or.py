from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================
# 設定
# =========================
# ★ここに新しく学習したOrdinal Regressionモデルのパスを入れてください
MODEL_PATH = 'outputs/checkpoints/resnet50_ordinal_regression_20260105-194716.pth'
image_path = "data/any_to_predict/スクリーンショット 2025-04-23 17.57.42.png"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Ordinal Regression用のクラス定義
NUM_CLASSES = 23
NUM_OUTPUTS = NUM_CLASSES - 1  # 22個の閾値

# =========================
# モデルの定義と読み込み
# =========================
model = models.resnet50(weights=None)

# 出力層を22個に変更
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, NUM_OUTPUTS)
)

# 重みのロード
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
except FileNotFoundError:
    print(f"エラー: モデルファイルが見つかりません -> {MODEL_PATH}")
    exit()
except RuntimeError as e:
    print(f"エラー: 重みの形状が一致しません。Ordinal Regression用のモデルを指定していますか？\n詳細: {e}")
    exit()

model = model.to(device)
model.eval()

# =========================
# 画像前処理の定義
# =========================
# 学習時のValidationと同じ処理に合わせる（精度向上）
transform = transforms.Compose([
    transforms.Resize(256),       # 短辺を256に
    transforms.CenterCrop(224),   # 中央224を切り抜き
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# 入力画像の読み込み・整形
# =========================
try:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
except FileNotFoundError:
    print(f"エラー: 画像が見つかりません -> {image_path}")
    exit()

# =========================
# 推論 (Ordinal Regression Logic)
# =========================
with torch.no_grad():
    output = model(image_tensor)
    
    # 1. Sigmoid関数を通して確率(0.0~1.0)に変換
    probs = torch.sigmoid(output)
    
    # 2. 閾値0.5を超えている要素の数をカウントして年齢とする
    # output: [0.99, 0.95, 0.80, 0.10, ...] -> 3つ超えている -> 3歳
    predicted_age = torch.sum(probs > 0.5).item()

    # 詳細な確率を見たい場合用（デバッグ）
    # print(f"Raw probabilities: {probs.cpu().numpy()}")

# =========================
# 結果出力
# =========================
print(f"画像: {image_path}")
print(f"予測された年齢: {predicted_age} 歳")