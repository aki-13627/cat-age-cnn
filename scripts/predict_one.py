from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================
# 入力画像のパスを指定
# =========================
image_path = "data/any_to_predict/スクリーンショット 2025-04-23 17.57.42.png"

# =========================
# モデルの定義と読み込み
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_classes = 23

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load('outputs/checkpoints/resnet18_cat_age_20250423-162621.pth'))
model = model.to(device)
model.eval()

# =========================
# 画像前処理の定義
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# 入力画像の読み込み・整形
# =========================
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# =========================
# 推論
# =========================
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# =========================
# 結果出力
# =========================
print(f"予測された年齢クラス: {predicted_class}")
